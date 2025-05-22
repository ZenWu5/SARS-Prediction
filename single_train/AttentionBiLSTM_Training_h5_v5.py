import os
import gc
import math
import time
import json
import h5py
import hashlib
import tempfile
import logging
import argparse
import threading
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from matplotlib.font_manager import FontProperties

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

# 配置命令行参数
parser = argparse.ArgumentParser(description='使用带注意力的BiLSTM模型训练蛋白质序列嵌入数据')
parser.add_argument('--data_path', type=str, default='preprocess/ESM/output/sampled_output_esm_embeddings.h5',
                    help='h5数据文件路径')
parser.add_argument('--max_seq_len', type=int, default=None, 
                    help='最大序列长度，默认使用数据集中的最大长度')
parser.add_argument('--hidden_dim', type=int, default=128, 
                    help='LSTM隐藏层维度')
parser.add_argument('--num_layers', type=int, default=3, 
                    help='LSTM层数')
parser.add_argument('--dropout', type=float, default=0.1, 
                    help='Dropout比例')
parser.add_argument('--lr', type=float, default=0.005, 
                    help='学习率')
parser.add_argument('--batch_size', type=int, default=256, 
                    help='批次大小')
parser.add_argument('--epochs', type=int, default=100, 
                    help='最大训练轮数')
parser.add_argument('--patience', type=int, default=5, 
                    help='早停耐心值')
parser.add_argument('--k_folds', type=int, default=5, 
                    help='交叉验证折数')
parser.add_argument('--seed', type=int, default=42, 
                    help='随机种子')
parser.add_argument('--output_dir', type=str, default='ModelResults/BiLSTMResults_h5', 
                    help='结果输出目录')
parser.add_argument('--cache_size', type=int, default=10240,
                    help='压缩缓存最大样本数')
parser.add_argument('--cache_dir', type=str, default=None,
                    help='缓存目录，默认使用临时目录')
parser.add_argument('--workers', type=int, default=8,
                    help='数据加载器的工作线程数')
args = parser.parse_args()

# 配置plt字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 检查并设置设备
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger("bilstm_attention_training")

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

# 定义带注意力机制的BiLSTM模型
class AttentionBiLSTM(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=64, num_layers=2, dropout=0.2):
        super(AttentionBiLSTM, self).__init__()
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 全连接层和Dropout
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

        # 初始化LSTM权重
        self._pack_padded_sequence = pack_padded_sequence
        self._pad_packed_sequence = pad_packed_sequence
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.size()
        
        # 如果提供了掩码，使用 pack_padded_sequence 跳过填充部分
        if mask is not None:
            lengths = mask.sum(dim=1).to(dtype=torch.int32)
            
            # 使用 pack_padded_sequence 处理变长序列
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            
            # 根据长度排序批次（pack_padded 需要降序排列）
            lengths, sort_idx = lengths.sort(0, descending=True)
            x_sorted = x[sort_idx]
            
            # 打包序列
            packed_input = pack_padded_sequence(x_sorted, lengths.cpu(), batch_first=True)
            packed_output, _ = self.lstm(packed_input)
            
            # 解包序列
            lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=seq_len)
            
            # 恢复原始顺序
            _, unsort_idx = sort_idx.sort(0)
            lstm_out = lstm_out[unsort_idx]
        else:
            # 不使用掩码时直接处理
            lstm_out, _ = self.lstm(x)
        
        # 计算注意力权重
        attention_scores = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        
        # 如果有掩码，应用到注意力分数
        if mask is not None:
            attention_scores = attention_scores.masked_fill(
                mask.unsqueeze(-1) == 0, float('-inf')
            )
            
        # 使用softmax获取归一化的注意力权重
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # 使用注意力权重计算加权和
        context = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, hidden_dim*2]
        
        # 通过全连接层
        out = self.relu(self.fc1(context))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out, attention_weights

# 定义基础数据集类
class CacheManager:
    """进程安全的缓存管理器，用于存储和检索压缩的样本数据"""
    
    def __init__(self, cache_dir=None, max_items=2048):
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="protein_cache_")
        self.index_file = os.path.join(self.cache_dir, "cache_index.json")
        self.max_items = max_items
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 统计信息
        self.stats_file = os.path.join(self.cache_dir, "stats.json")
        self._init_stats()
        
        # 确保在启动时创建索引文件
        self._ensure_index_exists()
    
    def _init_stats(self):
        """初始化统计信息"""
        if not os.path.exists(self.stats_file):
            stats = {
                "hits": 0,
                "misses": 0,
                "total_requests": 0
            }
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f)
    
    def _ensure_index_exists(self):
        """确保索引文件存在"""
        if not os.path.exists(self.index_file):
            with open(self.index_file, 'w') as f:
                json.dump({}, f)
    
    def _update_stats(self, hit=True):
        """更新缓存统计信息"""
        try:
            # 读取当前统计信息
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    stats = json.load(f)
            else:
                stats = {"hits": 0, "misses": 0, "total_requests": 0}
            
            # 更新统计信息
            stats["total_requests"] += 1
            if hit:
                stats["hits"] += 1
            else:
                stats["misses"] += 1
            
            # 写回文件
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f)
        except Exception as e:
            # 如果更新失败，不中断主流程
            pass
    
    def get_cache_path(self, idx):
        """获取缓存文件的路径"""
        # 使用索引的哈希值作为文件名前缀
        filename = f"sample_{hashlib.md5(str(idx).encode()).hexdigest()}.npz"
        return os.path.join(self.cache_dir, filename)
    
    def has_item(self, idx):
        """检查缓存中是否存在某项"""
        # 从索引文件读取
        try:
            with open(self.index_file, 'r') as f:
                index = json.load(f)
            
            # 检查索引和文件是否都存在
            cache_path = self.get_cache_path(idx)
            return str(idx) in index and os.path.exists(cache_path)
        except Exception:
            return False
    
    def get_item(self, idx):
        """从缓存获取项"""
        try:
            cache_path = self.get_cache_path(idx)
            
            if not os.path.exists(cache_path):
                self._update_stats(hit=False)
                return None
                
            # 读取npz文件
            data = np.load(cache_path)
            proc = data['proc']
            mask = data['mask']
            target = data['target'].item()
            
            # 更新统计信息
            self._update_stats(hit=True)
            
            # 更新访问时间
            self._update_access_time(idx)
            
            return proc, mask, target
        except Exception as e:
            # 如果读取失败，删除损坏的缓存
            try:
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                # 从索引中删除
                self._remove_from_index(idx)
            except:
                pass
            
            self._update_stats(hit=False)
            return None
    
    def add_item(self, idx, proc, mask, target):
        """添加项到缓存"""
        try:
            # 检查是否需要清理
            self._cleanup_if_needed()
            
            cache_path = self.get_cache_path(idx)
            
            # 保存数据
            np.savez_compressed(
                cache_path, 
                proc=proc, 
                mask=mask, 
                target=np.array(target)
            )
            
            # 更新索引
            self._add_to_index(idx, time.time())
            
            return True
        except Exception as e:
            return False
    
    def _add_to_index(self, idx, access_time):
        """将项添加到索引中"""
        try:
            with open(self.index_file, 'r') as f:
                index = json.load(f)
            
            # 添加或更新
            index[str(idx)] = access_time
            
            with open(self.index_file, 'w') as f:
                json.dump(index, f)
        except Exception:
            pass
    
    def _remove_from_index(self, idx):
        """从索引中删除项"""
        try:
            with open(self.index_file, 'r') as f:
                index = json.load(f)
            
            # 如果存在则删除
            if str(idx) in index:
                del index[str(idx)]
            
            with open(self.index_file, 'w') as f:
                json.dump(index, f)
        except Exception:
            pass
    
    def _update_access_time(self, idx):
        """更新访问时间"""
        self._add_to_index(idx, time.time())
    
    def _cleanup_if_needed(self):
        """如果缓存超过最大大小，清理最旧的项"""
        try:
            with open(self.index_file, 'r') as f:
                index = json.load(f)
            
            # 如果缓存项数量少于最大值，不执行清理
            if len(index) < self.max_items:
                return
            
            # 按访问时间排序，并移除最旧的项
            items = [(k, v) for k, v in index.items()]
            items.sort(key=lambda x: x[1])  # 按时间戳排序
            
            # 计算要删除的数量（移除1/5的缓存）
            remove_count = max(1, len(items) // 5)
            items_to_remove = items[:remove_count]
            
            # 移除文件和索引项
            for idx_str, _ in items_to_remove:
                try:
                    cache_path = self.get_cache_path(int(idx_str))
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
                    self._remove_from_index(int(idx_str))
                except:
                    pass
        except Exception:
            pass
    
    def clear(self):
        """清除所有缓存"""
        try:
            # 尝试清空索引
            with open(self.index_file, 'w') as f:
                json.dump({}, f)
                
            # 清空缓存目录中的所有npz文件
            for file in Path(self.cache_dir).glob("sample_*.npz"):
                try:
                    file.unlink()
                except:
                    pass
                
            # 重置统计信息
            self._init_stats()
        except Exception:
            pass
    
    def get_stats(self):
        """获取缓存统计信息"""
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    stats = json.load(f)
            else:
                stats = {"hits": 0, "misses": 0, "total_requests": 0}
            
            # 计算缓存项数量
            with open(self.index_file, 'r') as f:
                index = json.load(f)
            
            # 计算命中率
            total = stats.get("total_requests", 0)
            hit_rate = 0 if total == 0 else stats.get("hits", 0) / total * 100.0
            
            extended_stats = {
                "total_requests": stats.get("total_requests", 0),
                "cache_hits": stats.get("hits", 0),
                "cache_misses": stats.get("misses", 0),
                "hit_rate": hit_rate,
                "cache_items": len(index),
                "max_items": self.max_items
            }
            
            return extended_stats
        except Exception:
            # 如果发生错误，返回默认值
            return {
                "total_requests": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "hit_rate": 0,
                "cache_items": 0,
                "max_items": self.max_items
            }

# 创建一个单例的全局缓存管理器
global_cache_manager = None
def get_cache_manager(cache_dir=None, max_items=2048):
    """获取或创建全局缓存管理器实例"""
    global global_cache_manager
    if global_cache_manager is None:
        global_cache_manager = CacheManager(cache_dir=cache_dir, max_items=max_items)
    return global_cache_manager

# 改进的HDF5数据集类
class H5ProteinDataset(Dataset):
    """可序列化的蛋白质嵌入数据集类，适用于多进程环境"""
    
    def __init__(self, h5_path, max_seq_len=None, normalize=True, cache_dir=None, max_cache_items=2048):
        self.h5_path = h5_path if h5_path.endswith('.h5') else h5_path + '.h5'
        
        # 读取基本信息，然后关闭文件
        with h5py.File(self.h5_path, 'r') as f:
            self.total = int(f.attrs['total_samples'])
            self.emb_dim = int(f.attrs['embedding_dim'])
            # 只加载目标和diff_count到内存
            self.targets = f['mean_log10Ka'][:].astype(np.float32)      # (N,)
            self.diff_counts = f['diff_count'][:].astype(np.int16)       # (N,)
            # 统计所有序列长度
            self.seq_lengths = np.array([f['embeddings'][f'emb_{i}'].shape[0] for i in range(self.total)], dtype=np.int32)
        
        self.max_seq_len = max_seq_len or int(self.seq_lengths.max())
        
        # 缓存目录和配置
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="protein_cache_")
        self.max_cache_items = max_cache_items
        
        # 注意：缓存管理器在每个进程中单独初始化
        # 避免在这里初始化缓存管理器
        
        # 归一化参数
        self.normalize = normalize
        self.feature_mean = None
        self.feature_std = None
        
        if normalize:
            self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """计算所有嵌入向量的均值和标准差，用于标准化"""
        logging.info("计算嵌入向量的标准化统计量...")
        n_samples = min(1000, self.total)
        indices = np.random.choice(self.total, n_samples, replace=False)
        all_features = []
        with h5py.File(self.h5_path, 'r') as f:
            for idx in tqdm(indices, desc="采样特征统计"):
                emb = f['embeddings'][f'emb_{idx}'][:].astype(np.float32)
                all_features.append(emb)
        all_features = np.vstack(all_features)
        self.feature_mean = np.mean(all_features, axis=0)
        self.feature_std = np.std(all_features, axis=0) + 1e-8
        logging.info(f"特征归一化统计量计算完成: 均值范围[{self.feature_mean.min():.4f}, {self.feature_mean.max():.4f}], "
                    f"标准差范围[{self.feature_std.min():.4f}, {self.feature_std.max():.4f}]")
    
    def __len__(self):
        return self.total
    
    def __getitem__(self, idx):
        # 惰性初始化每个进程的缓存管理器
        cache_manager = get_cache_manager(self.cache_dir, self.max_cache_items)
        
        # 检查缓存
        cache_result = cache_manager.get_item(idx)
        
        if cache_result is not None:
            # 缓存命中
            proc, mask, target = cache_result
            return (
                torch.from_numpy(proc.copy()),
                torch.from_numpy(mask.copy()),
                torch.tensor(target, dtype=torch.float32)
            )
        
        # 缓存未命中，从文件加载
        with h5py.File(self.h5_path, 'r') as f:
            emb = f['embeddings'][f'emb_{idx}'][:].astype(np.float32)
        
        if self.normalize and self.feature_mean is not None:
            emb = (emb - self.feature_mean) / self.feature_std
        
        L = emb.shape[0]
        if L > self.max_seq_len:
            proc = emb[:self.max_seq_len]
            mask = np.ones(self.max_seq_len, dtype=bool)
        else:
            pad_len = self.max_seq_len - L
            proc = np.vstack([emb, np.zeros((pad_len, self.emb_dim), dtype=np.float32)])
            mask = np.concatenate([np.ones(L, dtype=bool), np.zeros(pad_len, dtype=bool)])
        
        target = float(self.targets[idx])
        
        # 后台添加到缓存
        def add_to_cache_background():
            cache_manager.add_item(idx, proc, mask, target)
        
        # 启动后台线程进行缓存，不影响主线程
        threading.Thread(target=add_to_cache_background).start()
        
        return (
            torch.from_numpy(proc),
            torch.from_numpy(mask),
            torch.tensor(target, dtype=torch.float32)
        )
    
    def get_diff_counts(self):
        """返回所有样本的diff_count（int16）"""
        return self.diff_counts
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        cache_manager = get_cache_manager(self.cache_dir, self.max_cache_items)
        return cache_manager.get_stats()
    
    def clear_cache(self):
        """清除所有缓存"""
        cache_manager = get_cache_manager(self.cache_dir, self.max_cache_items)
        cache_manager.clear()
        logging.info(f"已清理缓存目录: {self.cache_dir}")
    
    def preload_samples(self, indices, desc="预加载样本"):
        """预加载指定索引的样本到缓存"""
        logging.info(f"预加载 {len(indices)} 个样本到缓存...")
        for idx in tqdm(indices, desc=desc):
            # 只获取一次，触发缓存
            _ = self[idx]        
# 创建优化的数据加载器
def create_data_loader(dataset, batch_size, is_train=True, num_workers=8):
    """创建高效的数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,         # 工作线程数
        pin_memory=True,                 # 使用固定内存
        persistent_workers=True,         # 保持工作线程存活
        prefetch_factor=2,               # 每个工作线程预取批次数
        drop_last=is_train               # 训练时可丢弃不完整批次
    )

# 训练单个模型
def train_model(train_loader, val_loader, model_params, training_params):
    """
    使用DataLoader训练模型，集成 AMP 和 TF32 支持，以及序列打包优化
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 获取输入维度并创建模型
    for x_batch, _, _ in train_loader:
        input_dim = x_batch.shape[2]
        break

    model = AttentionBiLSTM(
        input_dim=input_dim,
        hidden_dim=model_params['hidden_dim'],
        num_layers=model_params['num_layers'],
        dropout=model_params['dropout']
    ).to(device)
    
    # PyTorch 2.0+ 支持JIT编译加速
    if False and hasattr(torch, 'compile'):  # 暂时禁用 torch.compile
        logger.info("使用 torch.compile 加速模型")
        model = torch.compile(model, mode="reduce-overhead")
        
    # 启用 cudnn 基准测试找到最快算法
    torch.backends.cudnn.benchmark = True

    optimizer = Adam(model.parameters(), lr=training_params['lr'])
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=True)

    # 定义预热+余弦退火学习率策略
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # 线性预热
        else:
            # 余弦退火
            min_lr_factor = 1e-3
            total_epochs = training_params['epochs']
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr_factor + (1 - min_lr_factor) * 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = training_params['patience']

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse': [],
        'val_mae': [],
        'val_r2': [],
        'val_pearson': [],
        'data_times': [],
        'compute_times': [],
        'cache_hit_rates': []
    }

    # 创建进度条对象
    epoch_pbar = tqdm(range(training_params['epochs']), desc="训练进度")
    
    for epoch in epoch_pbar:
        model.train()
        train_losses = []
        batch_times = []
        data_times = []
        compute_times = []
        
        data_start = time.time()
        
        for X_batch, mask_batch, y_batch in train_loader:
            data_time = time.time() - data_start
            data_times.append(data_time)
            
            compute_start = time.time()
            
            # 异步转移数据到GPU
            X_batch = X_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            # 混合精度计算
            with autocast(device_type='cuda', dtype=torch.float16):
                y_pred, _ = model(X_batch, mask_batch)
                y_pred = y_pred.squeeze()
                loss = criterion(y_pred, y_batch)

            # 优化步骤
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            compute_time = time.time() - compute_start
            compute_times.append(compute_time)
            
            train_losses.append(loss.item())
            
            # 准备下一批数据的计时
            data_start = time.time()

        # 计算平均时间指标
        avg_data_time = sum(data_times) / len(data_times) if data_times else 0
        avg_compute_time = sum(compute_times) / len(compute_times) if compute_times else 0
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        history['train_loss'].append(avg_train_loss)
        history['data_times'].append(avg_data_time)
        history['compute_times'].append(avg_compute_time)

        # 验证阶段
        model.eval()
        val_losses, val_preds, val_targets = [], [], []

        with torch.no_grad():
            for X_val_batch, mask_val_batch, y_val_batch in val_loader:
                X_val_batch = X_val_batch.to(device, non_blocking=True)
                mask_val_batch = mask_val_batch.to(device, non_blocking=True)
                y_val_batch = y_val_batch.to(device, non_blocking=True)

                with torch.amp.autocast(device_type='cuda',dtype=torch.float16):
                    val_pred, _ = model(X_val_batch, mask_val_batch)
                    val_pred = val_pred.squeeze()
                    val_loss = criterion(val_pred, y_val_batch)

                val_losses.append(val_loss.item())
                val_preds.append(val_pred.cpu().numpy())
                val_targets.append(y_val_batch.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        avg_val_loss = np.mean(val_losses)
        val_mse = mean_squared_error(val_targets, val_preds)
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_r2 = r2_score(val_targets, val_preds)
        val_pearson, _ = pearsonr(val_targets, val_preds)

        # 获取缓存统计
        hit_rate = 0
        if hasattr(train_loader.dataset, 'get_cache_stats'):
            cache_stats = train_loader.dataset.get_cache_stats()
            hit_rate = cache_stats.get('hit_rate', 0)
            history['cache_hit_rates'].append(hit_rate)
        
        scheduler.step()
        
        # 更新进度条
        epoch_pbar.set_postfix({
            'train_loss': f"{avg_train_loss:.4f}",
            'val_loss': f"{avg_val_loss:.4f}",
            'val_r2': f"{val_r2:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
            'data_t': f"{avg_data_time:.3f}s",
            'comp_t': f"{avg_compute_time:.3f}s",
            'cache_hit': f"{hit_rate:.1f}%"
        })

        # 详细日志
        logger.info(f"Epoch {epoch + 1}: 训练损失={avg_train_loss:.6f}, 验证损失={avg_val_loss:.6f}, "
                   f"MSE={val_mse:.6f}, MAE={val_mae:.6f}, R²={val_r2:.6f}, Pearson={val_pearson:.6f}, "
                   f"缓存命中率={hit_rate:.2f}%")
        logger.info(f"数据加载时间={avg_data_time:.4f}s, 计算时间={avg_compute_time:.4f}s")

        history['val_loss'].append(avg_val_loss)
        history['val_mse'].append(val_mse)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)
        history['val_pearson'].append(val_pearson)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
            logger.info("✓ 新的最佳模型!")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"早停触发！{patience}个epoch没有改善")
            break

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    validation_results = {
        'loss': best_val_loss,
        'mse': val_mse,
        'mae': val_mae,
        'r2': val_r2,
        'pearson': val_pearson
    }

    return model, validation_results, history

# 可视化数据加载器中的注意力权重
def visualize_attention(model, data_loader, n_samples=5):
    """
    从数据加载器中选择样本，可视化注意力权重
    """
    model.eval()
    
    # 收集一些样本及其预测和注意力权重
    samples = []
    for X_batch, mask_batch, y_batch in data_loader:
        if len(samples) >= n_samples:
            break
            
        X_batch = X_batch.to(device, non_blocking=True)
        mask_batch = mask_batch.to(device, non_blocking=True)
        
        with torch.no_grad():
            pred, attention = model(X_batch, mask_batch)
            
        for i in range(min(len(X_batch), n_samples - len(samples))):
            # 获取真实序列长度（非填充部分）
            seq_len = mask_batch[i].sum().item()
            
            samples.append({
                'x': X_batch[i].cpu().numpy(),
                'mask': mask_batch[i].cpu().numpy(),
                'y_true': y_batch[i].item(),
                'y_pred': pred[i].item(),
                'attention': attention[i, :seq_len, 0].cpu().numpy(),
                'seq_len': seq_len
            })
    
    # 创建可视化
    plt.figure(figsize=(12, 4 * n_samples))
    
    for i, sample in enumerate(samples):
        # 创建子图
        plt.subplot(n_samples, 1, i + 1)
        
        # 绘制注意力权重
        plt.bar(range(sample['seq_len']), sample['attention'])
        plt.xlabel('序列位置')
        plt.ylabel('注意力权重')
        
        # 添加真实值和预测值
        plt.title(f'样本 #{i}: 真实值={sample["y_true"]:.4f}, 预测值={sample["y_pred"]:.4f}')
    
    plt.tight_layout()
    return plt.gcf()

# 绘制训练历史
def plot_training_history(histories, best_fold=None):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(16, 15))
    
    # 绘制每个折叠的训练和验证损失
    plt.subplot(4, 2, 1)
    for i, history in enumerate(histories):
        plt.plot(history['train_loss'], '--', label=f'Fold {i+1} 训练')
        plt.plot(history['val_loss'], label=f'Fold {i+1} 验证')
    plt.title('训练和验证损失')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制每个折叠的验证MSE
    plt.subplot(4, 2, 2)
    for i, history in enumerate(histories):
        style = '-' if (best_fold is not None and i == best_fold) else '--'
        plt.plot(history['val_mse'], style, label=f'Fold {i+1}' + (' (最佳)' if i == best_fold else ''))
    plt.title('验证MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    # 绘制每个折叠的验证R2
    plt.subplot(4, 2, 3)
    for i, history in enumerate(histories):
        style = '-' if (best_fold is not None and i == best_fold) else '--'
        plt.plot(history['val_r2'], style, label=f'Fold {i+1}' + (' (最佳)' if i == best_fold else ''))
    plt.title('验证R²')
    plt.xlabel('Epochs')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)
    
    # 绘制每个折叠的验证Pearson相关系数
    plt.subplot(4, 2, 4)
    for i, history in enumerate(histories):
        style = '-' if (best_fold is not None and i == best_fold) else '--'
        plt.plot(history['val_pearson'], style, label=f'Fold {i+1}' + (' (最佳)' if i == best_fold else ''))
    plt.title('验证Pearson相关系数')
    plt.xlabel('Epochs')
    plt.ylabel('Pearson')
    plt.legend()
    plt.grid(True)
    
    # 绘制数据加载时间
    plt.subplot(4, 2, 5)
    for i, history in enumerate(histories):
        if 'data_times' in history and history['data_times']:
            plt.plot(history['data_times'], label=f'Fold {i+1}')
    plt.title('数据加载时间 (每个批次)')
    plt.xlabel('Epochs')
    plt.ylabel('时间 (秒)')
    plt.legend()
    plt.grid(True)
    
    # 绘制计算时间
    plt.subplot(4, 2, 6)
    for i, history in enumerate(histories):
        if 'compute_times' in history and history['compute_times']:
            plt.plot(history['compute_times'], label=f'Fold {i+1}')
    plt.title('计算时间 (每个批次)')
    plt.xlabel('Epochs')
    plt.ylabel('时间 (秒)')
    plt.legend()
    plt.grid(True)
    
    # 绘制缓存命中率
    plt.subplot(4, 2, 7)
    for i, history in enumerate(histories):
        if 'cache_hit_rates' in history and history['cache_hit_rates']:
            plt.plot(history['cache_hit_rates'], label=f'Fold {i+1}')
    plt.title('缓存命中率 (%)')
    plt.xlabel('Epochs')
    plt.ylabel('命中率')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    return plt.gcf()

# 评估模型（使用数据加载器）
def evaluate_model(model, test_loader, output_dir=None):
    """在测试集上评估模型性能"""
    model.eval()

    # 收集预测和真实值
    test_preds = []
    test_targets = []
    test_attentions = []
    
    for X_batch, mask_batch, y_batch in tqdm(test_loader, desc="评估模型"):
        X_batch = X_batch.to(device, non_blocking=True)
        mask_batch = mask_batch.to(device, non_blocking=True)
        
        with torch.no_grad():
            pred, attention = model(X_batch, mask_batch)
            test_preds.append(pred.cpu().numpy())
            test_attentions.append(attention.cpu().numpy())
            test_targets.append(y_batch.numpy())
    
    # 合并批次结果
    test_preds = np.concatenate(test_preds).squeeze()
    test_targets = np.concatenate(test_targets)
    
    # 计算测试集性能指标
    test_mse = mean_squared_error(test_targets, test_preds)
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_r2 = r2_score(test_targets, test_preds)
    test_pearson, _ = pearsonr(test_targets, test_preds)

    logger.info(f"测试集性能: MSE={test_mse:.6f}, MAE={test_mae:.6f}, R²={test_r2:.6f}, Pearson={test_pearson:.6f}")
    
    # 设置中文字体
    try:
        font = FontProperties(fname=r'C:\Windows\Fonts\simkai.ttf', size=10)
    except:
        font = FontProperties(size=10)

    # 绘制预测vs真实值散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(test_targets, test_preds, alpha=0.5)

    # 添加理想线
    min_val = min(min(test_targets), min(test_preds))
    max_val = max(max(test_targets), max(test_preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.xlabel('实际值', fontsize=12, fontproperties=font)
    plt.ylabel('预测值', fontsize=12, fontproperties=font)
    plt.title('BiLSTM+Attention模型预测结果', fontsize=14, fontproperties=font)
    plt.grid(True)

    # 添加性能指标文本
    text = f"MSE: {test_mse:.4f}\nMAE: {test_mae:.4f}\nR²: {test_r2:.4f}\nPearson: {test_pearson:.4f}"
    plt.annotate(text, xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # 保存图像
    if output_dir is not None:
        figtime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        fig_output_path = os.path.join(output_dir, f'bilstm_attention_predictions_{figtime}.png')
        plt.savefig(fig_output_path)
        logger.info(f"预测结果可视化已保存至 '{fig_output_path}'")

    return test_mse, test_mae, test_r2, test_pearson

# 监控GPU利用率
def monitor_gpu_utilization():
    """记录当前GPU利用率和内存使用情况"""
    if not torch.cuda.is_available():
        return "无GPU可用"
    
    try:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        
        report = f"GPU: {gpu_name}\n"
        report += f"已分配内存: {gpu_mem_allocated:.2f} GB\n"
        report += f"已保留内存: {gpu_mem_reserved:.2f} GB"
        
        # 尝试获取利用率（需要pynvml库）
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            report += f"\nGPU利用率: {util.gpu}%"
            pynvml.nvmlShutdown()
        except:
            pass
            
        return report
    except Exception as e:
        return f"获取GPU信息出错: {str(e)}"

# 主函数
def main():
    # 1. 解析命令行参数
    args = parser.parse_args()

    # 2. 设置随机种子
    set_seed(args.seed)
    logger.info(f"设置随机种子: {args.seed}")

    # 3. 启用 GPU 算力加速
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True  # 启用 TF32
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True         # 启用 cudnn 基准测试
    
    # 记录GPU信息
    gpu_info = monitor_gpu_utilization()
    logger.info(f"GPU信息:\n{gpu_info}")

    # 4. 创建结果输出目录
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"结果将保存至: {output_dir}")

    # 5. 构造压缩缓存数据集
    dataset = H5ProteinDataset(
        args.data_path, 
        max_seq_len=args.max_seq_len, 
        normalize=True,
        cache_dir=args.cache_dir,
        max_cache_items=args.cache_size
    )
    N = len(dataset)
    logger.info(f"HDF5 样本总数: {N}, pad 长度={dataset.max_seq_len}, emb_dim={dataset.emb_dim}")
    
    # 6. 划分训练/测试集（15% 测试）
    perm = np.random.permutation(N)
    n_test = int(0.15 * N)
    test_idx = perm[:n_test]
    train_val_idx = perm[n_test:]

    test_set = Subset(dataset, test_idx)
    train_val_set = Subset(dataset, train_val_idx)
    logger.info(f"切分完成：训练+验证={len(train_val_set)}, 测试={len(test_set)}")

    # 为测试集创建数据加载器
    test_loader = create_data_loader(test_set, args.batch_size, is_train=False, num_workers=args.workers)
    
    # 7. 准备模型参数和训练参数
    model_params = {
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout
    }
    
    training_params = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience
    }

    # 记录所有配置
    with open(os.path.join(output_dir, 'config.txt'), 'w', encoding='utf-8') as f:
        f.write(f"运行时间: {timestamp}\n")
        f.write(f"数据路径: {args.data_path}\n")
        f.write(f"最大序列长度: {dataset.max_seq_len}\n")
        f.write(f"压缩缓存样本数: {args.cache_size}\n")
        f.write(f"缓存目录: {dataset.cache_dir}\n")
        f.write(f"样本数: {N} (训练+验证: {len(train_val_set)}, 测试: {len(test_set)})\n")
        f.write(f"模型参数: {model_params}\n")
        f.write(f"训练参数: {training_params}\n")
        f.write(f"K折交叉验证: {args.k_folds}\n")
        f.write(f"随机种子: {args.seed}\n")
        f.write(f"GPU信息: {gpu_info}\n")
    
    # 8. K 折交叉验证
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    fold_results = []
    all_histories = []
    best_model, best_mse, best_fold = None, float('inf'), None

    for fold, (train_subidx, val_subidx) in enumerate(kf.split(train_val_idx), 1):
        logger.info(f"\n======= 第 {fold}/{args.k_folds} 折 =======")

        # 从训练验证集中获取当前折的索引
        fold_train_idx = [train_val_idx[i] for i in train_subidx]
        fold_val_idx = [train_val_idx[i] for i in val_subidx]
        
        # 创建子集
        train_subset = Subset(dataset, fold_train_idx)
        val_subset = Subset(dataset, fold_val_idx)
        
        # 创建数据加载器
        train_loader = create_data_loader(train_subset, args.batch_size, is_train=True, num_workers=args.workers)
        val_loader = create_data_loader(val_subset, args.batch_size, is_train=False, num_workers=args.workers)

        # 先预加载一部分高频样本（可选）
        if hasattr(dataset, 'preload_samples') and args.cache_size > 0:
            # 预加载前1000个训练样本
            preload_size = min(1000, len(fold_train_idx)) 
            preload_indices = [fold_train_idx[i] for i in range(preload_size)]
            dataset.preload_samples(preload_indices, desc=f"预加载折{fold}训练样本")
            
        # 训练模型
        model, val_res, history = train_model(
            train_loader, val_loader,
            model_params, training_params
        )
        fold_results.append(val_res)
        all_histories.append(history)

        # 显示缓存统计
        if hasattr(dataset, 'get_cache_stats'):
            cache_stats = dataset.get_cache_stats()
            logger.info(f"缓存统计: 命中率={cache_stats['hit_rate']:.2f}%, "
                      f"命中次数={cache_stats['cache_hits']}/{cache_stats['total_requests']}, "
                      f"缓存项={cache_stats['cache_items']}/{cache_stats['max_items']}")

        logger.info(f"第 {fold} 折 验证 MSE={val_res['mse']:.6f}")

        if val_res['mse'] < best_mse:
            best_mse = val_res['mse']
            best_model = model
            best_fold = fold - 1

        # 每折结束后清理缓存
        if hasattr(dataset, 'clear_cache'):
            dataset.clear_cache()

        # 清理内存
        del train_loader, val_loader, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 9. 保存训练历史可视化
    history_fig = plot_training_history(all_histories, best_fold)
    history_path = os.path.join(output_dir, f'history_{timestamp}.png')
    history_fig.savefig(history_path)
    logger.info(f"训练历史图已保存至: {history_path}")
    plt.close(history_fig)

    # 10. 在测试集上评估最佳模型
    test_mse, test_mae, test_r2, test_pearson = evaluate_model(best_model, test_loader, output_dir)
    
    # 11. 可视化注意力权重
    att_fig = visualize_attention(best_model, test_loader, n_samples=5)
    att_path = os.path.join(output_dir, f'attention_{timestamp}.png')
    att_fig.savefig(att_path)
    logger.info(f"注意力可视化已保存至: {att_path}")
    plt.close(att_fig)

    # 12. 保存模型与结果摘要
    model_path = os.path.join(output_dir, 'best_model.pt')
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'model_params': model_params,
        'input_dim': dataset.emb_dim,
        'max_seq_len': dataset.max_seq_len,
        'training_config': vars(args),
        'timestamp': timestamp
    }, model_path)
    logger.info(f"最佳模型已保存至: {model_path}")
    
    # 导出为 ONNX 格式（可选）
    try:
        dummy_input = (
            torch.randn(1, dataset.max_seq_len, dataset.emb_dim).to(device),
            torch.ones(1, dataset.max_seq_len, dtype=torch.bool).to(device)
        )
        onnx_path = os.path.join(output_dir, "model.onnx")
        torch.onnx.export(
            best_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            input_names=['input', 'mask'],
            output_names=['output', 'attention'],
        )
        logger.info(f"ONNX模型已导出至: {onnx_path}")
    except Exception as e:
        logger.warning(f"ONNX导出失败: {str(e)}")
    
    # 写入结果摘要
    summary_path = os.path.join(output_dir, f'results_summary_{timestamp}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("======= BiLSTM+Attention 训练结果摘要 =======\n")
        f.write(f"最佳折: {best_fold + 1}/{args.k_folds}\n")
        f.write(f"测试集性能: MSE={test_mse:.6f}, MAE={test_mae:.6f}, R²={test_r2:.6f}, Pearson={test_pearson:.6f}\n\n")
        f.write("各折性能:\n")
        for i, res in enumerate(fold_results):
            f.write(f"第 {i+1} 折: MSE={res['mse']:.6f}, MAE={res['mae']:.6f}, "
                    f"R²={res['r2']:.6f}, Pearson={res['pearson']:.6f}\n")
        
        # 添加缓存统计信息
        if hasattr(dataset, 'get_cache_stats'):
            cache_stats = dataset.get_cache_stats()
            f.write("\n缓存统计信息:\n")
            f.write(f"最终命中率: {cache_stats['hit_rate']:.2f}%\n")
            f.write(f"缓存命中次数: {cache_stats['cache_hits']}\n") 
            f.write(f"缓存未命中次数: {cache_stats['cache_misses']}\n")
            f.write(f"缓存项数: {cache_stats['cache_items']}/{cache_stats['max_items']}\n")
            f.write(f"缓存目录: {dataset.cache_dir}\n")

        # 添加GPU利用率信息
        f.write(f"\nGPU信息:\n{monitor_gpu_utilization()}\n")
    
    logger.info(f"结果摘要已保存至: {summary_path}")
    
    # 清理最终缓存
    if hasattr(dataset, 'close'):
        dataset.close()
    if hasattr(dataset, 'clear_cache'):
        dataset.clear_cache()
        logger.info("已清理所有缓存")


if __name__ == "__main__":
    main()

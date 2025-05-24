import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
import os
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import gc
import argparse
import math
import pandas as pd

# 配置命令行参数
parser = argparse.ArgumentParser(description='使用多头注意力神经网络模型训练ESMC蛋白质序列嵌入数据')
parser.add_argument('--data_path', type=str, default='preprocess/ESM/output/sampled_output_esm_embeddings.h5',
                    help='ESM嵌入数据h5文件路径')
parser.add_argument('--max_seq_len', type=int, default=201, 
                    help='最大序列长度')
parser.add_argument('--hidden_dim', type=int, default=256, 
                    help='隐藏层维度')
parser.add_argument('--num_layers', type=int, default=3, 
                    help='Transformer层数')
parser.add_argument('--num_heads', type=int, default=8, 
                    help='注意力头数')
parser.add_argument('--dropout', type=float, default=0.2, 
                    help='Dropout比例')
parser.add_argument('--lr', type=float, default=5e-4, 
                    help='学习率')
parser.add_argument('--batch_size', type=int, default=256, 
                    help='批次大小')
parser.add_argument('--epochs', type=int, default=100, 
                    help='最大训练轮数')
parser.add_argument('--patience', type=int, default=8, 
                    help='早停耐心值')
parser.add_argument('--k_folds', type=int, default=10,
                    help='交叉验证折数')
parser.add_argument('--pooling', type=str, default='attention', 
                    choices=['mean', 'max', 'attention', 'cls'],
                    help='序列级别池化方法')
parser.add_argument('--seed', type=int, default=1044, 
                    help='随机种子')
parser.add_argument('--output_dir', type=str, default='ModelResults/GraphAttentionNetwork', 
                    help='结果输出目录')
parser.add_argument('--debug', action='store_true',
                    help='调试模式，减少数据集大小和训练轮数')

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
logger = logging.getLogger("esmc_attention_training")


# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

# ==================== ESMC数据集类 ====================
class H5ESMCDataset(Dataset):
    """
    从HDF5文件加载ESMC蛋白质序列嵌入数据的Dataset类
    
    期望的HDF5数据结构:
    - embeddings/emb_{idx}: (L, 1152) ESMC嵌入矩阵，L为序列长度
    - targets: (N,) 目标值数组 (mean_log10Ka)
    - diff_count: (N,) 每个序列与标准序列的差异计数
    """
    def __init__(self, h5_path, max_seq_len=201, normalize=True, debug=False, debug_sample=100):
        self.h5_path = h5_path if h5_path.endswith('.h5') else h5_path + '.h5'
        self.max_seq_len = max_seq_len
        
        print(f"正在加载数据集: {self.h5_path}")
        
        with h5py.File(self.h5_path, 'r') as f:
            self.total = int(f.attrs['total_samples'])
            self.emb_dim = int(f.attrs.get('embedding_dim', 1152))
            
            # 加载目标值
            self.targets = f['mean_log10Ka'][:].astype(np.float32)
            self.total = min(self.total, len(self.targets))
            
            # 加载seq_lengths
            print("正在计算序列长度...")
            self.seq_lengths = np.array([f['embeddings'][f'emb_{i}'].shape[0] for i in tqdm(range(self.total), desc="读取序列长度")])
            
            # 加载diff_count数据
            self.diff_count = f['diff_count'][:self.total].astype(np.int32)
            
        if debug:
            print(f"Debug模式: 使用前{debug_sample}个样本")
            self.total = min(debug_sample, self.total)
            self.targets = self.targets[:self.total]
            self.seq_lengths = self.seq_lengths[:self.total]
            self.diff_count = self.diff_count[:self.total]
            
        self._file = None
        self.normalize = normalize
        self.feature_mean = None
        self.feature_std = None

        if self.normalize:
            self._compute_normalization_stats()

        # 输出统计信息
        unique_diffs, counts = np.unique(self.diff_count, return_counts=True)
        print(f"数据集初始化完成: {self.total}个样本, 嵌入维度: {self.emb_dim}")
        print(f"序列长度范围: [{self.seq_lengths.min()}, {self.seq_lengths.max()}]")
        print(f"目标值范围: [{self.targets.min():.4f}, {self.targets.max():.4f}]")
        print(f"diff_counts分布: {dict(zip(unique_diffs, counts))}")
    
    def _compute_normalization_stats(self):
        """计算嵌入向量的标准化统计量"""
        print("正在计算嵌入向量的标准化统计量...")
        
        # 使用更多样本以获得更稳定的统计量
        n_samples = min(1000, self.total)
        indices = np.random.choice(self.total, n_samples, replace=False)
        
        # 使用在线算法计算统计量，避免内存问题
        feature_sum = np.zeros(self.emb_dim, dtype=np.float64)
        feature_sum_sq = np.zeros(self.emb_dim, dtype=np.float64)
        total_positions = 0
        
        with h5py.File(self.h5_path, 'r') as f:
            for idx in tqdm(indices, desc="计算标准化统计量"):
                try:
                    emb = f['embeddings'][f'emb_{idx}'][:].astype(np.float32)  # (L, 1152)
                    
                    # 检查数据质量
                    if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
                        print(f"警告: 样本 {idx} 包含NaN或Inf值，跳过")
                        continue
                    
                    # 累加每个位置的特征
                    feature_sum += np.sum(emb, axis=0)  # 对序列长度维度求和
                    feature_sum_sq += np.sum(emb**2, axis=0)
                    total_positions += emb.shape[0]  # 累加序列长度
                    
                except Exception as e:
                    print(f"警告: 读取样本 {idx} 时出错: {e}")
                    continue
        
        if total_positions == 0:
            raise ValueError("无法计算标准化统计量：没有有效的样本数据")
        
        # 计算均值和标准差
        self.feature_mean = (feature_sum / total_positions).astype(np.float32)
        feature_variance = (feature_sum_sq / total_positions) - (self.feature_mean**2)
        self.feature_std = np.sqrt(np.maximum(feature_variance, 1e-8)).astype(np.float32)
        
        # 确保标准差不为零
        self.feature_std = np.maximum(self.feature_std, 1e-6)
        
        print(f"标准化统计量计算完成:")
        print(f"  - 使用 {n_samples} 个样本, {total_positions} 个位置")
        print(f"  - 特征均值范围: [{self.feature_mean.min():.6f}, {self.feature_mean.max():.6f}]")
        print(f"  - 特征标准差范围: [{self.feature_std.min():.6f}, {self.feature_std.max():.6f}]")
        
        # 检查是否有异常值
        if np.any(np.isnan(self.feature_mean)) or np.any(np.isnan(self.feature_std)):
            raise ValueError("标准化统计量中包含NaN值")
        
        if np.any(self.feature_std < 1e-6):
            n_small = np.sum(self.feature_std < 1e-6)
            print(f"警告: {n_small} 个特征的标准差过小 (<1e-6)")

    def __getitem__(self, idx):
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r', swmr=True)
        
        try:
            # 加载ESMC嵌入向量
            emb = self._file['embeddings'][f'emb_{idx}'][:].astype(np.float32)  # (L, 1152)
            
            # 应用标准化
            if self.normalize and self.feature_mean is not None:
                # 确保广播正确: (L, 1152) - (1152,) / (1152,) = (L, 1152)
                emb = (emb - self.feature_mean[None, :]) / self.feature_std[None, :]
                
                # 检查标准化后是否有异常值
                if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
                    print(f"警告: 样本 {idx} 标准化后包含NaN或Inf值")
                    # 用零填充异常值
                    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 获取真实序列长度
            L = emb.shape[0]
            
            # 处理序列长度：截断或填充
            if L > self.max_seq_len:
                proc_emb = emb[:self.max_seq_len]
                mask = np.ones(self.max_seq_len, dtype=bool)
                actual_len = self.max_seq_len
            else:
                pad_len = self.max_seq_len - L
                # 使用零填充（标准化后的零是有意义的）
                proc_emb = np.vstack([emb, np.zeros((pad_len, self.emb_dim), dtype=np.float32)])
                mask = np.concatenate([np.ones(L, dtype=bool), np.zeros(pad_len, dtype=bool)])
                actual_len = L
            
            target = float(self.targets[idx])
            diff_count = int(self.diff_count[idx])
            
            return (
                torch.from_numpy(proc_emb),           # (max_seq_len, 1152)
                torch.from_numpy(mask),               # (max_seq_len,)
                torch.tensor(target, dtype=torch.float32),
                torch.tensor(actual_len, dtype=torch.long),
                torch.tensor(diff_count, dtype=torch.long)
            )
            
        except Exception as e:
            print(f"错误: 加载样本 {idx} 失败: {e}")
            # 返回零填充的示例
            proc_emb = np.zeros((self.max_seq_len, self.emb_dim), dtype=np.float32)
            mask = np.zeros(self.max_seq_len, dtype=bool)
            return (
                torch.from_numpy(proc_emb),
                torch.from_numpy(mask),
                torch.tensor(0.0, dtype=torch.float32),
                torch.tensor(0, dtype=torch.long),
                torch.tensor(0, dtype=torch.long)
            )
    
    def __len__(self):
        return self.total
    
    def __del__(self):
        if hasattr(self, '_file') and self._file is not None:
            try:
                self._file.close()
            except:
                pass
    
    def get_diff_counts(self):
        """返回所有样本的diff_count"""
        return self.diff_count.copy()
    
    def get_normalization_stats(self):
        """返回标准化统计量"""
        return {
            'mean': self.feature_mean.copy() if self.feature_mean is not None else None,
            'std': self.feature_std.copy() if self.feature_std is not None else None
        }

# ==================== 多头注意力Transformer模型 ====================
class ESMCMultiHeadAttentionNetwork(nn.Module):
    """
    基于ESMC嵌入的多头注意力网络
    使用Transformer Encoder结构处理序列数据
    """
    def __init__(self, input_dim=1152, hidden_dim=256, num_layers=3, 
                 num_heads=8, dropout=0.2, pooling='attention', max_seq_len=201):
        super(ESMCMultiHeadAttentionNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.pooling = pooling
        self.max_seq_len = max_seq_len
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码（可选）
        self.positional_encoding = nn.Parameter(
            torch.randn(max_seq_len, hidden_dim) * 0.02
        )
        
        # Transformer Encoder层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 池化相关层
        if pooling == 'attention':
            # 可学习的注意力池化
            self.attention_pooling = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=max(1, num_heads//2),
                batch_first=True,
                dropout=dropout
            )
            self.pooling_query = nn.Parameter(torch.randn(1, hidden_dim))
        elif pooling == 'cls':
            # 类似BERT的CLS token
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # 输出层
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//4, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x, mask=None, seq_lengths=None):
        """
        x: (batch_size, seq_len, input_dim)
        mask: (batch_size, seq_len) - True for valid positions
        seq_lengths: (batch_size,) - actual sequence lengths
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
        
        # 添加位置编码
        if seq_len <= self.max_seq_len:
            x = x + self.positional_encoding[:seq_len].unsqueeze(0)
        
        # 处理CLS token
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, hidden_dim)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, seq_len+1, hidden_dim)
            if mask is not None:
                cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)  # (batch_size, seq_len+1)
        
        # 创建attention mask (对于padding位置)
        if mask is not None:
            # Transformer需要的mask格式：True表示需要mask的位置
            src_key_padding_mask = ~mask  # (batch_size, seq_len) 或 (batch_size, seq_len+1)
        else:
            src_key_padding_mask = None
        
        # 通过Transformer编码器
        encoded = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )  # (batch_size, seq_len[+1], hidden_dim)
        
        # 序列级别池化
        pooled_output, attention_weights = self._pool_sequence(
            encoded, mask, seq_lengths
        )
        
        # 层归一化和输出
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        output = self.classifier(pooled_output)  # (batch_size, 1)
        
        return output, attention_weights, encoded
    
    def _pool_sequence(self, encoded, mask, seq_lengths):
        """序列级别池化"""
        attention_weights = None
        
        if self.pooling == 'mean':
            # 平均池化（考虑mask）
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
                pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                pooled = encoded.mean(dim=1)
                
        elif self.pooling == 'max':
            # 最大池化
            if mask is not None:
                encoded_masked = encoded.clone()
                encoded_masked[~mask] = float('-inf')
                pooled = encoded_masked.max(dim=1)[0]
            else:
                pooled = encoded.max(dim=1)[0]
                
        elif self.pooling == 'cls':
            # 使用CLS token的表示
            pooled = encoded[:, 0, :]  # 第一个位置是CLS token
            
        elif self.pooling == 'attention':
            # 可学习的注意力池化
            batch_size = encoded.shape[0]
            query = self.pooling_query.expand(batch_size, 1, -1)  # (batch_size, 1, hidden_dim)
            
            # 计算注意力权重
            pooled, attention_weights = self.attention_pooling(
                query=query,                    # (batch_size, 1, hidden_dim)
                key=encoded,                   # (batch_size, seq_len, hidden_dim)
                value=encoded,                 # (batch_size, seq_len, hidden_dim)
                key_padding_mask=~mask if mask is not None else None
            )
            pooled = pooled.squeeze(1)  # (batch_size, hidden_dim)
        
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        return pooled, attention_weights

# ==================== 训练函数适配 ====================
def train_model(train_loader, val_loader, model_params, training_params):
    """训练ESMC多头注意力模型"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 获取输入维度 - 修复解包问题
    for batch_data in train_loader:
        if len(batch_data) == 5:  # 包含diff_counts
            x_batch, mask_batch, y_batch, len_batch, diff_batch = batch_data
        else:  # 向后兼容
            x_batch, mask_batch, y_batch, len_batch = batch_data
        
        input_dim = x_batch.shape[2]  # 应该是1152
        break
    
    # 初始化模型
    model = ESMCMultiHeadAttentionNetwork(
        input_dim=input_dim,
        hidden_dim=model_params['hidden_dim'],
        num_layers=model_params['num_layers'],
        num_heads=model_params['num_heads'],
        dropout=model_params['dropout'],
        pooling=model_params['pooling'],
        max_seq_len=model_params.get('max_seq_len', 201)
    ).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: 总计={total_params:,}, 可训练={trainable_params:,}")
    
    optimizer = Adam(model.parameters(), lr=training_params['lr'], weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # 学习率调度
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            min_lr_factor = 1e-2
            total_epochs = training_params['epochs']
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_factor + (1 - min_lr_factor) * cosine_decay
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # 训练记录
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = training_params['patience']
    
    history = {
        'train_loss': [], 'val_loss': [], 'val_mse': [],
        'val_mae': [], 'val_r2': [], 'val_pearson': [],
        'learning_rates': []
    }
    
    for epoch in range(training_params['epochs']):
        # 训练阶段
        model.train()
        train_losses = []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_params['epochs']} [Train]")
        for batch_data in train_pbar:
            # 修复：统一处理数据解包
            if len(batch_data) == 5:  # 包含diff_counts
                x_batch, mask_batch, y_batch, len_batch, diff_batch = batch_data
            else:  # 向后兼容
                x_batch, mask_batch, y_batch, len_batch = batch_data
            
            x_batch = x_batch.to(device)
            mask_batch = mask_batch.to(device)
            y_batch = y_batch.to(device)
            len_batch = len_batch.to(device)
            
            optimizer.zero_grad()
            
            y_pred, _, _ = model(x_batch, mask_batch, len_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_train_loss = sum(train_losses) / len(train_losses)

        # 验证阶段
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc="Validation"):
                # 修复：统一处理数据解包
                if len(batch_data) == 5:  # 包含diff_counts
                    x_batch, mask_batch, y_batch, len_batch, diff_batch = batch_data
                else:  # 向后兼容
                    x_batch, mask_batch, y_batch, len_batch = batch_data
                
                x_batch = x_batch.to(device)
                mask_batch = mask_batch.to(device)
                y_batch = y_batch.to(device)
                len_batch = len_batch.to(device)
                
                y_pred, _, _ = model(x_batch, mask_batch, len_batch)
                loss = criterion(y_pred.squeeze(), y_batch)
                
                val_losses.append(loss.item())
                val_preds.extend(y_pred.squeeze().cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())

        # 计算平均验证损失        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        # 计算验证指标
        val_mse = mean_squared_error(val_targets, val_preds)
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_r2 = r2_score(val_targets, val_preds)
        val_pearson, _ = pearsonr(val_targets, val_preds)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mse'].append(val_mse)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)
        history['val_pearson'].append(val_pearson)
        history['learning_rates'].append(current_lr)
        
        logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, "
                   f"Val Loss={avg_val_loss:.6f}, MSE={val_mse:.6f}, "
                   f"MAE={val_mae:.6f}, R²={val_r2:.6f}, Pearson={val_pearson:.6f}, "
                   f"LR={current_lr:.6f}")
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            logger.info("✓ 新的最佳模型已保存!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"早停触发！连续{patience}个epoch验证损失没有改善")
                break
    
    # 恢复最佳模型
    model.load_state_dict(best_model_state)
    
    final_metrics = {
        'loss': best_val_loss,
        'mse': val_mse, 'mae': val_mae,
        'r2': val_r2, 'pearson': val_pearson
    }
    
    return model, final_metrics, history

# ==================== 注意力可视化 ====================
def visualize_attention_weights(model, data_loader, n_samples=3):
    """可视化注意力权重"""
    model.eval()
    
    samples_data = []
    count = 0
    
    for batch_data in data_loader:  # 修改这里
        if count >= n_samples:
            break
        
        # 统一处理数据解包
        if len(batch_data) == 5:  # 包含diff_counts
            x_batch, mask_batch, y_batch, len_batch, diff_batch = batch_data
        else:  # 向后兼容
            x_batch, mask_batch, y_batch, len_batch = batch_data
            
        x_batch = x_batch.to(device)
        mask_batch = mask_batch.to(device)
        len_batch = len_batch.to(device)
        
        with torch.no_grad():
            y_pred, attention_weights, encoded = model(x_batch, mask_batch, len_batch)
        
        # 处理批次中的样本
        batch_size = min(len(y_batch), n_samples - count)
        for i in range(batch_size):
            seq_len = len_batch[i].item()
            
            sample_info = {
                'seq_length': seq_len,
                'true_value': y_batch[i].item(),
                'predicted_value': y_pred[i].item(),
                'attention_weights': None,
                'sequence_repr': encoded[i, :seq_len].cpu().numpy()  # (seq_len, hidden_dim)
            }
            
            # 处理注意力权重
            if attention_weights is not None and model.pooling == 'attention':
                # attention_weights shape: (batch_size, 1, seq_len)
                attn = attention_weights[i, 0, :seq_len].cpu().numpy()  # (seq_len,)
                sample_info['attention_weights'] = attn
            
            samples_data.append(sample_info)
            count += 1
            
            if count >= n_samples:
                break
    
    # 创建可视化
    fig, axes = plt.subplots(n_samples, 2, figsize=(15, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples_data):
        # 左图：注意力权重分布
        ax1 = axes[i, 0]
        if sample['attention_weights'] is not None:
            positions = np.arange(sample['seq_length'])
            bars = ax1.bar(positions, sample['attention_weights'], alpha=0.7, color='skyblue')
            ax1.set_xlabel('序列位置')
            ax1.set_ylabel('注意力权重')
            ax1.set_title(f'样本 {i+1}: 注意力权重分布\n'
                         f'真实值: {sample["true_value"]:.4f}, '
                         f'预测值: {sample["predicted_value"]:.4f}')
            ax1.grid(True, alpha=0.3)
            
            # 高亮最重要的位置
            max_idx = np.argmax(sample['attention_weights'])
            bars[max_idx].set_color('red')
            ax1.annotate(f'Max: {sample["attention_weights"][max_idx]:.4f}',
                        xy=(max_idx, sample['attention_weights'][max_idx]),
                        xytext=(max_idx, sample['attention_weights'][max_idx] + 0.01),
                        arrowprops=dict(arrowstyle='->', color='red'))
        else:
            ax1.text(0.5, 0.5, '注意力权重不可用\n(非attention pooling)', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title(f'样本 {i+1}: {sample["true_value"]:.4f} -> {sample["predicted_value"]:.4f}')
        
        # 右图：序列表示的主成分分析可视化
        ax2 = axes[i, 1]
        seq_repr = sample['sequence_repr']  # (seq_len, hidden_dim)
        
        # 计算每个位置表示向量的L2范数
        norms = np.linalg.norm(seq_repr, axis=1)  # (seq_len,)
        positions = np.arange(sample['seq_length'])
        
        ax2.plot(positions, norms, 'o-', alpha=0.7, color='green')
        ax2.set_xlabel('序列位置')
        ax2.set_ylabel('表示向量L2范数')
        ax2.set_title(f'序列表示强度分布 (长度: {sample["seq_length"]})')
        ax2.grid(True, alpha=0.3)
        
        # 标注最强的表示位置
        max_norm_idx = np.argmax(norms)
        ax2.annotate(f'Max: {norms[max_norm_idx]:.2f}',
                    xy=(max_norm_idx, norms[max_norm_idx]),
                    xytext=(max_norm_idx, norms[max_norm_idx] + 0.5),
                    arrowprops=dict(arrowstyle='->', color='green'))
    
    plt.tight_layout()
    return fig

# ==================== 评估函数 ====================
def model_evaluation(model, test_loader, output_dir=None):
    """详细评估模型性能，支持根据diff_counts着色"""
    model.eval()
    
    test_preds = []
    test_targets = []
    test_diff_counts = []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Testing"):
            if len(batch_data) == 5:  # 包含diff_counts
                x_batch, mask_batch, y_batch, len_batch, diff_batch = batch_data
            else:  # 向后兼容
                x_batch, mask_batch, y_batch, len_batch = batch_data
                diff_batch = torch.zeros_like(y_batch)
            
            x_batch = x_batch.to(device)
            mask_batch = mask_batch.to(device)
            len_batch = len_batch.to(device)
            
            y_pred, attention_weights, _ = model(x_batch, mask_batch, len_batch)
            
            test_preds.extend(y_pred.squeeze().cpu().numpy())
            test_targets.extend(y_batch.numpy())
            test_diff_counts.extend(diff_batch.numpy())
            
            # 收集注意力权重用于分析
            if attention_weights is not None:
                all_attention_weights.append(attention_weights.cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_diff_counts = np.array(test_diff_counts)
    
    # 计算基本指标（保持不变）
    test_mse = mean_squared_error(test_targets, test_preds)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_r2 = r2_score(test_targets, test_preds)
    test_mape = np.mean(np.abs((test_targets - test_preds) / test_targets)) * 100
    
    # 相关性分析
    test_pearson, pearson_pvalue = pearsonr(test_targets, test_preds)
    test_spearman, spearman_pvalue = spearmanr(test_targets, test_preds)
    
    # 残差分析
    residuals = test_targets - test_preds
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # 误差分位数
    abs_errors = np.abs(residuals)
    error_percentiles = {
        '25th': np.percentile(abs_errors, 25),
        '50th': np.percentile(abs_errors, 50),
        '75th': np.percentile(abs_errors, 75),
        '95th': np.percentile(abs_errors, 95)
    }
    
    # ========= 修复：按diff_counts分组分析性能 =========
    unique_diff_counts = np.unique(test_diff_counts)
    diff_count_stats = {}
    for diff_count in unique_diff_counts:
        mask = test_diff_counts == diff_count
        if np.sum(mask) > 0:
            group_targets = test_targets[mask]
            group_preds = test_preds[mask]
            
            # 基本统计（适用于任意样本数）
            stats = {
                'count': int(np.sum(mask)),
                'mse': float(mean_squared_error(group_targets, group_preds)),
                'mae': float(mean_absolute_error(group_targets, group_preds)),
            }
            
            # 需要至少2个样本的统计指标
            if len(group_targets) >= 2:
                try:
                    stats['r2'] = float(r2_score(group_targets, group_preds))
                    stats['pearson'] = float(pearsonr(group_targets, group_preds)[0])
                except Exception as e:
                    logger.warning(f"计算diff_count={diff_count}的R²/Pearson时出错: {e}")
                    stats['r2'] = float('nan')
                    stats['pearson'] = float('nan')
            else:
                # 样本数不足，设为NaN
                stats['r2'] = float('nan')
                stats['pearson'] = float('nan')
                logger.info(f"diff_count={diff_count}只有{len(group_targets)}个样本, R²和Pearson设为NaN")
            
            diff_count_stats[int(diff_count)] = stats
    
    # 日志输出
    logger.info("="*60)
    logger.info("测试集详细性能报告:")
    logger.info(f"样本数量: {len(test_targets)}")
    logger.info(f"MSE: {test_mse:.6f}")
    logger.info(f"RMSE: {test_rmse:.6f}")
    logger.info(f"MAE: {test_mae:.6f}")
    logger.info(f"R²: {test_r2:.6f}")
    logger.info(f"MAPE: {test_mape:.2f}%")
    logger.info(f"Pearson相关: {test_pearson:.6f} (p={pearson_pvalue:.6f})")
    logger.info(f"Spearman相关: {test_spearman:.6f} (p={spearman_pvalue:.6f})")
    
    # 输出按diff_counts分组的性能
    logger.info("\n按diff_counts分组的性能:")
    for diff_count, stats in diff_count_stats.items():
        r2_str = f"{stats['r2']:.6f}" if not np.isnan(stats['r2']) else "N/A"
        pearson_str = f"{stats['pearson']:.6f}" if not np.isnan(stats['pearson']) else "N/A"
        logger.info(f"  diff_count={diff_count}: 样本数={stats['count']}, "
                   f"MSE={stats['mse']:.6f}, MAE={stats['mae']:.6f}, "
                   f"R²={r2_str}, Pearson={pearson_str}")
    logger.info("="*60)
    
    # ========= 可视化结果 =========
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 设置颜色映射
    from matplotlib import cm
    import matplotlib.colors as mcolors
    
    n_diff_counts = len(unique_diff_counts)
    if n_diff_counts <= 10:
        # 使用离散颜色
        colors = plt.cm.tab10(np.linspace(0, 1, n_diff_counts))
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(boundaries=np.arange(n_diff_counts+1)-0.5, ncolors=n_diff_counts)
        color_values = np.searchsorted(unique_diff_counts, test_diff_counts)
    else:
        # 使用连续颜色映射
        cmap = plt.cm.viridis
        norm = mcolors.Normalize(vmin=test_diff_counts.min(), vmax=test_diff_counts.max())
        color_values = test_diff_counts
    
    # 1. 预测vs真实值散点图（按diff_counts着色）
    ax1 = axes[0, 0]
    scatter = ax1.scatter(test_targets, test_preds, c=color_values, cmap=cmap, norm=norm, 
                         alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    # 添加完美预测线
    min_val, max_val = min(min(test_targets), min(test_preds)), max(max(test_targets), max(test_preds))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
    
    ax1.set_xlabel('真实值', fontsize=12)
    ax1.set_ylabel('预测值', fontsize=12)
    ax1.set_title('预测值 vs 真实值 (按diff_counts着色)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('diff_counts', rotation=270, labelpad=15)
    
    # 添加性能指标文本
    text = f"R²: {test_r2:.4f}\nPearson: {test_pearson:.4f}\nRMSE: {test_rmse:.4f}\nMAE: {test_mae:.4f}"
    ax1.annotate(text, xy=(0.05, 0.75), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.8))
    
    # 2. 残差分布图
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(mean_residual, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_residual:.4f}')
    ax2.axvline(0, color='green', linestyle='-', linewidth=2, label='理想值: 0')
    ax2.set_xlabel('残差 (真实值 - 预测值)', fontsize=12)
    ax2.set_ylabel('频次', fontsize=12)
    ax2.set_title('残差分布', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 残差vs预测值（按diff_counts着色）
    ax3 = axes[0, 2]
    scatter2 = ax3.scatter(test_preds, residuals, c=color_values, cmap=cmap, norm=norm,
                          alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    ax3.axhline(0, color='red', linestyle='--', linewidth=2)
    ax3.axhline(mean_residual, color='green', linestyle='--', linewidth=2, label=f'残差均值: {mean_residual:.4f}')
    ax3.set_xlabel('预测值', fontsize=12)
    ax3.set_ylabel('残差', fontsize=12)
    ax3.set_title('残差 vs 预测值 (按diff_counts着色)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. diff_counts分布
    ax4 = axes[1, 0]
    diff_counts_unique, diff_counts_counts = np.unique(test_diff_counts, return_counts=True)
    bars = ax4.bar(diff_counts_unique, diff_counts_counts, alpha=0.7)
    ax4.set_xlabel('diff_counts', fontsize=12)
    ax4.set_ylabel('样本数量', fontsize=12)
    ax4.set_title('diff_counts分布', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bar, count in zip(bars, diff_counts_counts):
        height = bar.get_height()
        ax4.annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # 5. 按diff_counts的性能对比（修复版本）
    ax5 = axes[1, 1]
    
    # 只显示样本数>=2的diff_counts
    valid_diff_counts = []
    valid_r2_values = []
    valid_mse_values = []
    
    for dc in sorted(diff_count_stats.keys()):
        stats = diff_count_stats[dc]
        if stats['count'] >= 2 and not np.isnan(stats['r2']):
            valid_diff_counts.append(dc)
            valid_r2_values.append(stats['r2'])
            valid_mse_values.append(stats['mse'])
    
    if valid_diff_counts:  # 只有当有有效数据时才绘制
        ax5_twin = ax5.twinx()
        
        line1 = ax5.plot(valid_diff_counts, valid_r2_values, 'o-', color='blue', label='R²', linewidth=2, markersize=8)
        line2 = ax5_twin.plot(valid_diff_counts, valid_mse_values, 's-', color='red', label='MSE', linewidth=2, markersize=8)
        
        ax5.set_xlabel('diff_counts', fontsize=12)
        ax5.set_ylabel('R²', color='blue', fontsize=12)
        ax5_twin.set_ylabel('MSE', color='red', fontsize=12)
        ax5.set_title('不同diff_counts的性能对比\n(仅显示样本数≥2的组)', fontsize=14)
        ax5.grid(True, alpha=0.3)
        
        # 合并图例
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    else:
        ax5.text(0.5, 0.5, '没有足够样本\n计算性能对比', ha='center', va='center', 
                transform=ax5.transAxes, fontsize=14)
        ax5.set_title('性能对比 (数据不足)', fontsize=14)
    
    # 6. 误差统计箱线图（按diff_counts分组）
    ax6 = axes[1, 2]
    error_groups = []
    error_labels = []
    
    for dc in sorted(diff_count_stats.keys()):
        stats = diff_count_stats[dc]
        if stats['count'] >= 3:  # 至少3个样本才绘制箱线图
            mask = test_diff_counts == dc
            group_errors = abs_errors[mask]
            error_groups.append(group_errors)
            error_labels.append(f'diff={dc}\n(n={stats["count"]})')
    
    if error_groups:
        bp = ax6.boxplot(error_groups, labels=error_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
    else:
        ax6.text(0.5, 0.5, '样本数不足\n无法绘制箱线图', ha='center', va='center', 
                transform=ax6.transAxes, fontsize=14)
    
    ax6.set_xlabel('diff_counts组', fontsize=12)
    ax6.set_ylabel('绝对误差', fontsize=12)
    ax6.set_title('不同diff_counts组的误差分布\n(仅显示样本数≥3的组)', fontsize=14)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    if output_dir:
        figtime = time.strftime('%Y%m%d_%H%M%S')
        fig_path = os.path.join(output_dir, f'esmc_detailed_evaluation_with_diffcounts_{figtime}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"详细评估图（含diff_counts着色）已保存至: {fig_path}")
    plt.close()
    
    # 保存按diff_counts分组的详细统计
    if output_dir:
        stats_path = os.path.join(output_dir, f'diff_counts_stats_{time.strftime("%Y%m%d_%H%M%S")}.csv')
        stats_df = pd.DataFrame(diff_count_stats).T
        stats_df.index.name = 'diff_counts'
        stats_df.to_csv(stats_path)
        logger.info(f"diff_counts统计已保存至: {stats_path}")
    
    # 返回评估结果（包含diff_counts信息）
    return {
        'sample_count': len(test_targets),
        'mse': test_mse, 'rmse': test_rmse, 'mae': test_mae, 'r2': test_r2, 'mape': test_mape,
        'pearson_r': test_pearson, 'pearson_pvalue': pearson_pvalue,
        'spearman_r': test_spearman, 'spearman_pvalue': spearman_pvalue,
        'mean_residual': mean_residual, 'std_residual': std_residual,
        'within_1_std_percent': np.sum(abs_errors <= std_residual) / len(abs_errors) * 100,
        'within_2_std_percent': np.sum(abs_errors <= 2 * std_residual) / len(abs_errors) * 100,
        'error_percentiles': error_percentiles,
        'max_absolute_error': np.max(abs_errors), 'min_absolute_error': np.min(abs_errors),
        'predictions': test_preds, 'targets': test_targets, 'residuals': residuals,
        'diff_counts': test_diff_counts,
        'diff_count_stats': diff_count_stats
    }

# ==================== 绘制训练历史 ====================
def plot_training_history(histories, best_fold=None):
    """绘制训练历史"""
    plt.figure(figsize=(15, 10))
    
    # 训练和验证损失
    plt.subplot(2, 3, 1)
    for i, hist in enumerate(histories):
        style = '-' if (best_fold is not None and i == best_fold) else '--'
        plt.plot(hist['train_loss'], style, alpha=0.7, label=f'Fold {i+1} Train')
        plt.plot(hist['val_loss'], style, label=f'Fold {i+1} Val')
    plt.title('训练和验证损失')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 学习率变化
    plt.subplot(2, 3, 2)
    for i, hist in enumerate(histories):
        style = '-' if (best_fold is not None and i == best_fold) else '--'
        plt.plot(hist.get('learning_rates', []), style, label=f'Fold {i+1}')
    plt.title('学习率变化')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # MSE
    plt.subplot(2, 3, 3)
    for i, hist in enumerate(histories):
        style = '-' if (best_fold is not None and i == best_fold) else '--'
        plt.plot(hist['val_mse'], style, label=f'Fold {i+1}')
    plt.title('验证MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # MAE
    plt.subplot(2, 3, 4)
    for i, hist in enumerate(histories):
        style = '-' if (best_fold is not None and i == best_fold) else '--'
        plt.plot(hist['val_mae'], style, label=f'Fold {i+1}')
    plt.title('验证MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # R²
    plt.subplot(2, 3, 5)
    for i, hist in enumerate(histories):
        style = '-' if (best_fold is not None and i == best_fold) else '--'
        plt.plot(hist['val_r2'], style, label=f'Fold {i+1}')
    plt.title('验证R²')
    plt.xlabel('Epochs')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Pearson相关系数
    plt.subplot(2, 3, 6)
    for i, hist in enumerate(histories):
        style = '-' if (best_fold is not None and i == best_fold) else '--'
        plt.plot(hist['val_pearson'], style, label=f'Fold {i+1}')
    plt.title('验证Pearson相关系数')
    plt.xlabel('Epochs')
    plt.ylabel('Pearson')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt.gcf()

# ==================== 主函数 ====================
def main():
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    logger.info(f"设置随机种子: {args.seed}")
    
    # 创建输出目录
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"结果将保存至: {output_dir}")
    
    # 加载ESMC数据集
    dataset = H5ESMCDataset(
        args.data_path,
        max_seq_len=args.max_seq_len,
        normalize=True,
        debug=args.debug,
        debug_sample=100 if args.debug else None
    )
    
    logger.info(f"数据集加载完成: {len(dataset)}个样本")
    
    # 数据集划分
    data_num = len(dataset)
    perm = np.random.permutation(data_num)
    n_test = int(0.15 * data_num)
    test_idx = perm[:n_test]
    train_val_idx = perm[n_test:]
    
    test_set = Subset(dataset, test_idx)
    train_val_set = Subset(dataset, train_val_idx)
    logger.info(f"数据划分: 训练+验证={len(train_val_set)}, 测试={len(test_set)}")
    
    # 模型和训练参数
    model_params = {
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'dropout': args.dropout,
        'pooling': args.pooling,
        'max_seq_len': args.max_seq_len
    }
    
    training_params = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience
    }
    
    # K折交叉验证
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    best_model_states = []
    fold_results = []
    all_histories = []
    best_val_mse = float('inf')
    best_fold_idx = None
    
    for fold, (train_subidx, val_subidx) in enumerate(kf.split(train_val_idx), 1):
        logger.info(f"\n{'='*20} 第 {fold}/{args.k_folds} 折 {'='*20}")
        
        # 创建数据加载器
        train_loader = DataLoader(
            Subset(train_val_set, train_subidx),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            Subset(train_val_set, val_subidx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        
        # 训练模型
        model, val_metrics, history = train_model(
            train_loader, val_loader,
            model_params, training_params
        )
        
        fold_results.append(val_metrics)
        all_histories.append(history)
        best_model_states.append(model.state_dict())
        
        logger.info(f"第 {fold} 折 验证结果: MSE={val_metrics['mse']:.6f}")
        
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            best_fold_idx = fold - 1
            logger.info(f"✓ 更新最佳模型为第 {fold} 折")
        
        # 内存清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 加载最佳模型
    logger.info(f"\n加载最佳模型 (第 {best_fold_idx + 1} 折)")
    
    # 重新创建模型并加载权重
    best_model = ESMCMultiHeadAttentionNetwork(
        input_dim=1152,  # ESMC固定维度
        **model_params
    ).to(device)
    best_model.load_state_dict(best_model_states[best_fold_idx])
    
    # 保存训练历史
    history_fig = plot_training_history(all_histories, best_fold=best_fold_idx)
    history_path = os.path.join(output_dir, f'training_history_{timestamp}.png')
    history_fig.savefig(history_path, dpi=300, bbox_inches='tight')
    plt.close(history_fig)
    logger.info(f"训练历史已保存: {history_path}")
    
    # 测试集评估
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_results = model_evaluation(best_model, test_loader, output_dir)
    
    # 注意力可视化
    att_fig = visualize_attention_weights(best_model, test_loader, n_samples=5)
    att_path = os.path.join(output_dir, f'attention_visualization_{timestamp}.png')
    att_fig.savefig(att_path, dpi=300, bbox_inches='tight')
    plt.close(att_fig)
    logger.info(f"注意力可视化已保存: {att_path}")
    
    # 保存模型
    model_path = os.path.join(output_dir, 'best_esmc_model.pt')
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'model_params': model_params,
        'test_results': test_results,
        'fold_results': fold_results
    }, model_path)
    logger.info(f"模型已保存: {model_path}")
    
    # 保存详细结果摘要
    summary_path = os.path.join(output_dir, f'results_summary_{timestamp}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" "*25 + "ESMC多头注意力神经网络 训练结果摘要\n")
        f.write("="*80 + "\n\n")
        
        # 输入参数
        f.write("【输入参数】\n")
        f.write("-"*40 + "\n")
        for arg, value in vars(args).items():
            f.write(f"{arg:20s}: {value}\n")
        f.write(f"模型参数量: {sum(p.numel() for p in best_model.parameters()):,}\n")
        f.write("\n")
        
        # 交叉验证结果汇总
        f.write("【交叉验证结果汇总】\n")
        f.write("-"*40 + "\n")
        f.write(f"最佳验证折: {best_fold_idx + 1}/{args.k_folds}\n")
        f.write(f"最佳验证MSE: {best_val_mse:.6f}\n\n")
        
        val_mses = [r['mse'] for r in fold_results]
        val_r2s = [r['r2'] for r in fold_results]
        val_pearsons = [r['pearson'] for r in fold_results]
        
        f.write(f"验证MSE: {np.mean(val_mses):.6f} ± {np.std(val_mses):.6f}\n")
        f.write(f"验证R²: {np.mean(val_r2s):.6f} ± {np.std(val_r2s):.6f}\n")
        f.write(f"验证Pearson: {np.mean(val_pearsons):.6f} ± {np.std(val_pearsons):.6f}\n\n")
        
        # 测试集详细结果
        f.write("【测试集最终性能】\n")
        f.write("-"*40 + "\n")
        f.write(f"测试样本数: {test_results['sample_count']}\n")
        f.write(f"MSE: {test_results['mse']:.6f}\n")
        f.write(f"RMSE: {test_results['rmse']:.6f}\n")
        f.write(f"MAE: {test_results['mae']:.6f}\n")
        f.write(f"R²: {test_results['r2']:.6f}\n")
        f.write(f"MAPE: {test_results['mape']:.2f}%\n")
        f.write(f"Pearson相关: {test_results['pearson_r']:.6f} (p={test_results['pearson_pvalue']:.6f})\n")
        f.write(f"Spearman相关: {test_results['spearman_r']:.6f} (p={test_results['spearman_pvalue']:.6f})\n\n")
        
        # 误差分析
        f.write("【误差分析】\n")
        f.write("-"*40 + "\n")
        f.write(f"残差均值: {test_results['mean_residual']:.6f}\n")
        f.write(f"残差标准差: {test_results['std_residual']:.6f}\n")
        f.write(f"1σ内准确率: {test_results['within_1_std_percent']:.2f}%\n")
        f.write(f"2σ内准确率: {test_results['within_2_std_percent']:.2f}%\n\n")
        
        f.write("误差分位数:\n")
        for k, v in test_results['error_percentiles'].items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write(f"  最大误差: {test_results['max_absolute_error']:.6f}\n")
        f.write(f"  最小误差: {test_results['min_absolute_error']:.6f}\n\n")
        
        # 各折详细结果
        f.write("【各折详细验证结果】\n")
        f.write("-"*60 + "\n")
        f.write(f"{'折':<3} {'损失':<10} {'MSE':<10} {'MAE':<10} {'R²':<10} {'Pearson':<10}\n")
        f.write("-"*60 + "\n")
        for i, r in enumerate(fold_results):
            f.write(f"{i+1:<3} {r['loss']:<10.6f} {r['mse']:<10.6f} {r['mae']:<10.6f} "
                   f"{r['r2']:<10.6f} {r['pearson']:<10.6f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")
    
    logger.info(f"详细摘要已保存: {summary_path}")
    
    # 保存预测结果CSV
    results_df = pd.DataFrame({
        'true_values': test_results['targets'],
        'predictions': test_results['predictions'],
        'residuals': test_results['residuals'],
        'absolute_errors': np.abs(test_results['residuals'])
    })
    csv_path = os.path.join(output_dir, f'test_predictions_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)
    logger.info(f"预测结果CSV已保存: {csv_path}")
    
    logger.info(f"\n🎉 训练完成！所有结果已保存至: {output_dir}")

if __name__ == "__main__":
    main()

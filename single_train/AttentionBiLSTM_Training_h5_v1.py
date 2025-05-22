# -*- coding: utf-8 -*-
"""
基于 HDF5 格式蛋白质嵌入数据的 BiLSTM+Attention 训练脚本（内存优化版）
适用于大型数据集（如50G HDF5文件）
"""
import os
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import gc
import h5py
from torch.cuda.amp import autocast, GradScaler

# 命令行参数配置
def parse_args():
    parser = argparse.ArgumentParser(description='HDF5 嵌入数据 BiLSTM+Attention 训练脚本 (内存优化版)')
    parser.add_argument('--data_path', type=str, default=r"preprocess\ESM\output\sampled_output_esm_embeddings.h5", help='输入 HDF5 文件路径 (.h5)')
    parser.add_argument('--max_seq_len', type=int, default=None, help='最大序列长度，默认使用数据集中的最大长度')
    parser.add_argument('--hidden_dim', type=int, default=128, help='LSTM 隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM 层数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout 比例')
    parser.add_argument('--lr', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--batch_size', type=int, default=128, help='训练批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='最大训练轮数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--k_folds', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output_dir', type=str, default='ModelResults/BiLSTMResults_h5', help='结果输出目录')
    parser.add_argument('--test_size', type=float, default=0.15, help='测试集比例')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--cache_size', type=int, default=1000, help='嵌入缓存大小')
    return parser.parse_args()

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger('bilstm_attention_h5_opt')

# 随机种子
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

# 嵌入缓存类 - 避免重复加载相同数据
class EmbeddingCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get(self, key):
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if len(self.cache) >= self.max_size and key not in self.cache:
            # 移除最少访问的项
            least_key = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[least_key]
            del self.access_count[least_key]
        
        self.cache[key] = value
        self.access_count[key] = 1
    
    def clear(self):
        self.cache.clear()
        self.access_count.clear()

# 全局嵌入缓存
embedding_cache = EmbeddingCache()

# 自定义HDF5数据集
class H5ProteinDataset:
    """
    从 HDF5 文件加载蛋白质嵌入和标签，假设文件结构为：

    /
    ├── embeddings        # Group: 所有 emb_0, emb_1, …  
    │   ├── emb_0        # [L₀, D]
    │   ├── emb_1        # [L₁, D]
    │   └── …
    │
    ├── indices     [N]           # 全局索引
    ├── sequences   [N] ⟨vlen str⟩  # 原始蛋白质序列
    ├── diff_counts [N]           # diff_count 标签
    ├── mean_log10Ka [N]          # 亲和力标签
    ├── seq_lengths [N]           # 序列长度
    └── attrs…                    # total_samples, embedding_dim, created_time
    """
    def __init__(self, data_path, indices=None, max_seq_len=None, shuffle=True, seed=42):
        self.data_path   = data_path
        self.max_seq_len = max_seq_len
        self.shuffle     = shuffle
        self.seed        = seed

        # 先打开一次，读取所有 metadata
        with h5py.File(data_path, 'r') as f:
            self.all_indices    = f['indices'][:]                          # [N]
            self.sequences      = f['sequences'][:].astype(str)            # [N]
            self.diff_counts    = f['diff_counts'][:]                      # [N]
            self.targets        = f['mean_log10Ka'][:].astype(np.float32)  # [N]
            self.seq_lengths    = f['seq_lengths'][:]                      # [N]
            self.embedding_dim  = f.attrs['embedding_dim']                 # D

        # 决定使用哪些样本
        N = len(self.all_indices)
        if indices is None:
            self.sample_idx = np.arange(N)
        else:
            self.sample_idx = np.array(indices, dtype=int)

        # 对应到 HDF5 中的全局编号
        self.global_idx = self.all_indices[self.sample_idx]

    def __len__(self):
        return len(self.sample_idx)

    def get_embedding(self, idx):
        """
        返回 [L_i, D] 的 numpy 数组
        """
        gid = int(self.global_idx[idx])
        key = f'emb_{gid}'
        with h5py.File(self.data_path, 'r') as f:
            emb = f['embeddings'][key][:]
        return emb.astype(np.float32)

    def create_dataloader(self, batch_size, num_workers=0):
        return H5DataLoader(self, batch_size, self.max_seq_len,
                            shuffle=self.shuffle, seed=self.seed)

class H5DataLoader:
    """
    可迭代加载器，将可变长度嵌入 pad 到同一长度，
    输出 (features, masks, targets, diff_counts, sequences)。
    """
    def __init__(self, dataset, batch_size, max_seq_len, shuffle=True, seed=42):
        self.ds          = dataset
        self.bs          = batch_size
        self.max_seq_len = max_seq_len
        self.shuffle     = shuffle
        self.seed        = seed

        self.indices = np.arange(len(dataset))
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(self.indices)

        self.steps = (len(self.indices) + batch_size - 1) // batch_size

    def __len__(self):
        return self.steps

    def __iter__(self):
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(self.indices)

        for i in range(self.steps):
            batch_ids = self.indices[i*self.bs : (i+1)*self.bs]
            # 加载 embeddings
            embs = [self.ds.get_embedding(j) for j in batch_ids]
            lengths = [e.shape[0] for e in embs]

            # 决定 pad 长度
            if self.max_seq_len:
                max_len = min(self.max_seq_len, max(lengths))
            else:
                max_len = max(lengths)

            B, D = len(embs), self.ds.embedding_dim
            feats = np.zeros((B, max_len, D), dtype=np.float32)
            masks = np.zeros((B, max_len),    dtype=bool)

            # 填充
            for bi, e in enumerate(embs):
                L = min(e.shape[0], max_len)
                feats[bi, :L] = e[:L]
                masks[bi, :L] = True

            targets     = self.ds.targets[batch_ids]
            diff_counts = self.ds.diff_counts[batch_ids]
            seqs        = self.ds.sequences[batch_ids]

            yield (torch.FloatTensor(feats),
                   torch.BoolTensor(masks),
                   torch.FloatTensor(targets),
                   diff_counts,
                   seqs)


# BiLSTM+Attention 模型定义
class AttentionBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers>1 else 0
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,1)
        )
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim,1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
        # x: [B, L, D]
        out, _ = self.lstm(x)  # [B, L, 2H]
        scores = self.attn(out)  # [B, L, 1]
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1)==0, float('-inf'))
        weights = torch.softmax(scores, dim=1)  # [B, L, 1]
        context = torch.sum(weights * out, dim=1)  # [B, 2H]
        h = self.relu(self.fc1(context))
        h = self.dropout(h)
        return self.fc2(h), weights


# 优化后的训练函数
def train_model(train_loader, val_loader, model_params, train_params, device):
    """使用数据加载器训练模型，支持混合精度和梯度累积"""
    # 清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 创建模型
    model = AttentionBiLSTM(
        input_dim=model_params['input_dim'],
        hidden_dim=model_params['hidden_dim'],
        num_layers=model_params['num_layers'],
        dropout=model_params['dropout']
    ).to(device)
    
    # 优化器和学习率调度器
    optimizer = Adam(model.parameters(), lr=train_params['lr'])
    
    # 学习率预热和指数衰减
    def lr_lambda(epoch):
        if epoch < 5:
            return (epoch + 1) / 5  # 预热阶段
        return 0.95 ** (epoch - 5)  # 指数衰减
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    criterion = nn.MSELoss()
    
    # 混合精度训练
    scaler = GradScaler() if train_params['use_amp'] else None
    
    # 训练状态跟踪
    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse': [],
        'val_mae': [],
        'val_r2': [],
        'val_pearson': []
    }
    
    # 主训练循环
    for epoch in range(train_params['epochs']):
        # 训练阶段
        model.train()
        train_losses = []
        optimizer.zero_grad()  # 开始前清零梯度
        
        # 梯度累积相关变量
        accum_steps = train_params['grad_accum_steps']
        batch_count = 0
        
        for batch_idx, (Xb, Mb, yb, _, _) in enumerate(train_loader):
            Xb, Mb, yb = Xb.to(device), Mb.to(device), yb.to(device)
            
            # 使用混合精度训练
            if train_params['use_amp']:
                with autocast():
                    pred, _ = model(Xb, Mb)
                    loss = criterion(pred.squeeze(), yb) / accum_steps
                
                # 缩放损失并反向传播
                scaler.scale(loss).backward()
                train_losses.append(loss.item() * accum_steps)
                
                # 梯度累积
                batch_count += 1
                if batch_count % accum_steps == 0 or batch_idx == len(train_loader) - 1:
                    # 梯度裁剪和优化器步进
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # 标准训练流程
                pred, _ = model(Xb, Mb)
                loss = criterion(pred.squeeze(), yb) / accum_steps
                loss.backward()
                train_losses.append(loss.item() * accum_steps)
                
                # 梯度累积
                batch_count += 1
                if batch_count % accum_steps == 0 or batch_idx == len(train_loader) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
        
        # 计算平均训练损失
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for Xb, Mb, yb, _, _ in val_loader:
                Xb, Mb, yb = Xb.to(device), Mb.to(device), yb.to(device)
                
                if train_params['use_amp']:
                    with autocast():
                        pred, _ = model(Xb, Mb)
                        loss = criterion(pred.squeeze(), yb)
                else:
                    pred, _ = model(Xb, Mb)
                    loss = criterion(pred.squeeze(), yb)
                
                val_losses.append(loss.item())
                all_preds.append(pred.squeeze().cpu().numpy())
                all_targets.append(yb.cpu().numpy())
        
        # 如果有验证数据
        if all_preds and all_targets:
            # 合并预测和目标
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            
            # 计算各种指标
            avg_val_loss = np.mean(val_losses)
            val_mse = mean_squared_error(all_targets, all_preds)
            val_mae = mean_absolute_error(all_targets, all_preds)
            val_r2 = r2_score(all_targets, all_preds)
            val_pearson, _ = pearsonr(all_targets, all_preds)
            
            # 更新历史记录
            history['val_loss'].append(avg_val_loss)
            history['val_mse'].append(val_mse)
            history['val_mae'].append(val_mae)
            history['val_r2'].append(val_r2)
            history['val_pearson'].append(val_pearson)
            
            # 学习率调度
            scheduler.step()
            
            # 打印训练状态
            logger.info(
                f"Epoch [{epoch+1}/{train_params['epochs']}] "
                f"Train Loss: {avg_train_loss:.4f} "
                f"Val Loss: {avg_val_loss:.4f} "
                f"MSE: {val_mse:.4f} "
                f"MAE: {val_mae:.4f} "
                f"R²: {val_r2:.4f} "
                f"Pearson: {val_pearson:.4f}"
            )
            
            # 早停检查
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_state = model.state_dict()
                patience_counter = 0
                logger.info(f"✓ 新的最佳模型 (Val Loss: {best_loss:.4f})")
            else:
                patience_counter += 1
                logger.info(f"✗ 模型未改进 ({patience_counter}/{train_params['patience']})")
                
                if patience_counter >= train_params['patience']:
                    logger.info(f"早停! 验证损失 {train_params['patience']} 轮未改进")
                    break
        else:
            logger.warning("验证集未产生预测，请检查数据加载器")
    
    # 加载最佳模型状态
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # 返回最终结果
    final_metrics = (
        history['val_mse'][-1],
        history['val_mae'][-1],
        history['val_r2'][-1],
        history['val_pearson'][-1]
    )
    
    return model, history, final_metrics


# 交叉验证训练
def train_with_kfold(dataset, model_params, train_params, device, k_folds):
    """执行K折交叉验证训练"""
    # 准备交叉验证
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=train_params['seed'])
    all_indices = np.arange(len(dataset))
    
    # 结果存储
    best_model = None
    best_mse = float('inf')
    best_fold = -1
    all_histories = []
    all_results = []
    
    # 遍历每一折
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_indices)):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"开始训练 Fold {fold+1}/{k_folds}")
        logger.info(f"{'=' * 50}")
        
        # 创建训练和验证数据集
        train_dataset = H5ProteinDataset(
            dataset.data_path,
            indices=train_idx,
            max_seq_len=dataset.max_seq_len,
            shuffle=True,
            seed=train_params['seed']
        )
        
        val_dataset = H5ProteinDataset(
            dataset.data_path,
            indices=val_idx,
            max_seq_len=dataset.max_seq_len,
            shuffle=False,
            seed=train_params['seed']
        )
        
        # 创建数据加载器
        train_loader = train_dataset.create_dataloader(
            batch_size=train_params['batch_size']
        )
        
        val_loader = val_dataset.create_dataloader(
            batch_size=train_params['batch_size']
        )
        
        # 训练模型
        fold_model, fold_history, (fold_mse, fold_mae, fold_r2, fold_pearson) = train_model(
            train_loader,
            val_loader,
            model_params,
            train_params,
            device
        )
        
        # 记录结果
        all_histories.append(fold_history)
        all_results.append({
            'fold': fold + 1,
            'mse': fold_mse,
            'mae': fold_mae,
            'r2': fold_r2,
            'pearson': fold_pearson
        })
        
        # 检查是否是最佳模型
        if fold_mse < best_mse:
            best_mse = fold_mse
            best_model = fold_model
            best_fold = fold
            logger.info(f"✓ 新的最佳折: Fold {fold+1} (MSE: {best_mse:.4f})")
        
        # 清理内存
        del fold_model, train_loader, val_loader, train_dataset, val_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info(f"\n最佳折: Fold {best_fold+1} with MSE={best_mse:.4f}")
    return best_model, all_histories, all_results, best_fold


# 训练历史绘图函数
def plot_training_history(histories, best_fold, out_path):
    """绘制训练历史曲线"""
    plt.figure(figsize=(15, 10))
    
    # 训练和验证损失
    plt.subplot(2, 2, 1)
    for i, h in enumerate(histories):
        style = '-' if i == best_fold else '--'
        alpha = 1.0 if i == best_fold else 0.5
        plt.plot(h['train_loss'], style, alpha=alpha, label=f'Fold {i+1} 训练')
        plt.plot(h['val_loss'], style, alpha=alpha, label=f'Fold {i+1} 验证')
    plt.title('损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.grid(True)
    plt.legend()
    
    # MSE
    plt.subplot(2, 2, 2)
    for i, h in enumerate(histories):
        style = '-' if i == best_fold else '--'
        alpha = 1.0 if i == best_fold else 0.5
        plt.plot(h['val_mse'], style, alpha=alpha, label=f'Fold {i+1}')
    plt.title('MSE')
    plt.xlabel('轮次')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.legend()
    
    # R²
    plt.subplot(2, 2, 3)
    for i, h in enumerate(histories):
        style = '-' if i == best_fold else '--'
        alpha = 1.0 if i == best_fold else 0.5
        plt.plot(h['val_r2'], style, alpha=alpha, label=f'Fold {i+1}')
    plt.title('R²')
    plt.xlabel('轮次')
    plt.ylabel('R²')
    plt.grid(True)
    plt.legend()
    
    # Pearson相关系数
    plt.subplot(2, 2, 4)
    for i, h in enumerate(histories):
        style = '-' if i == best_fold else '--'
        alpha = 1.0 if i == best_fold else 0.5
        plt.plot(h['val_pearson'], style, alpha=alpha, label=f'Fold {i+1}')
    plt.title('Pearson相关系数')
    plt.xlabel('轮次')
    plt.ylabel('Pearson r')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# 测试集评估与散点图
def evaluate_model(model, test_loader, device, out_path):
    """在测试集上评估模型并生成散点图"""
    model.eval()
    all_preds = []
    all_targets = []
    all_diffs = []
    
    with torch.no_grad():
        for Xb, Mb, yb, diff_counts, _ in test_loader:
            Xb, Mb = Xb.to(device), Mb.to(device)
            
            # 使用模型预测
            pred, _ = model(Xb, Mb)
            
            # 收集结果
            all_preds.append(pred.squeeze().cpu().numpy())
            all_targets.append(yb.numpy())
            all_diffs.append(diff_counts)
    
    # 合并结果
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_diffs = np.concatenate(all_diffs)
    
    # 计算指标
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    pearson, _ = pearsonr(all_targets, all_preds)
    
    logger.info(f"测试集评估结果:")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    logger.info(f"Pearson: {pearson:.4f}")
    
    # 绘制散点图
    plt.figure(figsize=(8, 8))
    
    # 根据diff_counts设置颜色
    sc = plt.scatter(all_targets, all_preds, c=all_diffs, cmap='viridis', alpha=0.7, s=30)
    
    # 添加对角线
    min_val = min(all_targets.min(), all_preds.min())
    max_val = max(all_targets.max(), all_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    # 添加标签和图例
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(f'预测 vs 真实 (MSE={mse:.4f}, R²={r2:.4f}, Pearson={pearson:.4f})')
    plt.colorbar(sc, label='diff_counts')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    
    return mse, mae, r2, pearson


# 注意力可视化
def visualize_attention(model, test_loader, device, out_path, n_samples=5):
    """可视化模型对测试样本的注意力权重"""
    model.eval()
    
    # 收集一些样本
    samples = []
    with torch.no_grad():
        for Xb, Mb, yb, diff_counts, seqs in test_loader:
            for i in range(min(n_samples, len(Xb))):
                # 获取单个样本
                x = Xb[i:i+1].to(device)
                m = Mb[i:i+1].to(device)
                y = yb[i].item()
                seq = seqs[i]
                
                # 运行模型获取预测和注意力权重
                pred, attn = model(x, m)
                pred_val = pred.item()
                
                # 获取序列长度
                seq_len = int(m.sum().item())
                
                # 提取注意力权重
                weights = attn[0, :seq_len, 0].cpu().numpy()
                
                samples.append((seq, y, pred_val, weights))
                
                if len(samples) >= n_samples:
                    break
            
            if len(samples) >= n_samples:
                break
    
    # 可视化注意力权重
    plt.figure(figsize=(15, 3*len(samples)))
    
    for i, (seq, true_val, pred_val, weights) in enumerate(samples):
        plt.subplot(len(samples), 1, i+1)
        
        # 绘制注意力条形图
        plt.bar(range(len(weights)), weights)
        
        # 如果序列不是很长，显示序列字符
        if len(seq) <= 100:
            plt.xticks(range(len(weights)), list(seq), rotation=45, fontsize=8)
        else:
            plt.xticks([])
        
        plt.title(f"样本 {i+1}: 真实值={true_val:.3f}, 预测值={pred_val:.3f}")
        plt.tight_layout()
    
    plt.savefig(out_path)
    plt.close()


# 主函数
def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 配置H5文件
    h5py.get_config().track_order = False  # 改善内存使用
    
    # 加载完整数据集
    full_dataset = H5ProteinDataset(
        data_path=args.data_path,
        max_seq_len=args.max_seq_len,
        shuffle=True,
        seed=args.seed
    )
    
    # 获取嵌入维度
    embedding_dim = full_dataset.embedding_dim
    logger.info(f"数据集: {len(full_dataset)} 样本, 嵌入维度: {embedding_dim}")
    
    # 划分训练和测试集
    indices = np.arange(len(full_dataset))
    np.random.shuffle(indices)
    test_size = int(len(indices) * args.test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    logger.info(f"训练集: {len(train_indices)} 样本")
    logger.info(f"测试集: {len(test_indices)} 样本")
    
    # 创建测试数据集
    test_dataset = H5ProteinDataset(
        data_path=args.data_path,
        indices=test_indices,
        max_seq_len=args.max_seq_len,
        shuffle=False
    )
    
    test_loader = test_dataset.create_dataloader(
        batch_size=args.batch_size
    )
    
    # 设置模型参数
    model_params = {
        'input_dim': embedding_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout
    }
    
    # 设置训练参数
    train_params = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience,
        'seed': args.seed,
        'use_amp': args.use_amp,
        'grad_accum_steps': args.grad_accum_steps
    }
    
    # 创建训练数据集
    train_dataset = H5ProteinDataset(
        data_path=args.data_path,
        indices=train_indices,
        max_seq_len=args.max_seq_len,
        shuffle=True,
        seed=args.seed
    )
    
    # 执行K折交叉验证
    best_model, histories, results, best_fold = train_with_kfold(
        train_dataset,
        model_params,
        train_params,
        device,
        args.k_folds
    )
    
    # 保存训练历史图
    history_path = os.path.join(output_dir, 'training_history.png')
    plot_training_history(histories, best_fold, history_path)
    
    # 保存最佳模型
    model_path = os.path.join(output_dir, 'best_model.pt')
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'model_params': model_params,
        'results': results,
        'best_fold': best_fold
    }, model_path)
    
    # 在测试集上评估模型
    logger.info("在测试集上评估模型...")
    test_scatter_path = os.path.join(output_dir, 'test_scatter.png')
    eval_metrics = evaluate_model(best_model, test_loader, device, test_scatter_path)
    
    # 可视化注意力权重
    attn_vis_path = os.path.join(output_dir, 'attention_vis.png')
    visualize_attention(best_model, test_loader, device, attn_vis_path)
    
    # 保存结果摘要
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== BiLSTM+Attention 训练结果摘要 ===\n\n")
        
        f.write("模型参数:\n")
        for key, value in model_params.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\n训练参数:\n")
        for key, value in train_params.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\n交叉验证结果:\n")
        for fold_result in results:
            f.write(f"  Fold {fold_result['fold']}: MSE={fold_result['mse']:.4f}, "
                   f"MAE={fold_result['mae']:.4f}, R2={fold_result['r2']:.4f}, "
                   f"Pearson={fold_result['pearson']:.4f}\n")
        
        f.write(f"\n最佳折: {best_fold + 1}\n")
        
        f.write("\n测试集评估结果:\n")
        f.write(f"  MSE: {eval_metrics[0]:.4f}\n")
        f.write(f"  MAE: {eval_metrics[1]:.4f}\n")
        f.write(f"  R2: {eval_metrics[2]:.4f}\n")
        f.write(f"  Pearson: {eval_metrics[3]:.4f}\n")
    
    logger.info(f"训练完成，所有结果已保存至 {output_dir}")
    
    # 清理缓存
    embedding_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

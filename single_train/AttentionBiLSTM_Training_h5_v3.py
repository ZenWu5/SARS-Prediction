# AttentionBiLSTM_Training_h5_v3.py

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
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
args = parser.parse_args()

# 配置plt字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False


# 检查并设置设备
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
# logger.info(f"使用设备: {device_str}")

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
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, input_dim]
        
        # 使用LSTM处理序列
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_dim*2]
        
        # 计算注意力权重
        attention_scores = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        
        # 如果有掩码，应用到注意力分数（将填充部分的注意力设为极小值）
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

# 定义数据集类
class H5ProteinDataset(Dataset):
    def __init__(self, h5_path, max_seq_len=None, normalize=True):
        """
        按需从 HDF5 文件加载单条样本。仅支持新版数据结构（embeddings, mean_log10Ka, diff_count）。
        """
        self.h5_path = h5_path if h5_path.endswith('.h5') else h5_path + '.h5'
        with h5py.File(self.h5_path, 'r') as f:
            self.total = int(f.attrs['total_samples'])
            self.emb_dim = int(f.attrs['embedding_dim'])
            # 只加载目标和diff_count
            self.targets = f['mean_log10Ka'][:].astype(np.float32)      # (N,)
            self.diff_counts = f['diff_count'][:].astype(np.int16)       # (N,)
            # 统计所有序列长度
            self.seq_lengths = np.array([f['embeddings'][f'emb_{i}'].shape[0] for i in range(self.total)], dtype=np.int32)
        self.max_seq_len = max_seq_len or int(self.seq_lengths.max())
        self._file = None

        self.normalize = normalize
        self.feature_mean = None
        self.feature_std = None

        if normalize:
            self._compute_normalization_stats()

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')
        emb = self._file['embeddings'][f'emb_{idx}'][:].astype(np.float32)  # (L, D)
        if self.normalize and self.feature_mean is not None:
            emb = (emb - self.feature_mean) / self.feature_std
        L = emb.shape[0]
        # pad/truncate
        if L > self.max_seq_len:
            proc = emb[:self.max_seq_len]
            mask = np.ones(self.max_seq_len, dtype=bool)
        else:
            pad_len = self.max_seq_len - L
            proc = np.vstack([emb, np.zeros((pad_len, self.emb_dim), dtype=np.float32)])
            mask = np.concatenate([np.ones(L, dtype=bool), np.zeros(pad_len, dtype=bool)])
        target = float(self.targets[idx])
        # 返回三元组：feature, mask, target
        return (
            torch.from_numpy(proc),           # float32 tensor (max_seq_len, D)
            torch.from_numpy(mask),           # bool tensor  (max_seq_len,)
            torch.tensor(target, dtype=torch.float32)
        )

    def _compute_normalization_stats(self):
        """计算所有嵌入向量的均值和标准差，用于标准化"""
        logger.info("计算嵌入向量的标准化统计量...")
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
        logger.info(f"特征归一化统计量计算完成: 均值范围[{self.feature_mean.min():.4f}, {self.feature_mean.max():.4f}], "
                    f"标准差范围[{self.feature_std.min():.4f}, {self.feature_std.max():.4f}]")

    def get_diff_counts(self):
        """返回所有样本的diff_count（int8）"""
        return self.diff_counts

# 训练单个模型（使用DataLoader方式）
def train_model(train_loader, val_loader, model_params, training_params):
    """
    使用DataLoader训练模型，集成 AMP 和 TF32 支持
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 获取输入维度
    for x_batch, _, _ in train_loader:
        input_dim = x_batch.shape[2]
        break

    model = AttentionBiLSTM(
        input_dim=input_dim,
        hidden_dim=model_params['hidden_dim'],
        num_layers=model_params['num_layers'],
        dropout=model_params['dropout']
    ).to(device)

    optimizer = Adam(model.parameters(), lr=training_params['lr'])
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    # 定义动态学习率策略
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # 线性上升
        else:
            # return 0.95 ** (epoch - warmup_epochs)  # 指数衰减
            # 余弦退火阶段
            min_lr_factor = 1e-3
            total_epochs = training_params['epochs']
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_factor + (1 - min_lr_factor) * cosine_decay
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
        'val_pearson': []
    }

    for epoch in range(training_params['epochs']):
        model.train()
        train_losses = []

        for X_batch, mask_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{training_params['epochs']}"):
            X_batch = X_batch.to(device)
            mask_batch = mask_batch.to(device)
            y_batch = y_batch.to(device)

            with torch.cuda.amp.autocast(dtype=torch.float32):
                y_pred, _ = model(X_batch, mask_batch)
                y_pred = y_pred.squeeze()
                loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        history['train_loss'].append(avg_train_loss)

        # 验证阶段（不使用 AMP）
        model.eval()
        val_losses, val_preds, val_targets = [], [], []

        for X_val_batch, mask_val_batch, y_val_batch in val_loader:
            X_val_batch = X_val_batch.to(device)
            mask_val_batch = mask_val_batch.to(device)
            y_val_batch = y_val_batch.to(device)

            with torch.no_grad():
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

        scheduler.step()
        logger.info(f"Epoch {epoch + 1}: 当前学习率 = {optimizer.param_groups[0]['lr']:.6f}")
        logger.info(f"Epoch {epoch + 1}: 训练损失={avg_train_loss:.6f}, 验证损失={avg_val_loss:.6f}, "
                    f"MSE={val_mse:.6f}, MAE={val_mae:.6f}, R²={val_r2:.6f}, Pearson={val_pearson:.6f}")

        history['val_loss'].append(avg_val_loss)
        history['val_mse'].append(val_mse)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)
        history['val_pearson'].append(val_pearson)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            logger.info("新的最佳模型已保存!")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"早停触发！{patience}个epoch没有改善")
            break

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
            
        X_batch = X_batch.to(device)
        mask_batch = mask_batch.to(device)
        
        with torch.no_grad():
            pred, attention = model(X_batch, mask_batch)
            
        for i in range(min(len(X_batch), n_samples - len(samples))):
            # 获取真实序列长度（非填充部分）
            seq_len = mask_batch[i].sum().cpu().item()
            
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
    plt.figure(figsize=(12, 10))
    
    # 绘制每个折叠的训练和验证损失
    plt.subplot(2, 2, 1)
    for i, history in enumerate(histories):
        plt.plot(history['train_loss'], '--', label=f'Fold {i+1} 训练')
        plt.plot(history['val_loss'], label=f'Fold {i+1} 验证')
    plt.title('训练和验证损失')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制每个折叠的验证MSE
    plt.subplot(2, 2, 2)
    for i, history in enumerate(histories):
        style = '-' if (best_fold is not None and i == best_fold) else '--'
        plt.plot(history['val_mse'], style, label=f'Fold {i+1}' + (' (最佳)' if i == best_fold else ''))
    plt.title('验证MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    # 绘制每个折叠的验证R2
    plt.subplot(2, 2, 3)
    for i, history in enumerate(histories):
        style = '-' if (best_fold is not None and i == best_fold) else '--'
        plt.plot(history['val_r2'], style, label=f'Fold {i+1}' + (' (最佳)' if i == best_fold else ''))
    plt.title('验证R²')
    plt.xlabel('Epochs')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)
    
    # 绘制每个折叠的验证Pearson相关系数
    plt.subplot(2, 2, 4)
    for i, history in enumerate(histories):
        style = '-' if (best_fold is not None and i == best_fold) else '--'
        plt.plot(history['val_pearson'], style, label=f'Fold {i+1}' + (' (最佳)' if i == best_fold else ''))
    plt.title('验证Pearson相关系数')
    plt.xlabel('Epochs')
    plt.ylabel('Pearson')
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
    
    for X_batch, mask_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        mask_batch = mask_batch.to(device)
        
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

# 将 Subset 数据加载到内存中
class InMemoryDataset(Dataset):
    def __init__(self, features, masks, targets):
        self.features = features
        self.masks = masks
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.masks[idx], self.targets[idx]

def load_subset_to_memory(subset: Subset) -> Dataset:
    """
    分块加载，不堆叠成大 Tensor，从而避免内存瞬时占用峰值。
    """
    features, masks, targets = [], [], []
    for i in tqdm(range(len(subset)), desc="逐条加载子集"):
        feat, mask, target = subset[i]
        features.append(feat)       # 不 stack
        masks.append(mask)
        targets.append(target)
    return InMemoryDataset(features, masks, targets)
# 主函数
def main():
    # 1. 解析命令行参数
    args = parser.parse_args()

    # 2. 设置随机种子
    set_seed(args.seed)
    logger.info(f"设置随机种子: {args.seed}")

    # 启用 TF32 加速（适用于 RTX 30 系列以上显卡）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


    # 3. 创建结果输出目录
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"结果将保存至: {output_dir}")

    # 4. 构造按需加载的 HDF5 Dataset
    dataset = H5ProteinDataset(args.data_path, max_seq_len=args.max_seq_len, normalize=True)
    N = len(dataset)
    logger.info(f"HDF5 样本总数: {N}, pad 长度={dataset.max_seq_len}, emb_dim={dataset.emb_dim}")

    # 5. 划分训练/测试集（15% 测试）
    perm = np.random.permutation(N)
    n_test = int(0.15 * N)
    test_idx = perm[:n_test]
    train_val_idx = perm[n_test:]

    test_set = Subset(dataset, test_idx)
    train_val_set = Subset(dataset, train_val_idx)
    logger.info(f"切分完成：训练+验证={len(train_val_set)}, 测试={len(test_set)}")

    # 6. 准备模型参数和训练参数
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

    # 7. K 折交叉验证
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    fold_results = []
    all_histories = []
    best_model, best_mse, best_fold = None, float('inf'), None

    for fold, (train_subidx, val_subidx) in enumerate(kf.split(train_val_idx), 1):
        logger.info(f"\n======= 第 {fold}/{args.k_folds} 折 =======")

        # 加载到内存（替代 Subset + HDF5 lazy access）
        train_subset = Subset(train_val_set, train_subidx)
        val_subset = Subset(train_val_set, val_subidx)

        train_tensor_ds = load_subset_to_memory(train_subset)
        val_tensor_ds = load_subset_to_memory(val_subset)

        train_loader = DataLoader(
            train_tensor_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,     # 数据已在内存，无需 worker 并发
            pin_memory=True
        )
        val_loader = DataLoader(
            val_tensor_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        model, val_res, history = train_model(
            train_loader, val_loader,
            model_params, training_params
        )
        fold_results.append(val_res)
        all_histories.append(history)

        logger.info(f"第 {fold} 折 验证 MSE={val_res['mse']:.6f}")

        if val_res['mse'] < best_mse:
            best_mse = val_res['mse']
            best_model = model
            best_fold = fold - 1  # 索引从0开始

        # 清理显存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 8. 保存训练历史可视化
    history_fig = plot_training_history(all_histories, best_fold)
    history_path = os.path.join(output_dir, f'history_{timestamp}.png')
    history_fig.savefig(history_path)
    logger.info(f"训练历史图已保存至: {history_path}")
    plt.close(history_fig)

    # 9. 在测试集上评估最佳模型
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )
    evaluate_model(best_model, test_loader, output_dir)

    # 10. 可视化部分测试样本的注意力权重
    att_fig = visualize_attention(
        best_model, test_loader,
        n_samples=5
    )
    att_path = os.path.join(output_dir, f'attention_{timestamp}.png')
    att_fig.savefig(att_path)
    logger.info(f"注意力可视化已保存至: {att_path}")
    plt.close(att_fig)

    # 11. 保存模型与结果摘要
    model_path = os.path.join(output_dir, 'best_model.pt')
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'model_params': model_params
    }, model_path)
    logger.info(f"最佳模型已保存至: {model_path}")

    summary_path = os.path.join(output_dir, f'results_summary_{timestamp}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("======= BiLSTM+Attention 训练结果摘要 =======\n")
        f.write(f"最佳折: {best_fold + 1}/{args.k_folds}\n")
        f.write(f"测试集 MSE: {best_mse:.6f}\n")
        for i, res in enumerate(fold_results):
            f.write(f"第 {i+1} 折: MSE={res['mse']:.6f}, MAE={res['mae']:.6f}, "
                    f"R²={res['r2']:.6f}, Pearson={res['pearson']:.6f}\n")
    logger.info(f"结果摘要已保存至: {summary_path}")


if __name__ == "__main__":
    main()

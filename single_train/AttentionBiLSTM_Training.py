# AttentionBiLSTM_Training.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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

# 配置命令行参数
parser = argparse.ArgumentParser(description='使用带注意力的BiLSTM模型训练蛋白质序列嵌入数据')
parser.add_argument('--data_path', type=str, default='preprocess/ESM/output/sampled_output_esm_embeddings.npz',
                    help='NPZ数据文件路径')
parser.add_argument('--max_seq_len', type=int, default=None, 
                    help='最大序列长度，默认使用数据集中的最大长度')
parser.add_argument('--hidden_dim', type=int, default=128, 
                    help='LSTM隐藏层维度')
parser.add_argument('--num_layers', type=int, default=2, 
                    help='LSTM层数')
parser.add_argument('--dropout', type=float, default=0.2, 
                    help='Dropout比例')
parser.add_argument('--lr', type=float, default=0.01, 
                    help='学习率')
parser.add_argument('--batch_size', type=int, default=128, 
                    help='批次大小')
parser.add_argument('--epochs', type=int, default=100, 
                    help='最大训练轮数')
parser.add_argument('--patience', type=int, default=10, 
                    help='早停耐心值')
parser.add_argument('--k_folds', type=int, default=5, 
                    help='交叉验证折数')
parser.add_argument('--seed', type=int, default=42, 
                    help='随机种子')
parser.add_argument('--output_dir', type=str, default='ModelResults/BiLSTMResults', 
                    help='结果输出目录')
args = parser.parse_args()

# 配置plt字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger("bilstm_attention_training")

# 检查并设置设备
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
logger.info(f"使用设备: {device_str}")

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

# 数据加载和预处理函数
def load_and_preprocess_data(data_path, max_seq_len=None):
    try:
        logger.info(f"尝试加载NPZ文件: {data_path}")
        
        # 加载NPZ文件
        data = np.load(data_path, allow_pickle=True)
        
        # 提取NPZ文件中的数据
        embeddings_list = data['embeddings']
        targets = data['mean_log10Ka']
        
        logger.info(f"加载了 {len(embeddings_list)} 条蛋白质序列记录和对应标签")
        
        # 统计序列长度信息
        seq_lengths = [emb.shape[0] for emb in embeddings_list]
        min_len = min(seq_lengths)
        max_len = max(seq_lengths)
        avg_len = sum(seq_lengths) / len(seq_lengths)
        
        logger.info(f"序列统计信息: 最短={min_len}, 最长={max_len}, 平均={avg_len:.2f}")
        
        # 如果未指定max_seq_len，使用数据集中的最大长度
        if max_seq_len is None:
            max_seq_len = max_len
            logger.info(f"使用数据集中的最大长度: {max_seq_len}")
        else:
            logger.info(f"使用指定的最大长度: {max_seq_len}")
        
        # 获取嵌入向量的维度
        embedding_dim = embeddings_list[0].shape[1]
        logger.info(f"嵌入向量维度: {embedding_dim}")
        
        # 对序列进行填充或截断处理
        processed_embeddings = []
        mask_list = []  # 创建掩码列表，用于标识实际序列vs填充
        
        for embedding in embeddings_list:
            seq_len = embedding.shape[0]
            # 创建掩码 (1表示实际序列，0表示填充)
            mask = np.ones(max_seq_len, dtype=np.bool_)
            
            # 如果序列长度超过最大长度，则截断
            if seq_len > max_seq_len:
                processed_embedding = embedding[:max_seq_len, :]
            # 如果序列长度小于最大长度，则填充
            elif seq_len < max_seq_len:
                padding = np.zeros((max_seq_len - seq_len, embedding_dim), dtype=np.float32)
                processed_embedding = np.vstack([embedding, padding])
                # 更新掩码，填充部分设置为0
                mask[seq_len:] = 0
            else:
                processed_embedding = embedding
            
            processed_embeddings.append(processed_embedding)
            mask_list.append(mask)
        
        # 转换为numpy数组
        features = np.array(processed_embeddings, dtype=np.float32)
        masks = np.array(mask_list, dtype=np.bool_)
        targets = targets.astype(np.float32)
        
        logger.info(f"处理后的特征形状: {features.shape}")
        logger.info(f"掩码形状: {masks.shape}")
        logger.info(f"标签形状: {targets.shape}")
        
        return features, masks, targets
    
    except Exception as e:
        logger.error(f"加载或预处理数据时出错: {str(e)}")
        return None, None, None

# 训练单个模型
def train_model(X_train, masks_train, y_train, X_val, masks_val, y_val, model_params, training_params):
    """
    训练带注意力的BiLSTM模型并返回验证集性能（含动态学习率调度）
    """
    import math

    # 清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 准备训练数据加载器
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(masks_train, dtype=torch.bool),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_params['batch_size'],
        shuffle=True
    )

    # 初始化模型
    model = AttentionBiLSTM(
        input_dim=X_train.shape[2],
        hidden_dim=model_params['hidden_dim'],
        num_layers=model_params['num_layers'],
        dropout=model_params['dropout']
    ).to(device)

    # 优化器和损失函数
    optimizer = Adam(model.parameters(), lr=training_params['lr'])
    criterion = nn.MSELoss()

    # 定义动态学习率策略（Linear Warmup + Exponential Decay）
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # 线性上升
        else:
            return 0.95 ** (epoch - warmup_epochs)  # 指数衰减

    scheduler = LambdaLR(optimizer, lr_lambda)

    # 初始化记录
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

            y_pred, _ = model(X_batch, mask_batch)
            y_pred = y_pred.squeeze()
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        history['train_loss'].append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(masks_val, dtype=torch.bool),
            torch.tensor(y_val, dtype=torch.float32)
        )
        val_loader = DataLoader(val_dataset, batch_size=training_params['batch_size'])

        val_losses = []
        val_preds = []
        val_targets = []

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
        avg_val_loss = sum(val_losses) / len(val_losses)
        val_mse = mean_squared_error(val_targets, val_preds)
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_r2 = r2_score(val_targets, val_preds)
        val_pearson, _ = pearsonr(val_targets, val_preds)

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch + 1}: 当前学习率 = {current_lr:.6f}")

        # 记录指标
        history['val_loss'].append(avg_val_loss)
        history['val_mse'].append(val_mse)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)
        history['val_pearson'].append(val_pearson)

        logger.info(f"Epoch {epoch + 1}: 训练损失={avg_train_loss:.6f}, 验证损失={avg_val_loss:.6f}, "
                    f"MSE={val_mse:.6f}, MAE={val_mae:.6f}, R²={val_r2:.6f}, Pearson={val_pearson:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            logger.info(f"新的最佳模型已保存!")
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


# def train_model(X_train, masks_train, y_train, X_val, masks_val, y_val, model_params, training_params):
#     """
#     训练带注意力的BiLSTM模型并返回验证集性能（含动态学习率调度）
#     """
#     import math

#     # 清理内存
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     # 准备训练数据加载器
#     train_dataset = TensorDataset(
#         torch.tensor(X_train, dtype=torch.float32),
#         torch.tensor(masks_train, dtype=torch.bool),
#         torch.tensor(y_train, dtype=torch.float32)
#     )
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=training_params['batch_size'],
#         shuffle=True
#     )

#     # 初始化模型
#     model = AttentionBiLSTM(
#         input_dim=X_train.shape[2],
#         hidden_dim=model_params['hidden_dim'],
#         num_layers=model_params['num_layers'],
#         dropout=model_params['dropout']
#     ).to(device)

#     # 优化器和损失函数
#     optimizer = Adam(model.parameters(), lr=training_params['lr'])
#     criterion = nn.MSELoss()

#     # 定义动态学习率策略（Linear Warmup + Exponential Decay）
#     warmup_epochs = 5
#     def lr_lambda(epoch):
#         if epoch < warmup_epochs:
#             return (epoch + 1) / warmup_epochs  # 线性上升
#         else:
#             return 0.95 ** (epoch - warmup_epochs)  # 指数衰减

#     scheduler = LambdaLR(optimizer, lr_lambda)

#     # 初始化记录
#     best_val_loss = float('inf')
#     best_model_state = None
#     patience_counter = 0
#     patience = training_params['patience']

#     history = {
#         'train_loss': [],
#         'val_loss': [],
#         'val_mse': [],
#         'val_mae': [],
#         'val_r2': [],
#         'val_pearson': []
#     }

#     for epoch in range(training_params['epochs']):
#         model.train()
#         train_losses = []

#         for X_batch, mask_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{training_params['epochs']}"):
#             X_batch = X_batch.to(device)
#             mask_batch = mask_batch.to(device)
#             y_batch = y_batch.to(device)

#             y_pred, _ = model(X_batch, mask_batch)
#             y_pred = y_pred.squeeze()
#             loss = criterion(y_pred, y_batch)

#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()

#             train_losses.append(loss.item())

#         avg_train_loss = sum(train_losses) / len(train_losses)
#         history['train_loss'].append(avg_train_loss)

#         # 验证阶段
#         model.eval()
#         val_dataset = TensorDataset(
#             torch.tensor(X_val, dtype=torch.float32),
#             torch.tensor(masks_val, dtype=torch.bool),
#             torch.tensor(y_val, dtype=torch.float32)
#         )
#         val_loader = DataLoader(val_dataset, batch_size=training_params['batch_size'])

#         val_losses = []
#         val_preds = []
#         val_targets = []

#         for X_val_batch, mask_val_batch, y_val_batch in val_loader:
#             X_val_batch = X_val_batch.to(device)
#             mask_val_batch = mask_val_batch.to(device)
#             y_val_batch = y_val_batch.to(device)

#             with torch.no_grad():
#                 val_pred, _ = model(X_val_batch, mask_val_batch)
#                 val_pred = val_pred.squeeze()
#                 val_loss = criterion(val_pred, y_val_batch)
#                 val_losses.append(val_loss.item())

#                 val_preds.append(val_pred.cpu().numpy())
#                 val_targets.append(y_val_batch.cpu().numpy())

#         val_preds = np.concatenate(val_preds)
#         val_targets = np.concatenate(val_targets)
#         avg_val_loss = sum(val_losses) / len(val_losses)
#         val_mse = mean_squared_error(val_targets, val_preds)
#         val_mae = mean_absolute_error(val_targets, val_preds)
#         val_r2 = r2_score(val_targets, val_preds)
#         val_pearson, _ = pearsonr(val_targets, val_preds)

#         # 更新学习率
#         scheduler.step()
#         current_lr = optimizer.param_groups[0]['lr']
#         logger.info(f"Epoch {epoch + 1}: 当前学习率 = {current_lr:.6f}")

#         # 记录指标
#         history['val_loss'].append(avg_val_loss)
#         history['val_mse'].append(val_mse)
#         history['val_mae'].append(val_mae)
#         history['val_r2'].append(val_r2)
#         history['val_pearson'].append(val_pearson)

#         logger.info(f"Epoch {epoch + 1}: 训练损失={avg_train_loss:.6f}, 验证损失={avg_val_loss:.6f}, "
#                     f"MSE={val_mse:.6f}, MAE={val_mae:.6f}, R²={val_r2:.6f}, Pearson={val_pearson:.6f}")

#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_model_state = model.state_dict().copy()
#             patience_counter = 0
#             logger.info(f"新的最佳模型已保存!")
#         else:
#             patience_counter += 1

#         if patience_counter >= patience:
#             logger.info(f"早停触发！{patience}个epoch没有改善")
#             break

#     # 加载最佳模型
#     model.load_state_dict(best_model_state)

#     validation_results = {
#         'loss': best_val_loss,
#         'mse': val_mse,
#         'mae': val_mae,
#         'r2': val_r2,
#         'pearson': val_pearson
#     }

#     return model, validation_results, history
# # def train_model(X_train, masks_train, y_train, X_val, masks_val, y_val, model_params, training_params):
#     """
#     使用 ReduceLROnPlateau 策略训练 BiLSTM+Attention 模型，兼容旧版本 PyTorch
#     """
#     # 清理内存
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     # 数据加载器
#     train_dataset = TensorDataset(
#         torch.tensor(X_train, dtype=torch.float32),
#         torch.tensor(masks_train, dtype=torch.bool),
#         torch.tensor(y_train, dtype=torch.float32)
#     )
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=training_params['batch_size'],
#         shuffle=True
#     )

#     # 模型初始化
#     model = AttentionBiLSTM(
#         input_dim=X_train.shape[2],
#         hidden_dim=model_params['hidden_dim'],
#         num_layers=model_params['num_layers'],
#         dropout=model_params['dropout']
#     ).to(device)

#     optimizer = Adam(model.parameters(), lr=training_params['lr'])
#     criterion = nn.MSELoss()

#     # 使用 ReduceLROnPlateau（不带 verbose）
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

#     best_val_loss = float('inf')
#     best_model_state = None
#     patience_counter = 0
#     patience = training_params['patience']

#     history = {
#         'train_loss': [],
#         'val_loss': [],
#         'val_mse': [],
#         'val_mae': [],
#         'val_r2': [],
#         'val_pearson': []
#     }

#     for epoch in range(training_params['epochs']):
#         model.train()
#         train_losses = []

#         for X_batch, mask_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{training_params['epochs']}"):
#             X_batch = X_batch.to(device)
#             mask_batch = mask_batch.to(device)
#             y_batch = y_batch.to(device)

#             y_pred, _ = model(X_batch, mask_batch)
#             y_pred = y_pred.squeeze()
#             loss = criterion(y_pred, y_batch)

#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()

#             train_losses.append(loss.item())

#         avg_train_loss = sum(train_losses) / len(train_losses)
#         history['train_loss'].append(avg_train_loss)

#         # 验证阶段
#         model.eval()
#         val_dataset = TensorDataset(
#             torch.tensor(X_val, dtype=torch.float32),
#             torch.tensor(masks_val, dtype=torch.bool),
#             torch.tensor(y_val, dtype=torch.float32)
#         )
#         val_loader = DataLoader(val_dataset, batch_size=training_params['batch_size'])

#         val_losses = []
#         val_preds = []
#         val_targets = []

#         for X_val_batch, mask_val_batch, y_val_batch in val_loader:
#             X_val_batch = X_val_batch.to(device)
#             mask_val_batch = mask_val_batch.to(device)
#             y_val_batch = y_val_batch.to(device)

#             with torch.no_grad():
#                 val_pred, _ = model(X_val_batch, mask_val_batch)
#                 val_pred = val_pred.squeeze()
#                 val_loss = criterion(val_pred, y_val_batch)
#                 val_losses.append(val_loss.item())

#                 val_preds.append(val_pred.cpu().numpy())
#                 val_targets.append(y_val_batch.cpu().numpy())

#         val_preds = np.concatenate(val_preds)
#         val_targets = np.concatenate(val_targets)
#         avg_val_loss = sum(val_losses) / len(val_losses)
#         val_mse = mean_squared_error(val_targets, val_preds)
#         val_mae = mean_absolute_error(val_targets, val_preds)
#         val_r2 = r2_score(val_targets, val_preds)
#         val_pearson, _ = pearsonr(val_targets, val_preds)

#         # 调度器监控验证损失
#         scheduler.step(avg_val_loss)
#         current_lr = optimizer.param_groups[0]['lr']
#         logger.info(f"Epoch {epoch + 1}: 当前学习率 = {current_lr:.6f}")

#         # 记录指标
#         history['val_loss'].append(avg_val_loss)
#         history['val_mse'].append(val_mse)
#         history['val_mae'].append(val_mae)
#         history['val_r2'].append(val_r2)
#         history['val_pearson'].append(val_pearson)

#         logger.info(f"Epoch {epoch + 1}: 训练损失={avg_train_loss:.6f}, 验证损失={avg_val_loss:.6f}, "
#                     f"MSE={val_mse:.6f}, MAE={val_mae:.6f}, R²={val_r2:.6f}, Pearson={val_pearson:.6f}")

#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_model_state = model.state_dict().copy()
#             patience_counter = 0
#             logger.info(f"新的最佳模型已保存!")
#         else:
#             patience_counter += 1

#         if patience_counter >= patience:
#             logger.info(f"早停触发！{patience}个epoch没有改善")
#             break

#     model.load_state_dict(best_model_state)

#     validation_results = {
#         'loss': best_val_loss,
#         'mse': val_mse,
#         'mae': val_mae,
#         'r2': val_r2,
#         'pearson': val_pearson
#     }

#     return model, validation_results, history

# 使用注意力权重可视化序列重要性
def visualize_attention(model, X, mask, y_true, sequences=None, n_samples=5):
    """
    可视化注意力权重，显示序列中重要的部分
    """
    model.eval()
    
    # 随机选择n_samples个样本
    indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    
    plt.figure(figsize=(12, 4 * n_samples))
    
    for i, idx in enumerate(indices):
        x = torch.tensor(X[idx:idx+1], dtype=torch.float32).to(device)
        m = torch.tensor(mask[idx:idx+1], dtype=torch.bool).to(device)
        
        with torch.no_grad():
            pred, attention = model(x, m)
            
        # 获取真实序列长度（非填充部分）
        if m is not None:
            seq_len = m[0].sum().cpu().item()
        else:
            seq_len = len(x[0])
            
        # 提取注意力权重
        att_weights = attention[0, :seq_len, 0].cpu().numpy()
        
        # 创建子图
        plt.subplot(n_samples, 1, i + 1)
        
        # 绘制注意力权重
        plt.bar(range(seq_len), att_weights)
        plt.xlabel('序列位置')
        plt.ylabel('注意力权重')
        
        # 添加真实值和预测值
        pred_value = pred.item()
        true_value = y_true[idx]
        plt.title(f'样本 #{idx}: 真实值={true_value:.4f}, 预测值={pred_value:.4f}')
        
        # 如果提供了序列信息，显示序列
        if sequences is not None and idx < len(sequences):
            seq = sequences[idx]
            plt.xticks(range(seq_len), seq[:seq_len], rotation=45)
    
    plt.tight_layout()
    return plt.gcf()

# K折交叉验证训练
def train_with_kfold(features, masks, targets, model_params, training_params, k=5):
    """使用K折交叉验证训练BiLSTM模型"""
    kf = KFold(n_splits=k, shuffle=True, random_state=args.seed)

    # 存储每个折叠的性能
    fold_results = []
    best_model = None
    best_mse = float('inf')
    all_histories = []  # 存储所有折叠的训练历史

    for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
        logger.info(f"\n{'=' * 50}\n开始训练第 {fold + 1}/{k} 折\n{'=' * 50}")

        # 准备当前折叠的数据
        X_train, masks_train, y_train = features[train_idx], masks[train_idx], targets[train_idx]
        X_val, masks_val, y_val = features[val_idx], masks[val_idx], targets[val_idx]

        logger.info(f"训练集数据形状: {X_train.shape}, 验证集数据形状: {X_val.shape}")

        # 训练模型
        start_time = time.time()
        model, val_results, history = train_model(
            X_train, masks_train, y_train, 
            X_val, masks_val, y_val, 
            model_params, training_params
        )
        training_time = time.time() - start_time
        
        # 存储训练历史
        all_histories.append(history)

        # 记录结果
        fold_result = {
            'fold': fold + 1,
            'val_loss': val_results['loss'],
            'val_mse': val_results['mse'],
            'val_mae': val_results['mae'],
            'val_r2': val_results['r2'],
            'val_pearson': val_results['pearson'],
            'time': training_time
        }
        fold_results.append(fold_result)

        logger.info(f"第 {fold + 1} 折结果: MSE={val_results['mse']:.6f}, MAE={val_results['mae']:.6f}, "
                   f"R²={val_results['r2']:.6f}, Pearson={val_results['pearson']:.6f}, "
                   f"训练时间: {training_time:.2f}秒")

        # 判断是否是最佳模型
        if val_results['mse'] < best_mse:
            best_mse = val_results['mse']
            best_model = model
            best_fold = fold

        # 在每个折叠后释放内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 计算平均性能
    avg_mse = sum(fold['val_mse'] for fold in fold_results) / k
    avg_mae = sum(fold['val_mae'] for fold in fold_results) / k
    avg_r2 = sum(fold['val_r2'] for fold in fold_results) / k
    avg_pearson = sum(fold['val_pearson'] for fold in fold_results) / k
    avg_time = sum(fold['time'] for fold in fold_results) / k

    logger.info(f"\n{'=' * 50}")
    logger.info(f"交叉验证平均结果: MSE={avg_mse:.6f}, MAE={avg_mae:.6f}, "
               f"R²={avg_r2:.6f}, Pearson={avg_pearson:.6f}, "
               f"平均训练时间: {avg_time:.2f}秒")

    return best_model, fold_results, all_histories, best_fold

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

# 测试最终模型性能
def evaluate_model(model, X_test, masks_test, y_test, batch_size=32, diff_counts=None, output_dir=None):
    """在测试集上评估模型性能，并按 diff_counts 绘图着色"""
    model.eval()

    # 准备测试数据加载器
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(masks_test, dtype=torch.bool),
        torch.tensor(y_test, dtype=torch.float32)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 收集预测和注意力权重
    test_preds = []
    test_attentions = []
    test_targets = []
    
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

    # 绘制预测vs真实值散点图（带颜色编码）
    plt.figure(figsize=(10, 6))
    
    if diff_counts is not None:
        scatter = plt.scatter(test_targets, test_preds, c=diff_counts, cmap='viridis', alpha=0.7)
        cbar = plt.colorbar(scatter)
        cbar.set_label('diff_counts')
    else:
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
    figtime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    fig_output_dir = os.path.join(output_dir,f'bilstm_attention_predictions_{figtime}.png')
    plt.savefig(fig_output_dir)
    logger.info(f"预测结果可视化已保存至 '{fig_output_dir}'")

    return test_mse, test_mae, test_r2, test_pearson, test_preds, test_attentions

# 主函数
def main():
    # 设置随机种子
    set_seed(args.seed)
    logger.info(f"设置随机种子: {args.seed}")

    output_dir = os.path.join(args.output_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())))

    # 创建结果目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info(f"结果将保存至: {output_dir}")

    # 加载数据
    features, masks, targets = load_and_preprocess_data(args.data_path, args.max_seq_len)
    if features is None or masks is None or targets is None:
        return

    # 数据分割
    indices = np.random.permutation(len(features))
    test_size = int(0.15 * len(features))

    test_indices = indices[:test_size]
    train_val_indices = indices[test_size:]

    X_test, masks_test, y_test = features[test_indices], masks[test_indices], targets[test_indices]
    X_train_val, masks_train_val, y_train_val = features[train_val_indices], masks[train_val_indices], targets[train_val_indices]

    logger.info(f"数据分割完成: 训练+验证集={X_train_val.shape}, 测试集={X_test.shape}")

    # 模型参数
    model_params = {
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout
    }
    
    # 训练参数
    training_params = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience
    }

    logger.info(f"模型参数: {model_params}")
    logger.info(f"训练参数: {training_params}")

    # 训练模型
    logger.info("\n开始交叉验证训练...")
    best_model, fold_results, all_histories, best_fold = train_with_kfold(
        X_train_val, masks_train_val, y_train_val, 
        model_params, training_params, k=args.k_folds
    )

    # 绘制训练历史并保存
    history_fig = plot_training_history(all_histories, best_fold)
    history_figtime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    history_fig_path = os.path.join(output_dir, f'bilstm_attention_history_{history_figtime}.png')
    history_fig.savefig(history_fig_path)
    logger.info(f"训练历史可视化已保存至: {history_fig_path}")
    plt.close(history_fig)

    # 保存最佳模型
    model_path = os.path.join(output_dir, 'best_bilstm_attention_model.pt')
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'model_params': model_params,
        'input_dim': X_train_val.shape[2]
    }, model_path)
    logger.info(f"最佳模型已保存至: {model_path}")

    # 评估最佳模型
    logger.info("\n在测试集上评估最佳模型...")

    data = np.load(args.data_path, allow_pickle=True)
    diff_counts = data.get('diff_counts')
    if diff_counts is not None:
        diff_counts = diff_counts.astype(int)
        test_diff_counts = diff_counts[test_indices]
    else:
        test_diff_counts = None

    # 调用evaluate_model
    test_mse, test_mae, test_r2, test_pearson, test_preds, test_attentions = evaluate_model(
        best_model, X_test, masks_test, y_test, batch_size=args.batch_size, diff_counts=test_diff_counts, output_dir=output_dir
    )

    
    # 可视化部分测试样本的注意力权重 (假设有序列数据)
    try:
        # 这里尝试从NPZ文件加载原始序列数据，如果不可用则跳过
        sequences = np.load(args.data_path, allow_pickle=True).get('sequences')
        test_sequences = sequences[test_indices] if sequences is not None else None
        
        if test_sequences is not None:
            att_fig = visualize_attention(best_model, X_test, masks_test, y_test, 
                                          sequences=test_sequences, n_samples=5)
            att_figtime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
            att_fig_path = os.path.join(output_dir, f'attention_visualization_{att_figtime}.png')
            att_fig.savefig(att_fig_path)
            logger.info(f"注意力可视化已保存至: {att_fig_path}")
            plt.close(att_fig)
    except Exception as e:
        logger.warning(f"注意力可视化失败: {str(e)}")

    # 保存结果摘要
    logtime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    summary_file = os.path.join(output_dir, f"bilstm_attention_results_{logtime}.txt")
    with open(summary_file, 'w', encoding="UTF-8") as f:
        f.write("============== BiLSTM+Attention模型训练结果摘要 ==============\n\n")
        f.write(f"数据文件: {args.data_path}\n")
        f.write(f"最大序列长度: {args.max_seq_len if args.max_seq_len else '数据集最大长度'}\n\n")
        f.write(f"模型参数: {model_params}\n")
        f.write(f"训练参数: {training_params}\n\n")

        f.write("交叉验证结果:\n")
        for fold in fold_results:
            f.write(f"  - 第 {fold['fold']} 折: MSE={fold['val_mse']:.6f}, MAE={fold['val_mae']:.6f}, "
                    f"R²={fold['val_r2']:.6f}, Pearson={fold['val_pearson']:.6f}\n")

        avg_mse = sum(fold['val_mse'] for fold in fold_results) / len(fold_results)
        avg_mae = sum(fold['val_mae'] for fold in fold_results) / len(fold_results)
        avg_r2 = sum(fold['val_r2'] for fold in fold_results) / len(fold_results)
        avg_pearson = sum(fold['val_pearson'] for fold in fold_results) / len(fold_results)
        
        f.write(f"\n交叉验证平均性能: MSE={avg_mse:.6f}, MAE={avg_mae:.6f}, "
                f"R²={avg_r2:.6f}, Pearson={avg_pearson:.6f}\n\n")

        f.write(f"测试集性能: MSE={test_mse:.6f}, MAE={test_mae:.6f}, "
                f"R²={test_r2:.6f}, Pearson={test_pearson:.6f}\n")

    logger.info(f"结果摘要已保存至: {summary_file}")
    logger.info("\n训练和评估完成!")


if __name__ == "__main__":
    main()

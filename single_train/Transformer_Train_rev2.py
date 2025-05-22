# ProteinTransformer_Training.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, lr_scheduler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
import os
import logging
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import argparse
import json

plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei'] 等
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transformer_training.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("transformer_training")

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


# 定义蛋白质序列Transformer模型
class ProteinTransformer(nn.Module):
    def __init__(self, input_dim=100, d_model=128, nhead=8, num_layers=3, 
                 dim_feedforward=512, dropout=0.1, max_seq_len=1000):
        super(ProteinTransformer, self).__init__()
        
        # 输入映射到模型维度
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers
        )
        
        # 输出头
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            src_key_padding_mask: 填充掩码 [batch_size, seq_len] 
                                  True表示填充位置
        Returns:
            输出预测值 [batch_size, 1]
        """
        # 映射到模型维度
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码器
        encoder_output = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )  # [batch_size, seq_len, d_model]
        
        # 通过平均池化汇总序列信息
        if src_key_padding_mask is not None:
            # 使用掩码确保只对有效位置进行平均
            mask_expanded = src_key_padding_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            
            # 有效位置计数 (非填充位置)
            valid_counts = (1.0 - mask_expanded).sum(dim=1)  # [batch, 1]
            
            # 对填充位置的值设为0，然后对非填充位置求和，再除以非填充位置数量
            pooled = ((1.0 - mask_expanded) * encoder_output).sum(dim=1) / (valid_counts + 1e-10)
        else:
            # 没有掩码，直接对所有位置求平均
            pooled = encoder_output.mean(dim=1)  # [batch_size, d_model]
        
        # 输出层
        output = self.output_layer(pooled)  # [batch_size, 1]
        
        return output


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 计算位置编码矩阵
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为缓冲区，这样它就不会作为模型参数
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


# 创建填充掩码
def create_padding_mask(sequences, max_len):
    """
    为序列创建填充掩码
    
    Args:
        sequences: 序列列表，每个元素是一个变长的2D数组
        max_len: 最大序列长度
        
    Returns:
        填充掩码，True表示填充位置
    """
    batch_size = len(sequences)
    masks = torch.ones(batch_size, max_len, dtype=torch.bool)
    
    for i, seq in enumerate(sequences):
        seq_len = seq.shape[0]  # 序列的实际长度
        masks[i, :seq_len] = False  # 实际序列位置设置为False
        
    return masks


# 数据加载和预处理函数
def load_and_preprocess_data(data_path, max_seq_len=None, val_split=0.15):
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
        max_observed_len = max(seq_lengths)
        avg_len = sum(seq_lengths) / len(seq_lengths)
        
        logger.info(f"序列统计信息: 最短={min_len}, 最长={max_observed_len}, 平均={avg_len:.2f}")
        
        # 如果未指定max_seq_len，使用数据集中的最大长度
        if max_seq_len is None:
            max_seq_len = max_observed_len
            
        logger.info(f"使用最大序列长度: {max_seq_len}")
        
        # 获取嵌入向量的维度
        embedding_dim = embeddings_list[0].shape[1]
        logger.info(f"嵌入向量维度: {embedding_dim}")
        
        # 将目标值转换为numpy数组
        targets = np.array(targets, dtype=np.float32)
        
        # 返回未处理的数据和额外信息
        return {
            'embeddings': embeddings_list,
            'targets': targets,
            'max_seq_len': max_seq_len,
            'embedding_dim': embedding_dim,
            'seq_lengths': np.array(seq_lengths)
        }
    
    except Exception as e:
        logger.error(f"加载或预处理数据时出错: {str(e)}")
        raise


# 数据生成器
class ProteinDataGenerator:
    def __init__(self, embeddings, targets, max_seq_len, batch_size=32, shuffle=True):
        self.embeddings = embeddings
        self.targets = targets
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(embeddings))
        
    def __len__(self):
        return (len(self.embeddings) + self.batch_size - 1) // self.batch_size
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        self.on_epoch_end()
        self.index = 0
        return self
    
    def __next__(self):
        if self.index >= len(self.embeddings):
            raise StopIteration
        
        # 获取当前批次的索引
        batch_indices = self.indices[self.index:min(self.index + self.batch_size, len(self.embeddings))]
        self.index += self.batch_size
        
        # 处理序列数据
        batch_embeddings = [self.embeddings[i] for i in batch_indices]
        batch_targets = self.targets[batch_indices]
        
        # 填充序列到相同长度
        padded_embeddings = []
        for emb in batch_embeddings:
            if emb.shape[0] > self.max_seq_len:
                # 截断
                padded_emb = emb[:self.max_seq_len, :]
            else:
                # 填充
                padding = np.zeros((self.max_seq_len - emb.shape[0], emb.shape[1]), dtype=np.float32)
                padded_emb = np.vstack([emb, padding])
            padded_embeddings.append(padded_emb)
        
        # 转换为张量
        X = torch.tensor(np.array(padded_embeddings), dtype=torch.float32)
        y = torch.tensor(batch_targets, dtype=torch.float32)
        
        # 创建序列掩码
        padding_mask = create_padding_mask(batch_embeddings, self.max_seq_len)
        
        return X, y, padding_mask


# 训练单个模型
def train_single_model(train_data, val_data, params, model_dir):
    """训练单个Transformer模型并返回验证集性能"""
    # 清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 解包训练和验证数据
    X_train, y_train = train_data['embeddings'], train_data['targets']
    X_val, y_val = val_data['embeddings'], val_data['targets']
    max_seq_len = params['max_seq_len']
    
    # 创建数据生成器
    train_gen = ProteinDataGenerator(
        X_train, y_train, max_seq_len, batch_size=params['batch_size'], shuffle=True
    )
    val_gen = ProteinDataGenerator(
        X_val, y_val, max_seq_len, batch_size=params['batch_size'], shuffle=False
    )
    
    # 初始化模型
    model = ProteinTransformer(
        input_dim=train_data['embedding_dim'],
        d_model=params['d_model'],
        nhead=params['nhead'],
        num_layers=params['num_layers'],
        dim_feedforward=params['dim_feedforward'],
        dropout=params['dropout'],
        max_seq_len=max_seq_len
    ).to(device)
    
    # 输出模型结构
    logger.info(f"模型结构:\n{model}")
    
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数量: {total_params:,}")

    # 优化器和损失函数
    optimizer = Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.MSELoss()
    
    # 学习率调度器 - 移除verbose参数
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 记录初始学习率
    current_lr = optimizer.param_groups[0]['lr']
    logger.info(f"初始学习率: {current_lr}")

    # 训练循环
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = params['early_stopping_patience']
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse': [],
        'val_r2': []
    }

    for epoch in range(params['epochs']):
        # 训练模式
        model.train()
        train_losses = []

        # 使用tqdm显示进度条
        progress_bar = tqdm(train_gen, desc=f"Epoch {epoch + 1}/{params['epochs']}")
        
        for X_batch, y_batch, padding_mask in progress_bar:
            # 将数据移动到设备
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            padding_mask = padding_mask.to(device)

            # 前向传播
            y_pred = model(X_batch, src_key_padding_mask=padding_mask).squeeze()
            loss = criterion(y_pred, y_batch)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), params['grad_clip'])
            
            optimizer.step()

            # 记录损失
            train_losses.append(loss.item())
            progress_bar.set_postfix({'loss': f"{loss.item():.6f}"})

        # 计算平均训练损失
        avg_train_loss = sum(train_losses) / len(train_losses)
        history['train_loss'].append(avg_train_loss)

        # 验证模式
        model.eval()
        val_losses = []
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for X_val_batch, y_val_batch, val_padding_mask in val_gen:
                # 将数据移动到设备
                X_val_batch = X_val_batch.to(device)
                y_val_batch = y_val_batch.to(device)
                val_padding_mask = val_padding_mask.to(device)

                # 前向传播
                val_pred = model(X_val_batch, src_key_padding_mask=val_padding_mask).squeeze()
                val_loss = criterion(val_pred, y_val_batch)
                
                # 记录损失和预测结果
                val_losses.append(val_loss.item())
                
                # 确保val_pred是向量而不是标量
                if val_pred.ndim == 0:  # 如果是标量(0维张量)
                    all_val_preds.append(val_pred.item())
                else:
                    all_val_preds.extend(val_pred.cpu().numpy().tolist())
                    
                # 确保y_val_batch是向量而不是标量
                if y_val_batch.ndim == 0:  # 如果是标量
                    all_val_targets.append(y_val_batch.item())
                else:
                    all_val_targets.extend(y_val_batch.cpu().numpy().tolist())

        # 计算验证指标
        avg_val_loss = sum(val_losses) / len(val_losses)
        val_mse = mean_squared_error(all_val_targets, all_val_preds)
        val_r2 = r2_score(all_val_targets, all_val_preds)
        
        # 记录验证历史
        history['val_loss'].append(avg_val_loss)
        history['val_mse'].append(val_mse)
        history['val_r2'].append(val_r2)
        
        # 学习率调度
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # 检查学习率是否变化
        if new_lr != old_lr:
            logger.info(f"学习率从 {old_lr:.6f} 调整为 {new_lr:.6f}")

        logger.info(f"Epoch {epoch+1}/{params['epochs']}: "
                    f"训练损失={avg_train_loss:.6f}, 验证损失={avg_val_loss:.6f}, "
                    f"验证MSE={val_mse:.6f}, 验证R²={val_r2:.6f}")

        # 检查是否需要保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            
            # 保存当前最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_mse': val_mse,
                'val_r2': val_r2,
                'params': params
            }, os.path.join(model_dir, 'best_model.pt'))
            
            logger.info(f"第 {epoch+1} 轮更新了最佳模型, 验证损失: {best_val_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= early_stopping_patience:
            logger.info(f"早停触发！{early_stopping_patience}个epoch没有改善")
            break
    
    # 加载最佳模型状态
    model.load_state_dict(best_model_state)
    
    # 保存训练历史
    with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # 可视化训练历史
    plot_training_history(history, os.path.join(model_dir, 'training_curves.png'))

    # 在全部验证集上评估最佳模型
    model.eval()
    all_val_preds = []
    all_val_targets = []
    
    with torch.no_grad():
        for X_val_batch, y_val_batch, val_padding_mask in val_gen:
            X_val_batch = X_val_batch.to(device)
            val_padding_mask = val_padding_mask.to(device)
            
            val_pred = model(X_val_batch, src_key_padding_mask=val_padding_mask).squeeze()
            
            # 确保val_pred是向量而不是标量
            if val_pred.ndim == 0:  # 如果是标量(0维张量)
                all_val_preds.append(val_pred.cpu().item())
                all_val_targets.append(y_val_batch.item())
            else:
                all_val_preds.extend(val_pred.cpu().numpy().tolist())
                all_val_targets.extend(y_val_batch.numpy().tolist())
    
    # 计算验证集性能指标
    val_mse = mean_squared_error(all_val_targets, all_val_preds)
    val_r2 = r2_score(all_val_targets, all_val_preds)
    
    logger.info(f"最佳模型在验证集上的性能: MSE={val_mse:.6f}, R²={val_r2:.6f}")
    
    # 可视化预测结果
    plot_prediction_vs_actual(
        all_val_targets, all_val_preds, 
        os.path.join(model_dir, 'validation_predictions.png'),
        title="验证集上的预测结果"
    )

    return model, val_mse, val_r2, history


# 绘制训练曲线
def plot_training_history(history, save_path):
    plt.figure(figsize=(12, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epochs')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 绘制评估指标曲线
    plt.subplot(2, 1, 2)
    plt.plot(history['val_mse'], label='验证MSE')
    ax1 = plt.gca()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MSE')
    
    ax2 = ax1.twinx()
    ax2.plot(history['val_r2'], 'r-', label='验证R²')
    ax2.set_ylabel('R²')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()


# 绘制预测对比实际值
def plot_prediction_vs_actual(y_true, y_pred, save_path, title="预测对比实际值"):
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # 添加对角线 (理想预测线)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # 计算和显示指标
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    plt.title(f"{title}\nMSE: {mse:.6f}, R²: {r2:.6f}")
    
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.grid(True)
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()


# 评估模型
def evaluate_model(model, test_data, params, save_dir):
    """在测试集上评估模型性能"""
    X_test, y_test = test_data['embeddings'], test_data['targets']
    max_seq_len = params['max_seq_len']
    
    # 创建测试数据生成器
    test_gen = ProteinDataGenerator(
        X_test, y_test, max_seq_len, batch_size=params['batch_size'], shuffle=False
    )
    
    model.eval()
    all_test_preds = []
    all_test_targets = []
    
    with torch.no_grad():
        for X_test_batch, y_test_batch, test_padding_mask in test_gen:
            X_test_batch = X_test_batch.to(device)
            test_padding_mask = test_padding_mask.to(device)
            
            test_pred = model(X_test_batch, src_key_padding_mask=test_padding_mask).squeeze()
            
            # 修复：处理标量张量
            if test_pred.ndim == 0:  # 如果是标量张量
                all_test_preds.append(test_pred.cpu().item())
                all_test_targets.append(y_test_batch.item())
            else:
                all_test_preds.extend(test_pred.cpu().numpy().tolist())
                all_test_targets.extend(y_test_batch.numpy().tolist())
    
    # 计算测试集性能指标
    test_mse = mean_squared_error(all_test_targets, all_test_preds)
    test_r2 = r2_score(all_test_targets, all_test_preds)
    
    logger.info(f"测试集性能: MSE={test_mse:.6f}, R²={test_r2:.6f}")
    
    # 可视化测试集预测
    plot_prediction_vs_actual(
        all_test_targets, all_test_preds, 
        os.path.join(save_dir, 'test_predictions.png'),
        title="测试集上的预测结果"
    )
    
    # 保存详细结果
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump({
            'test_mse': test_mse,
            'test_r2': test_r2
        }, f, indent=4)
    
    return test_mse, test_r2


# 添加交叉验证功能
def cross_validation(data, params, output_dir, n_splits=5, seed=42):
    """执行k折交叉验证"""
    logger.info(f"开始{n_splits}折交叉验证")
    
    # 准备数据
    embeddings = data['embeddings']
    targets = data['targets']
    embedding_dim = data['embedding_dim']
    max_seq_len = data['max_seq_len']
    
    # 创建KFold对象
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # 保存每折的结果
    cv_results = {
        'fold_train_loss': [],
        'fold_val_loss': [],
        'fold_val_mse': [],
        'fold_val_r2': [],
        'fold_models': []
    }
    
    # 创建存储每折结果的目录
    folds_dir = os.path.join(output_dir, 'cv_folds')
    os.makedirs(folds_dir, exist_ok=True)
    
    # 遍历每一折
    for fold, (train_idx, val_idx) in enumerate(kfold.split(embeddings)):
        logger.info(f"开始训练折 {fold+1}/{n_splits}")
        
        # 创建当前折的目录
        fold_dir = os.path.join(folds_dir, f'fold_{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # 准备训练和验证数据
        train_data = {
            'embeddings': [embeddings[i] for i in train_idx],
            'targets': targets[train_idx],
            'embedding_dim': embedding_dim
        }

        val_data = {
            'embeddings': [embeddings[i] for i in val_idx],
            'targets': targets[val_idx],
            'embedding_dim': embedding_dim
        }
        
        logger.info(f"折 {fold+1}: 训练样本={len(train_data['embeddings'])}, 验证样本={len(val_data['embeddings'])}")
        
        # 训练模型
        model, val_mse, val_r2, history = train_single_model(
            train_data, val_data, params, fold_dir
        )
        
        # 记录该折的结果
        cv_results['fold_train_loss'].append(history['train_loss'][-1])
        cv_results['fold_val_loss'].append(history['val_loss'][-1])
        cv_results['fold_val_mse'].append(val_mse)
        cv_results['fold_val_r2'].append(val_r2)
        cv_results['fold_models'].append(fold_dir)
    
    # 计算交叉验证平均指标
    avg_val_mse = np.mean(cv_results['fold_val_mse'])
    avg_val_r2 = np.mean(cv_results['fold_val_r2'])
    std_val_mse = np.std(cv_results['fold_val_mse'])
    std_val_r2 = np.std(cv_results['fold_val_r2'])
    
    # 记录交叉验证结果
    cv_summary = {
        'avg_val_mse': avg_val_mse,
        'avg_val_r2': avg_val_r2,
        'std_val_mse': std_val_mse,
        'std_val_r2': std_val_r2,
        'fold_results': {
            'val_mse': cv_results['fold_val_mse'],
            'val_r2': cv_results['fold_val_r2']
        }
    }
    
    # 保存交叉验证结果
    with open(os.path.join(output_dir, 'cv_results.json'), 'w') as f:
        json.dump(cv_summary, f, indent=4)
    
    # 将结果写入摘要文件
    summary_file = os.path.join(output_dir, "cv_summary.txt")
    with open(summary_file, 'w', encoding="utf-8") as f:
        f.write(f"============= {n_splits}折交叉验证结果 =============\n\n")
        f.write(f"平均验证MSE: {avg_val_mse:.6f} ± {std_val_mse:.6f}\n")
        f.write(f"平均验证R²: {avg_val_r2:.6f} ± {std_val_r2:.6f}\n\n")
        
        f.write("每折结果:\n")
        for i in range(n_splits):
            f.write(f"  折 {i+1}: MSE={cv_results['fold_val_mse'][i]:.6f}, R²={cv_results['fold_val_r2'][i]:.6f}\n")
    
    logger.info(f"交叉验证完成: 平均MSE={avg_val_mse:.6f}±{std_val_mse:.6f}, 平均R²={avg_val_r2:.6f}±{std_val_r2:.6f}")
    
    return cv_summary


# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练蛋白质序列Transformer模型')
    parser.add_argument('--data_path', type=str, 
                        default='preprocess/embedding/output/sampled_output_embeddings.npz',
                        help='NPZ数据文件路径')
    parser.add_argument('--max_seq_len', type=int, default=None,
                        help='最大序列长度，留空则使用数据集中的最大长度')
    parser.add_argument('--output_dir', type=str, default='./transformer_results',
                        help='结果保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--use_cv', action='store_true', help='是否使用交叉验证')
    parser.add_argument('--n_splits', type=int, default=5, help='交叉验证折数')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存命令行参数
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # 添加文件日志处理器
    file_handler = logging.FileHandler(os.path.join(output_dir, 'training.log'), encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"开始任务，输出将保存到: {output_dir}")
    
    try:
        # 加载数据
        data_info = load_and_preprocess_data(args.data_path, max_seq_len=args.max_seq_len)
        
        # 获取用户输入的参数配置
        try:
            print("\n请输入训练参数 (直接回车则使用默认值):")
            
            default_params = {
                'd_model': 128,            # Transformer模型维度
                'nhead': 4,                # 多头注意力头数
                'num_layers': 2,           # Transformer编码器层数
                'dim_feedforward': 512,    # 前馈网络维度
                'dropout': 0.1,            # Dropout比例
                'lr': 0.0005,              # 学习率
                'weight_decay': 0.01,      # 权重衰减
                'batch_size': 32,          # 批次大小
                'epochs': 100,             # 训练轮数
                'grad_clip': 1.0,          # 梯度裁剪阈值
                'early_stopping_patience': 10,  # 早停耐心值
                'max_seq_len': data_info['max_seq_len']  # 最大序列长度
            }
            
            user_d_model = input(f"模型维度 (默认: {default_params['d_model']}): ")
            user_nhead = input(f"注意力头数 (默认: {default_params['nhead']}): ")
            user_num_layers = input(f"Transformer层数 (默认: {default_params['num_layers']}): ")
            user_dim_feedforward = input(f"前馈网络维度 (默认: {default_params['dim_feedforward']}): ")
            user_dropout = input(f"Dropout比例 (默认: {default_params['dropout']}): ")
            user_lr = input(f"学习率 (默认: {default_params['lr']}): ")
            user_batch_size = input(f"批次大小 (默认: {default_params['batch_size']}): ")
            user_epochs = input(f"训练轮数 (默认: {default_params['epochs']}): ")
            
            params = {
                'd_model': int(user_d_model) if user_d_model else default_params['d_model'],
                'nhead': int(user_nhead) if user_nhead else default_params['nhead'],
                'num_layers': int(user_num_layers) if user_num_layers else default_params['num_layers'],
                'dim_feedforward': int(user_dim_feedforward) if user_dim_feedforward else default_params['dim_feedforward'],
                'dropout': float(user_dropout) if user_dropout else default_params['dropout'],
                'lr': float(user_lr) if user_lr else default_params['lr'],
                'weight_decay': default_params['weight_decay'],
                'batch_size': int(user_batch_size) if user_batch_size else default_params['batch_size'],
                'epochs': int(user_epochs) if user_epochs else default_params['epochs'],
                'grad_clip': default_params['grad_clip'],
                'early_stopping_patience': default_params['early_stopping_patience'],
                'max_seq_len': data_info['max_seq_len']
            }
            
            # 询问是否使用交叉验证
            if not args.use_cv:
                user_cv = input(f"是否使用交叉验证? (y/n, 默认: n): ")
                args.use_cv = user_cv.lower() == 'y'
            
            if args.use_cv:
                user_splits = input(f"交叉验证折数 (默认: {args.n_splits}): ")
                if user_splits:
                    args.n_splits = int(user_splits)
            
            # 验证nhead是否能整除d_model
            if params['d_model'] % params['nhead'] != 0:
                logger.warning(f"模型维度 ({params['d_model']}) 必须能被注意力头数 ({params['nhead']}) 整除")
                params['nhead'] = min(params['nhead'], params['d_model']) 
                while params['d_model'] % params['nhead'] != 0:
                    params['nhead'] -= 1
                logger.info(f"已调整注意力头数为: {params['nhead']}")
            
            logger.info(f"使用的训练参数: {params}")
            
            # 保存参数到文件
            with open(os.path.join(output_dir, 'params.json'), 'w') as f:
                json.dump(params, f, indent=4)
                
        except ValueError as e:
            logger.error(f"参数输入格式错误: {e}")
            logger.info(f"使用默认参数: {default_params}")
            params = default_params
            
            # 保存参数到文件
            with open(os.path.join(output_dir, 'params.json'), 'w') as f:
                json.dump(params, f, indent=4)
        
        # 使用交叉验证
        if args.use_cv:
            logger.info(f"使用{args.n_splits}折交叉验证")
            cv_results = cross_validation(
                data_info, params, output_dir, n_splits=args.n_splits, seed=args.seed
            )
            
            logger.info("\n训练和交叉验证完成!")
        else:
            # 常规训练流程
            # 划分训练集和测试集
            train_idx, test_idx = train_test_split(
                range(len(data_info['embeddings'])), 
                test_size=0.15, 
                random_state=args.seed
            )
            
            # 创建训练和测试数据集
            train_data = {
                'embeddings': [data_info['embeddings'][i] for i in train_idx],
                'targets': data_info['targets'][train_idx],
                'embedding_dim': data_info['embedding_dim']
            }
            
            test_data = {
                'embeddings': [data_info['embeddings'][i] for i in test_idx],
                'targets': data_info['targets'][test_idx],
                'embedding_dim': data_info['embedding_dim']
            }
            
            logger.info(f"数据分割完成: 训练集={len(train_data['embeddings'])}, 测试集={len(test_data['embeddings'])}")
            
            # 划分训练集和验证集
            train_indices, val_indices = train_test_split(
                range(len(train_data['embeddings'])), 
                test_size=0.15, 
                random_state=args.seed
            )
            
            train_dataset = {
                'embeddings': [train_data['embeddings'][i] for i in train_indices],
                'targets': train_data['targets'][train_indices],
                'embedding_dim': data_info['embedding_dim']
            }
            
            val_dataset = {
                'embeddings': [train_data['embeddings'][i] for i in val_indices],
                'targets': train_data['targets'][val_indices],
                'embedding_dim': data_info['embedding_dim']
            }
            
            logger.info(f"验证集划分: 训练={len(train_dataset['embeddings'])}, 验证={len(val_dataset['embeddings'])}")
            
            # 训练模型
            logger.info("\n开始训练...")
            model, val_mse, val_r2, history = train_single_model(
                train_dataset, val_dataset, params, output_dir
            )
            
            # 评估最佳模型
            logger.info("\n在测试集上评估最佳模型...")
            test_mse, test_r2 = evaluate_model(model, test_data, params, output_dir)
            
            # 保存结果摘要
            summary_file = os.path.join(output_dir, "results_summary.txt")
            with open(summary_file, 'w', encoding="utf-8") as f:
                f.write("============== Transformer模型训练结果摘要 ==============\n\n")
                f.write(f"参数配置:\n")
                for param_name, param_value in params.items():
                    f.write(f"  - {param_name}: {param_value}\n")
                f.write("\n")
                
                f.write(f"验证集性能: MSE={val_mse:.6f}, R²={val_r2:.6f}\n")
                f.write(f"测试集性能: MSE={test_mse:.6f}, R²={test_r2:.6f}\n\n")

                f.write("训练过程:\n")
                f.write(f"  - 最终训练损失: {history['train_loss'][-1]:.6f}\n")
                f.write(f"  - 最终验证损失: {history['val_loss'][-1]:.6f}\n")
                f.write(f"  - 训练轮数: {len(history['train_loss'])}/{params['epochs']}\n")
            
            logger.info(f"结果摘要已保存至: {summary_file}")
            logger.info("\n训练和评估完成!")
    
    except Exception as e:
        logger.exception(f"训练过程中出错: {str(e)}")
    

if __name__ == "__main__":
    main()

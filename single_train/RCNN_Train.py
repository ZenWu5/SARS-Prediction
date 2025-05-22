# RCNN_Training.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import os
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import gc
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

# 配置plt字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger("rcnn_training")

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


# 创建一个纯PyTorch版本的RCNN模型
class PyTorchRCNN(nn.Module):
    def __init__(self, input_dim, hidden_size=64, dropout=0.2):
        super(PyTorchRCNN, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # GRU层
        gru_out, _ = self.gru(x)
        # 转换维度以适应Conv1d (从[batch, seq_len, features]到[batch, features, seq_len])
        conv_in = gru_out.permute(0, 2, 1)
        # 卷积层
        conv_out = self.conv(conv_in)
        # 转换回原始维度
        conv_out = conv_out.permute(0, 2, 1)
        # 池化层 - 沿序列维度进行最大池化
        pooled, _ = torch.max(conv_out, dim=1)
        # Dropout和全连接层
        out = self.dropout(pooled)
        out = self.fc(out)
        return out


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
        
        # 获取嵌入向量的维度
        embedding_dim = embeddings_list[0].shape[1]
        logger.info(f"嵌入向量维度: {embedding_dim}")
        
        # 对序列进行填充或截断处理
        processed_embeddings = []
        for embedding in embeddings_list:
            # 如果序列长度超过最大长度，则截断
            if embedding.shape[0] > max_seq_len:
                processed_embedding = embedding[:max_seq_len, :]
            # 如果序列长度小于最大长度，则填充
            elif embedding.shape[0] < max_seq_len:
                padding = np.zeros((max_seq_len - embedding.shape[0], embedding_dim), dtype=np.float32)
                processed_embedding = np.vstack([embedding, padding])
            else:
                processed_embedding = embedding
            
            processed_embeddings.append(processed_embedding)
        
        # 转换为numpy数组
        features = np.array(processed_embeddings, dtype=np.float32)
        targets = targets.astype(np.float32)
        
        logger.info(f"处理后的特征形状: {features.shape}")
        logger.info(f"标签形状: {targets.shape}")
        
        return features, targets
    
    except Exception as e:
        logger.error(f"加载或预处理数据时出错: {str(e)}")
        return None, None


# 单批数据处理函数，处理可能的内存限制
def process_in_batches(model, data, batch_size=32):
    predictions = []
    n_samples = len(data)

    for i in range(0, n_samples, batch_size):
        batch = data[i:i + batch_size]
        batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
        with torch.no_grad():
            batch_pred = model(batch_tensor).cpu().numpy()
        predictions.append(batch_pred)

    return np.vstack(predictions).squeeze()


# 训练单个模型
# def train_single_model(X_train, y_train, X_val, y_val, params):
#     """训练单个RCNN模型并返回验证集性能"""
#     # 清理内存
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     # 准备数据加载器
#     train_dataset = TensorDataset(
#         torch.tensor(X_train, dtype=torch.float32),
#         torch.tensor(y_train, dtype=torch.float32)
#     )
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=params['batch_size'],
#         shuffle=True
#     )

#     # 初始化模型
#     model = PyTorchRCNN(
#         input_dim=X_train.shape[2],  # 最后一个维度是特征维度
#         hidden_size=params['hidden_size'],
#         dropout=params['dropout']
#     ).to(device)

#     # 优化器和损失函数
#     optimizer = Adam(model.parameters(), lr=params['lr'])
#     criterion = nn.MSELoss()

#     # 训练循环
#     best_val_loss = float('inf')
#     best_model_state = None
#     patience_counter = 0
#     patience = 5  # 早停耐心值

#     for epoch in range(params['epochs']):
#         # 训练模式
#         model.train()
#         train_losses = []

#         # 使用tqdm显示进度条
#         for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{params['epochs']}"):
#             # 将数据移动到设备
#             X_batch = X_batch.to(device)
#             y_batch = y_batch.to(device)

#             # 前向传播
#             y_pred = model(X_batch).squeeze()
#             loss = criterion(y_pred, y_batch)

#             # 反向传播和优化
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # 记录损失
#             train_losses.append(loss.item())

#         # 计算平均训练损失
#         avg_train_loss = sum(train_losses) / len(train_losses)

#         # 验证模式
#         model.eval()

#         # 分批处理验证集，避免内存问题
#         X_val_batches = [X_val[i:i + params['batch_size']] for i in range(0, len(X_val), params['batch_size'])]
#         y_val_batches = [y_val[i:i + params['batch_size']] for i in range(0, len(y_val), params['batch_size'])]

#         val_losses = []
#         for X_val_batch, y_val_batch in zip(X_val_batches, y_val_batches):
#             X_val_tensor = torch.tensor(X_val_batch, dtype=torch.float32).to(device)
#             y_val_tensor = torch.tensor(y_val_batch, dtype=torch.float32).to(device)

#             with torch.no_grad():
#                 val_pred = model(X_val_tensor).squeeze()
#                 val_loss = criterion(val_pred, y_val_tensor)
#                 val_losses.append(val_loss.item())

#         avg_val_loss = sum(val_losses) / len(val_losses)

#         logger.info(f"Epoch {epoch + 1}: 训练损失={avg_train_loss:.6f}, 验证损失={avg_val_loss:.6f}")

#         # 检查是否需要保存模型
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_model_state = model.state_dict().copy()
#             patience_counter = 0
#         else:
#             patience_counter += 1

#         # 早停
#         if patience_counter >= patience:
#             logger.info(f"早停触发！{patience}个epoch没有改善")
#             break

#     # 加载最佳模型状态
#     model.load_state_dict(best_model_state)

#     # 在验证集上评估最佳模型
#     model.eval()

#     # 对验证集进行批量预测
#     val_pred = process_in_batches(model, X_val, params['batch_size'])

#     # 计算验证集性能指标
#     val_mse = mean_squared_error(y_val, val_pred)
#     val_r2 = r2_score(y_val, val_pred)

#     return model, val_mse, val_r2

def train_single_model(X_train, y_train, X_val, y_val, params):
    """训练单个RCNN模型并返回验证集性能和训练历史"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True
    )

    model = PyTorchRCNN(
        input_dim=X_train.shape[2],
        hidden_size=params['hidden_size'],
        dropout=params['dropout']
    ).to(device)

    optimizer = Adam(model.parameters(), lr=params['lr'])
    criterion = nn.MSELoss()

    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.95 ** (epoch - warmup_epochs)
    scheduler = LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 5

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse': [],
        'val_r2': [],
        'val_pearson': []
    }

    for epoch in range(params['epochs']):
        model.train()
        train_losses = []

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{params['epochs']}"):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        history['train_loss'].append(avg_train_loss)

        # 验证
        model.eval()
        val_preds = []
        val_targets = []
        X_val_batches = [X_val[i:i + params['batch_size']] for i in range(0, len(X_val), params['batch_size'])]
        y_val_batches = [y_val[i:i + params['batch_size']] for i in range(0, len(y_val), params['batch_size'])]
        val_losses = []
        for X_val_batch, y_val_batch in zip(X_val_batches, y_val_batches):
            X_val_tensor = torch.tensor(X_val_batch, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val_batch, dtype=torch.float32).to(device)
            with torch.no_grad():
                val_pred = model(X_val_tensor).squeeze()
                val_loss = criterion(val_pred, y_val_tensor)
                val_losses.append(val_loss.item())
                val_preds.append(val_pred.cpu().numpy())
                val_targets.append(y_val_tensor.cpu().numpy())
        avg_val_loss = sum(val_losses) / len(val_losses)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_mse = mean_squared_error(val_targets, val_preds)
        val_r2 = r2_score(val_targets, val_preds)
        val_pearson, _ = pearsonr(val_targets, val_preds)

        history['val_loss'].append(avg_val_loss)
        history['val_mse'].append(val_mse)
        history['val_r2'].append(val_r2)
        history['val_pearson'].append(val_pearson)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch + 1}: 当前学习率 = {current_lr:.6f}")
        logger.info(f"Epoch {epoch + 1}: 训练损失={avg_train_loss:.6f}, 验证损失={avg_val_loss:.6f}, "
                    f"MSE={val_mse:.6f}, R²={val_r2:.6f}, Pearson={val_pearson:.6f}")

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

    model.load_state_dict(best_model_state)
    # 验证集最终评估
    val_pred = process_in_batches(model, X_val, params['batch_size'])
    val_mse = mean_squared_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    val_pearson, _ = pearsonr(y_val, val_pred)

    validation_results = {
        'loss': best_val_loss,
        'mse': val_mse,
        'r2': val_r2,
        'pearson': val_pearson
    }

    return model, validation_results, history

# K折交叉验证训练
def train_with_kfold(features, targets, params, k=5):
    """使用K折交叉验证训练RCNN模型"""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # 存储每个折叠的性能
    fold_results = []
    best_model = None
    best_mse = float('inf')

    for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
        logger.info(f"\n{'=' * 50}\n开始训练第 {fold + 1}/{k} 折\n{'=' * 50}")

        # 准备当前折叠的数据
        X_train, y_train = features[train_idx], targets[train_idx]
        X_val, y_val = features[val_idx], targets[val_idx]

        logger.info(f"训练集数据形状: {X_train.shape}, 验证集数据形状: {X_val.shape}")

        # 训练模型
        start_time = time.time()
        model, validation_results, history = train_single_model(X_train, y_train, X_val, y_val, params)
        training_time = time.time() - start_time

        fold_results.append({
            'fold': fold + 1,
            'val_mse': validation_results['mse'],
            'val_r2': validation_results['r2'],
            'val_pearson': validation_results['pearson'],
            'time': training_time
        })

        logger.info(f"第 {fold + 1} 折结果: MSE={validation_results['mse']:.6f}, R²={validation_results['r2']:.6f}, "
                    f"Pearson={validation_results['pearson']:.6f}, 训练时间: {training_time:.2f}秒")

        # 判断是否是最佳模型
        if validation_results['mse'] < best_mse:
            best_mse = validation_results['mse']
            best_model = model

        # 在每个折叠后释放内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 计算平均性能
    avg_mse = sum(fold['val_mse'] for fold in fold_results) / k
    avg_r2 = sum(fold['val_r2'] for fold in fold_results) / k
    avg_time = sum(fold['time'] for fold in fold_results) / k

    logger.info(f"\n{'=' * 50}")
    logger.info(f"交叉验证平均结果: MSE={avg_mse:.6f}, R²={avg_r2:.6f}, 平均训练时间: {avg_time:.2f}秒")

    return best_model, fold_results


# 测试最终模型性能
def evaluate_model(model, X_test, y_test, batch_size=32):
    """在测试集上评估模型性能"""
    model.eval()

    # 批量预测
    test_pred = process_in_batches(model, X_test, batch_size)

    # 计算测试集性能指标
    test_mse = mean_squared_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    logger.info(f"测试集性能: MSE={test_mse:.6f}, R²={test_r2:.6f}")
    # 设置中文字体
    try:
        font = FontProperties(fname=r'C:\Windows\Fonts\simkai.ttf', size=10)
    except:
        # 如果指定字体不存在，使用系统默认字体
        font = FontProperties(size=10)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, test_pred, alpha=0.5)

    # 添加对角线 (理想预测线)
    min_val = min(min(y_test), min(test_pred))
    max_val = max(max(y_test), max(test_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.xlabel('实际值', fontsize=12, fontproperties=font)
    plt.ylabel('预测值', fontsize=12, fontproperties=font)
    plt.title('RCNN模型预测结果', fontsize=14, fontproperties=font)
    plt.grid(True)

    # 保存图像
    figtime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    plt.savefig(f'rcnn_predictions_{figtime}.png')
    logger.info(f"预测结果可视化已保存至 'rcnn_predictions_{figtime}.png'")

    return test_mse, test_r2


# 主函数
def main():
    # 设置随机种子
    set_seed()

    # 创建结果目录
    results_dir = 'ModelResults/RCNNResults'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 加载NPZ格式的蛋白质嵌入数据
    data_path = input("请输入NPZ数据文件路径 (按回车使用默认路径 'preprocess/embedding/output/sampled_output_embeddings.npz'): ")
    if not data_path:
        data_path = 'preprocess/embedding/output/sampled_output_embeddings.npz'
    
    # 获取最大序列长度或使用默认值
    max_seq_len_input = input("请输入最大序列长度 (按回车使用数据集中的最大长度): ")
    max_seq_len = int(max_seq_len_input) if max_seq_len_input else None
    
    # 加载并预处理数据
    features, targets = load_and_preprocess_data(data_path, max_seq_len)
    if features is None or targets is None:
        return

    # 数据分割
    indices = np.random.permutation(len(features))
    test_size = int(0.15 * len(features))

    test_indices = indices[:test_size]
    train_val_indices = indices[test_size:]

    X_test, y_test = features[test_indices], targets[test_indices]
    X_train_val, y_train_val = features[train_val_indices], targets[train_val_indices]

    # 训练集fit
    scaler_X = StandardScaler()
    X_train_val = scaler_X.fit_transform(X_train_val.reshape(-1, X_train_val.shape[-1])).reshape(X_train_val.shape)
    X_test = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    # 若对y也标准化
    scaler_y = StandardScaler()
    y_train_val = scaler_y.fit_transform(y_train_val.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    logger.info(f"数据分割完成: 训练+验证集={X_train_val.shape}, 测试集={X_test.shape}")

    # 获取用户输入的参数配置
    try:
        print("\n请输入训练参数 (直接回车则使用默认值):")

        default_params = {
            'hidden_size': 128,
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 128,
            'epochs': 100
        }

        user_hidden_size = input(f"隐藏层大小 (默认: {default_params['hidden_size']}): ")
        user_dropout = input(f"Dropout比例 (默认: {default_params['dropout']}): ")
        user_lr = input(f"学习率 (默认: {default_params['lr']}): ")
        user_batch_size = input(f"批次大小 (默认: {default_params['batch_size']}): ")
        user_epochs = input(f"训练轮数 (默认: {default_params['epochs']}): ")

        params = {
            'hidden_size': int(user_hidden_size) if user_hidden_size else default_params['hidden_size'],
            'dropout': float(user_dropout) if user_dropout else default_params['dropout'],
            'lr': float(user_lr) if user_lr else default_params['lr'],
            'batch_size': int(user_batch_size) if user_batch_size else default_params['batch_size'],
            'epochs': int(user_epochs) if user_epochs else default_params['epochs']
        }

        logger.info(f"使用的训练参数: {params}")
    except ValueError as e:
        logger.error(f"参数输入格式错误: {e}")
        logger.info(f"使用默认参数: {default_params}")
        params = default_params

    # 训练模型
    logger.info("\n开始交叉验证训练...")
    best_model, fold_results = train_with_kfold(X_train_val, y_train_val, params, k=5)

    # 保存最佳模型
    model_path = os.path.join(results_dir, 'best_rcnn_model.pt')
    torch.save(best_model.state_dict(), model_path)
    logger.info(f"最佳模型已保存至: {model_path}")

    # 评估最佳模型
    logger.info("\n在测试集上评估最佳模型...")
    test_mse, test_r2 = evaluate_model(best_model, X_test, y_test, batch_size=params['batch_size'])

    # 保存结果摘要
    logtime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    summary_file = os.path.join(results_dir, f"rcnn_results_{logtime}.txt")
    with open(summary_file, 'w', encoding = "UTF-8") as f:
        f.write("============== RCNN模型训练结果摘要 ==============\n\n")
        f.write(f"参数配置: {params}\n\n")

        f.write("交叉验证结果:\n")
        for fold in fold_results:
            f.write(f"  - 第 {fold['fold']} 折: MSE={fold['val_mse']:.6f}, R²={fold['val_r2']:.6f}\n")

        avg_mse = sum(fold['val_mse'] for fold in fold_results) / len(fold_results)
        avg_r2 = sum(fold['val_r2'] for fold in fold_results) / len(fold_results)
        f.write(f"\n交叉验证平均性能: MSE={avg_mse:.6f}, R²={avg_r2:.6f}\n\n")

        f.write(f"测试集性能: MSE={test_mse:.6f}, R²={test_r2:.6f}\n")

    logger.info(f"结果摘要已保存至: {summary_file}")
    logger.info("\n训练和评估完成!")


if __name__ == "__main__":
    main()

import numpy as np
import h5py
import os
import time
import logging
import argparse
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from matplotlib.font_manager import FontProperties

# 配置命令行参数
parser = argparse.ArgumentParser(description='训练和评估多个机器学习基线模型')
parser.add_argument('--data_path', type=str, default='preprocess/ESM/output/sampled_output_esm_embeddings.h5',
                    help='h5数据文件路径')
parser.add_argument('--output_dir', type=str, default='ModelResults/BaselineModels',
                    help='结果输出目录')
parser.add_argument('--k_folds', type=int, default=5,
                    help='交叉验证折数')
parser.add_argument('--seed', type=int, default=42,
                    help='随机种子')
parser.add_argument('--test_size', type=float, default=0.15,
                    help='测试集比例')
parser.add_argument('--n_jobs', type=int, default=-1,
                    help='并行任务数，-1表示使用所有CPU')
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
logger = logging.getLogger("baseline_training")

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)

class H5FeatureProcessor:
    """处理H5数据文件并提取平均特征向量"""
    def __init__(self, h5_path):
        self.h5_path = h5_path if h5_path.endswith('.h5') else h5_path + '.h5'
        with h5py.File(self.h5_path, 'r') as f:
            self.total = int(f.attrs['total_samples'])
            self.emb_dim = int(f.attrs['embedding_dim'])
            # 加载目标值
            self.targets = f['mean_log10Ka'][:].astype(np.float32)  # (N,)
            # 计算序列长度
            self.seq_lengths = np.array([f['embeddings'][f'emb_{i}'].shape[0] for i in range(self.total)], dtype=np.int32)

    def extract_features(self, normalize=True, max_samples=None):
        n_samples = min(self.total, max_samples) if max_samples else self.total
        X = np.zeros((n_samples, self.emb_dim), dtype=np.float32)

        with h5py.File(self.h5_path, 'r') as f:
            # 调试：查看 emb_0 的前几个值
            sample = f['embeddings']['emb_0'][:5, :5]
            logger.info(f"[DEBUG] emb_0 dtype={sample.dtype}, 前5×5样本=\n{sample}")

            for i in tqdm(range(n_samples), desc="提取特征"):
                emb = f['embeddings'][f'emb_{i}'][:].astype(np.float32)
                X[i] = np.mean(emb, axis=0)

        # 调试：查看 X 原始统计信息
        logger.info(f"[DEBUG] X 原始 stats: mean={X.mean():.4e}, std={X.std():.4e}, "
                    f"min={X.min():.4e}, max={X.max():.4e}")

        y = self.targets[:n_samples]

        if normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            # 调试：查看 scaler.mean_ 和 scale_ 的前几个元素
            logger.info(f"[DEBUG] scaler.mean_[:5]={scaler.mean_[:5]}")
            logger.info(f"[DEBUG] scaler.scale_[:5]={scaler.scale_[:5]}")
            logger.info(f"特征已标准化: 均值={scaler.mean_.mean():.4f}, 标准差={scaler.scale_.mean():.4f}")
            return X, y, scaler
        else:
            return X, y, None



def create_baseline_models():
    """创建基线模型字典"""
    models = {
        # 'svm': SVR(kernel='rbf', gamma='scale', C=1.0, epsilon=0.1),
        'rf': RandomForestRegressor(n_estimators=100, max_depth=None, random_state=args.seed),
        'knn': KNeighborsRegressor(n_neighbors=5, weights='distance'),
        'bayes': GaussianNB(),  # 朴素贝叶斯
        'lr': LinearRegression()  # 线性回归
    }
    return models

def train_and_evaluate_models(X_train, y_train, X_test, y_test, output_dir):
    """训练并评估所有基线模型"""
    models = create_baseline_models()
    results = {}
    
    # 创建结果目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置K折交叉验证
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    
    # 遍历训练所有模型
    for name, model in models.items():
        logger.info(f"\n======= 训练 {name} 模型 =======")
        
        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, 
                                   scoring='neg_mean_squared_error', 
                                   n_jobs=args.n_jobs)
        
        # 转换为正MSE并计算平均
        cv_mse = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        logger.info(f"{args.k_folds}折交叉验证: MSE = {cv_mse:.6f} ± {cv_std:.6f}")
        
        # 在全部训练集上重新训练
        model.fit(X_train, y_train)
        
        # 在测试集上评估
        y_pred = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        test_pearson, _ = pearsonr(y_test, y_pred)
        
        logger.info(f"测试集性能: MSE={test_mse:.6f}, MAE={test_mae:.6f}, "
                    f"R²={test_r2:.6f}, Pearson={test_pearson:.6f}")
        
        # 保存结果
        results[name] = {
            'cv_mse': cv_mse,
            'cv_std': cv_std,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_pearson': test_pearson,
            'predictions': y_pred,
            'model': model
        }
        
        # 保存模型
        model_path = os.path.join(output_dir, f'{name}_model.joblib')
        joblib.dump(model, model_path)
        logger.info(f"模型已保存至 {model_path}")
        
        # 绘制预测结果散点图
        plot_predictions(y_test, y_pred, name, output_dir)
    
    return results

def plot_predictions(y_true, y_pred, model_name, output_dir):
    """绘制预测vs真实值散点图"""
    # 设置中文字体
    try:
        font = FontProperties(fname=r'C:\Windows\Fonts\simkai.ttf', size=10)
    except:
        font = FontProperties(size=10)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # 添加理想线
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # 计算性能指标
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson, _ = pearsonr(y_true, y_pred)
    
    plt.xlabel('实际值', fontsize=12, fontproperties=font)
    plt.ylabel('预测值', fontsize=12, fontproperties=font)
    
    model_name_map = {
        'svm': '支持向量回归',
        'rf': '随机森林',
        'knn': 'K近邻',
        'bayes': '朴素贝叶斯',
        'lr': '线性回归'
    }
    
    title = f"{model_name_map.get(model_name, model_name)}模型预测结果"
    plt.title(title, fontsize=14, fontproperties=font)
    plt.grid(True)
    
    # 添加性能指标文本
    text = f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}\nPearson: {pearson:.4f}"
    plt.annotate(text, xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 保存图像
    fig_path = os.path.join(output_dir, f'{model_name}_predictions.png')
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"{model_name}预测结果可视化已保存至 {fig_path}")

def plot_model_comparison(results, output_dir):
    """绘制不同模型性能比较图"""
    # 提取性能指标
    models = list(results.keys())
    test_mse = [results[m]['test_mse'] for m in models]
    test_r2 = [results[m]['test_r2'] for m in models]
    test_pearson = [results[m]['test_pearson'] for m in models]
    
    # 设置中文字体
    try:
        font = FontProperties(fname=r'C:\Windows\Fonts\simkai.ttf', size=10)
    except:
        font = FontProperties(size=10)
        
    # 模型名称映射
    model_name_map = {
        'svm': '支持向量回归',
        'rf': '随机森林',
        'knn': 'K近邻',
        'bayes': '朴素贝叶斯',
        'lr': '线性回归'
    }
    
    zh_models = [model_name_map.get(m, m) for m in models]
    
    # 创建子图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # MSE (越低越好)
    ax1.bar(zh_models, test_mse, color='skyblue')
    ax1.set_title('均方误差 (MSE)', fontproperties=font)
    ax1.set_ylabel('MSE (越低越好)', fontproperties=font)
    for i, v in enumerate(test_mse):
        ax1.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    # R² (越高越好)
    ax2.bar(zh_models, test_r2, color='lightgreen')
    ax2.set_title('决定系数 (R²)', fontproperties=font)
    ax2.set_ylabel('R² (越高越好)', fontproperties=font)
    for i, v in enumerate(test_r2):
        ax2.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    # Pearson (越高越好)
    ax3.bar(zh_models, test_pearson, color='salmon')
    ax3.set_title('皮尔逊相关系数', fontproperties=font)
    ax3.set_ylabel('Pearson (越高越好)', fontproperties=font)
    for i, v in enumerate(test_pearson):
        ax3.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    
    # 保存比较图
    comparison_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(comparison_path)
    plt.close()
    logger.info(f"模型比较结果已保存至 {comparison_path}")

def write_summary(results, output_dir):
    """写入结果摘要"""
    summary_path = os.path.join(output_dir, 'results_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("======= 机器学习基线模型训练结果摘要 =======\n\n")
        
        # 模型名称映射
        model_name_map = {
            'svm': '支持向量回归',
            'rf': '随机森林',
            'knn': 'K近邻',
            'bayes': '朴素贝叶斯',
            'lr': '线性回归'
        }
        
        # 按MSE排序找出最佳模型
        sorted_models = sorted(results.items(), key=lambda x: x[1]['test_mse'])
        best_model, best_results = sorted_models[0]
        
        f.write(f"最佳模型: {model_name_map.get(best_model, best_model)}\n")
        f.write(f"测试集 MSE: {best_results['test_mse']:.6f}\n")
        f.write(f"测试集 R²: {best_results['test_r2']:.6f}\n")
        f.write(f"测试集 Pearson: {best_results['test_pearson']:.6f}\n\n")
        
        f.write("所有模型性能对比:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'模型':<15} {'交叉验证MSE':<20} {'测试MSE':<15} {'测试MAE':<15} {'测试R²':<15} {'测试Pearson':<15}\n")
        f.write("-" * 80 + "\n")
        
        for name, res in sorted_models:
            f.write(f"{model_name_map.get(name, name):<15} ")
            f.write(f"{res['cv_mse']:.6f} ± {res['cv_std']:.6f} ")
            f.write(f"{res['test_mse']:<15.6f} ")
            f.write(f"{res['test_mae']:<15.6f} ")
            f.write(f"{res['test_r2']:<15.6f} ")
            f.write(f"{res['test_pearson']:<15.6f}\n")
    
    logger.info(f"结果摘要已保存至 {summary_path}")

def main():
    # 设置随机种子
    set_seed(args.seed)
    logger.info(f"设置随机种子: {args.seed}")
    
    # 创建时间戳目录
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"结果将保存至: {output_dir}")
    
    # 加载和处理数据
    logger.info(f"从{args.data_path}加载数据...")
    processor = H5FeatureProcessor(args.data_path)
    
    # 提取平均特征向量
    X, y, scaler = processor.extract_features(normalize=True)
    
    # 如果有标准化器，保存它
    if scaler:
        scaler_path = os.path.join(output_dir, 'feature_scaler.joblib')
        joblib.dump(scaler, scaler_path)
        logger.info(f"特征标准化器已保存至 {scaler_path}")
    
    # 划分训练/测试集
    n_samples = len(y)
    indices = np.random.permutation(n_samples)
    test_size = int(args.test_size * n_samples)
    
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    logger.info(f"数据集划分完成: 训练集={len(train_idx)}样本, 测试集={len(test_idx)}样本")
    
    # 训练和评估模型
    results = train_and_evaluate_models(X_train, y_train, X_test, y_test, output_dir)
    
    # 绘制模型比较图
    plot_model_comparison(results, output_dir)
    
    # 写入结果摘要
    write_summary(results, output_dir)
    
    logger.info(f"所有模型训练和评估完成，结果保存在: {output_dir}")
    
    # 找出最佳模型
    best_model = min(results.items(), key=lambda x: x[1]['test_mse'])[0]
    model_name_map = {
        'svm': '支持向量回归',
        'rf': '随机森林',
        'knn': 'K近邻',
        'bayes': '朴素贝叶斯',
        'lr': '线性回归'
    }
    logger.info(f"最佳模型: {model_name_map.get(best_model, best_model)}")

if __name__ == "__main__":
    main()

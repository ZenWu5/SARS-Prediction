from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import numpy as np
from typing import Dict, Any

def calculate_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    计算回归评估指标
    
    参数:
        y_true: 真实标签
        y_pred: 预测值
        
    返回:
        包含各指标的字典
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'pearson_r': pearsonr(y_true, y_pred)[0],
        'spearman_r': spearmanr(y_true, y_pred)[0]
    }
    
    return metrics

def format_metrics(metrics: Dict[str, float]) -> str:
    """将指标格式化为便于打印的字符串"""
    return ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])

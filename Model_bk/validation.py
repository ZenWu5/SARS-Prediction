# -*- coding: utf-8 -*-
"""
validation.py

本模块提供：
- 分类评估函数（混淆矩阵、准确率、精确率、召回率、F1分数、Matthews相关系数）
- 回归评估函数（MSE、MAE、R²、Pearson相关系数）
- 回归结果可视化函数：
  - 实测值 vs 预测值散点图
  - 残差分析图
  - 值分布直方图
  - 绝对误差分布图（含阈值线）
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# 配置plt字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------
# 分类评估函数
# -----------------------------------
def get_confusion_matrix(y_true, y_pred):
    """
    计算混淆矩阵，返回[[TP, FP], [FN, TN]]
    """
    TP = FP = TN = FN = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            TP += 1
        elif yt == 0 and yp == 1:
            FP += 1
        elif yt == 1 and yp == 0:
            FN += 1
        elif yt == 0 and yp == 0:
            TN += 1
    return [[TP, FP], [FN, TN]]


def get_accuracy(conf_matrix):
    """计算准确率 (Accuracy)"""
    TP, FP = conf_matrix[0]
    FN, TN = conf_matrix[1]
    return (TP + TN) / (TP + FP + FN + TN) if TP+FP+FN+TN>0 else 0


def get_precision(conf_matrix):
    """计算精确率 (Precision)"""
    TP, FP = conf_matrix[0]
    return TP / (TP + FP) if (TP + FP) > 0 else 0


def get_recall(conf_matrix):
    """计算召回率 (Recall)"""
    TP = conf_matrix[0][0]
    FN = conf_matrix[1][0]
    return TP / (TP + FN) if (TP + FN) > 0 else 0


def get_f1score(conf_matrix):
    """计算F1分数"""
    p = get_precision(conf_matrix)
    r = get_recall(conf_matrix)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0


def get_mcc(conf_matrix):
    """计算Matthew相关系数 (MCC)"""
    TP, FP = conf_matrix[0]
    FN, TN = conf_matrix[1]
    denom = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return (TP*TN - FP*FN) / denom if denom>0 else 0


def evaluate(Y_real, Y_pred):
    """
    一键执行分类评估，返回 precision, recall, f1, mcc, accuracy
    """
    cm = get_confusion_matrix(Y_real, Y_pred)
    precision = get_precision(cm)
    recall = get_recall(cm)
    f1 = get_f1score(cm)
    mcc = get_mcc(cm)
    acc = get_accuracy(cm)
    return precision, recall, f1, mcc, acc


# -----------------------------------
# 回归评估函数
# -----------------------------------
def regression_metrics(y_true, y_pred):
    """
    计算回归常用指标：MSE、MAE、R²、Pearson相关系数
    返回字典：{'mse','mae','r2','pearson'}
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)[0]
    return {'mse': mse, 'mae': mae, 'r2': r2, 'pearson': pearson}


# -----------------------------------
# 回归结果可视化
# -----------------------------------
def plot_regression_analysis(y_true, y_pred, output_path=None):
    """
    绘制回归分析图，共4个子图：
      1. 实测 vs 预测 散点图 + y=x参考线
      2. 残差分析 (残差 vs 预测)
      3. 值分布对比直方图 (真实 & 预测)
      4. 绝对误差分布 + 阈值线 (0.25,0.5,1.0)
    如果提供output_path，则保存PNG文件；否则返回Figure对象
    """
    residuals = y_pred - y_true
    abs_errors = np.abs(residuals)

    fig, axes = plt.subplots(2,2, figsize=(12,10))

    # 子图1: 实测 vs 预测
    ax = axes[0,0]
    ax.scatter(y_true, y_pred, alpha=0.5)
    minv = min(np.min(y_true), np.min(y_pred))
    maxv = max(np.max(y_true), np.max(y_pred))
    ax.plot([minv, maxv], [minv, maxv], '--')
    ax.set_title('实测值 vs 预测值')
    ax.set_xlabel('实测值')
    ax.set_ylabel('预测值')

    # 子图2: 残差分析
    ax = axes[0,1]
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.hlines(0, xmin=np.min(y_pred), xmax=np.max(y_pred), linestyles='--')
    ax.set_title('残差分析')
    ax.set_xlabel('预测值')
    ax.set_ylabel('残差 (预测 - 实测)')

    # 子图3: 值分布直方图
    ax = axes[1,0]
    ax.hist(y_true, bins=30, density=True, alpha=0.5, label='实测')
    ax.hist(y_pred, bins=30, density=True, alpha=0.5, label='预测')
    ax.set_title('值分布对比')
    ax.set_xlabel('数值')
    ax.set_ylabel('密度')
    ax.legend()

    # 子图4: 绝对误差分布
    ax = axes[1,1]
    ax.hist(abs_errors, bins=30, density=True, alpha=0.7)
    # 添加阈值线
    for thr in [0.25, 0.5, 1.0]:
        ax.axvline(thr, linestyle='--')
    ax.set_title('绝对误差分布')
    ax.set_xlabel('绝对误差')
    ax.set_ylabel('密度')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig

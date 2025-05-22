import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from matplotlib.font_manager import FontProperties
from typing import List, Dict, Any, Tuple, Optional
import os

def setup_matplotlib_fonts():
    """设置matplotlib中文字体"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    
    try:
        font = FontProperties(fname=r'C:\Windows\Fonts\simkai.ttf', size=10)
    except:
        font = FontProperties(size=10)
        
    return font

def plot_training_history(histories, best_fold=None, save_path=None):
    """绘制训练历史"""
    setup_matplotlib_fonts()
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
    
    if save_path:
        plt.savefig(save_path, dpi=120)
        
    return plt.gcf()

def plot_predictions(y_true, y_pred, metrics=None, save_path=None):
    """绘制预测vs真实值散点图"""
    font = setup_matplotlib_fonts()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)

    # 添加理想线
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.xlabel('实际值', fontsize=12, fontproperties=font)
    plt.ylabel('预测值', fontsize=12, fontproperties=font)
    plt.title('多头注意力RCNN模型预测结果', fontsize=14, fontproperties=font)
    plt.grid(True)
    
    # 添加性能指标
    if metrics:
        text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))])
        plt.annotate(text, xy=(0.05, 0.95), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=120)
        
    return plt.gcf()

def visualize_attention_weights(model, data_loader, device, n_samples=5, output_dir=None):
    """可视化注意力权重"""
    model.eval()
    
    samples = []
    for X_batch, mask_batch, y_batch in data_loader:
        if len(samples) >= n_samples:
            break
            
        X_batch = X_batch.to(device)
        mask_batch = mask_batch.to(device)
        
        with torch.no_grad():
            pred, attention = model(X_batch, mask_batch)
            
        for i in range(min(len(X_batch), n_samples - len(samples))):
            seq_len = mask_batch[i].sum().cpu().item()
            
            if len(attention.shape) == 4:  # (batch, heads, seq_len, seq_len)
                # 取所有头的平均，然后对每个位置求和
                att_weights = attention[i].mean(dim=0)  # (seq_len, seq_len)
                position_importance = att_weights.sum(dim=0)[:seq_len].cpu().numpy()
            else:
                position_importance = attention[i, :seq_len].cpu().numpy()
            
            samples.append({
                'x': X_batch[i].cpu().numpy(),
                'mask': mask_batch[i].cpu().numpy(),
                'y_true': y_batch[i].item(),
                'y_pred': pred[i].item(),
                'attention': position_importance,
                'seq_len': seq_len
            })
    
    # 创建可视化
    font = setup_matplotlib_fonts()
    plt.figure(figsize=(12, 4 * n_samples))
    
    for i, sample in enumerate(samples):
        plt.subplot(n_samples, 1, i + 1)
        plt.bar(
            x=list(range(sample['seq_len'])), 
            height=sample['attention'], 
            width=1.0
        )
        plt.xlabel('序列位置', fontproperties=font)
        plt.ylabel('注意力权重', fontproperties=font)
        plt.title(f'样本 #{i}: 真实值={sample["y_true"]:.4f}, 预测值={sample["y_pred"]:.4f}', 
                 fontproperties=font)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(f"{output_dir}/attention_weights.png", dpi=120)
        
    return plt.gcf()

def visualize_multihead_attention(model, data_loader, device, n_samples=3, output_dir=None):
    """可视化多头注意力权重分布"""
    # 实现代码，参考前面详细介绍的可解释性可视化方案

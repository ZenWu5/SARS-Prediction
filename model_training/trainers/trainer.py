import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import time
import logging
from tqdm import tqdm
import gc
import os
import math
from typing import Dict, List, Tuple, Any, Optional

class Trainer:
    def __init__(self, model, config, device):
        """
        模型训练器
        
        参数:
            model: 待训练的模型
            config: 配置对象
            device: 训练设备
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logging.getLogger("trainer")
        
        # 初始化优化器
        self.optimizer = optim.Adam(model.parameters(), lr=config.training.lr)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 动态学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mse': [],
            'val_mae': [],
            'val_r2': [],
            'val_pearson': []
        }
        
        # 最佳模型状态
        self.best_model_state = None
        self.best_val_loss = float('inf')
        
    def _create_scheduler(self):
        """创建学习率调度器"""
        warmup_epochs = min(5, self.config.training.epochs // 10)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs  # 线性预热
            else:
                # 余弦退火
                min_lr_factor = 5e-2
                total_epochs = self.config.training.epochs
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return min_lr_factor + (1 - min_lr_factor) * cosine_decay
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        train_losses = []
        
        with tqdm(train_loader, desc="Training", leave=False) as pbar:
            for X_batch, mask_batch, y_batch in pbar:
                X_batch = X_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 前向传播
                y_pred, _ = self.model(X_batch, mask_batch)
                y_pred = y_pred.squeeze()
                loss = self.criterion(y_pred, y_batch)
                
                # 反向传播与优化
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # 记录损失
                train_losses.append(loss.item())
                pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        return sum(train_losses) / len(train_losses)
    
    def validate(self, val_loader):
        """验证模型性能"""
        self.model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for X_val, mask_val, y_val in val_loader:
                X_val = X_val.to(self.device)
                mask_val = mask_val.to(self.device)
                y_val = y_val.to(self.device)
                
                # 前向传播
                val_pred, _ = self.model(X_val, mask_val)
                val_pred = val_pred.squeeze()
                val_loss = self.criterion(val_pred, y_val)
                
                # 记录结果
                val_losses.append(val_loss.item())
                val_preds.append(val_pred.cpu().numpy())
                val_targets.append(y_val.cpu().numpy())
        
        # 计算指标
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        
        metrics = {
            'loss': sum(val_losses) / len(val_losses),
            'mse': mean_squared_error(val_targets, val_preds),
            'mae': mean_absolute_error(val_targets, val_preds),
            'r2': r2_score(val_targets, val_preds),
            'pearson': pearsonr(val_targets, val_preds)[0]
        }
        
        return metrics
    
    def train(self, train_loader, val_loader):
        """训练模型"""
        patience = self.config.training.patience
        patience_counter = 0
        
        for epoch in range(self.config.training.epochs):
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录训练历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_mse'].append(val_metrics['mse'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['val_r2'].append(val_metrics['r2'])
            self.history['val_pearson'].append(val_metrics['pearson'])
            
            # 打印训练状态
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.training.epochs}: "
                f"train_loss={train_loss:.6f}, val_loss={val_metrics['loss']:.6f}, "
                f"val_mse={val_metrics['mse']:.6f}, val_r2={val_metrics['r2']:.6f}, "
                f"lr={current_lr:.6f}"
            )
            
            # 保存最佳模型
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                self.logger.info(f"新的最佳模型已保存!")
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= patience:
                self.logger.info(f"早停触发！{patience}个epoch没有改善")
                break
        
        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        return self.history
    
    def evaluate(self, test_loader):
        """在测试集上评估模型"""
        self.model.eval()
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for X_test, mask_test, y_test in test_loader:
                X_test = X_test.to(self.device)
                mask_test = mask_test.to(self.device)
                
                test_pred, _ = self.model(X_test, mask_test)
                
                test_preds.append(test_pred.cpu().numpy())
                test_targets.append(y_test.numpy())
        
        # 合并批次结果
        test_preds = np.concatenate(test_preds).squeeze()
        test_targets = np.concatenate(test_targets)
        
        # 计算指标
        results = {
            'mse': mean_squared_error(test_targets, test_preds),
            'mae': mean_absolute_error(test_targets, test_preds),
            'r2': r2_score(test_targets, test_preds),
            'pearson': pearsonr(test_targets, test_preds)[0],
            'predictions': test_preds,
            'targets': test_targets
        }
        
        return results
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }, path)
        self.logger.info(f"模型已保存至 {path}")
        
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.logger.info(f"模型已从 {path} 加载")
        
    def cleanup(self):
        """清理资源"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

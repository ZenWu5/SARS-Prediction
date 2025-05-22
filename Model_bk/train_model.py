# -*- coding: utf-8 -*-
"""
train_model.py

本模块包含：
- 深度模型训练函数：train_model（单次训练）和 train_with_kfold（K折交叉验证）
- 基线模型接口：SVM、随机森林、KNN、贝叶斯、逻辑回归
"""
import os
import time
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

import validation  # 导入评估工具


def train_model(
    X_train, masks_train, y_train,
    X_val, masks_val, y_val,
    model, training_params,
    device: torch.device
):
    """
    单模型训练（含早停与动态学习率）

    参数:
        X_train, masks_train, y_train: 训练集输入、掩码、标签（numpy数组）
        X_val, masks_val, y_val: 验证集输入、掩码、标签（numpy数组）
        model: 已创建并to(device)的模型实例
        training_params: 字典，包含 'lr','batch_size','epochs','patience'
        device: 运行设备
    返回:
        best_model: 最佳模型状态
        val_results: 验证集性能字典
        history: 训练历史记录字典
    """
    # 清理缓存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 构建DataLoader
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
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(masks_val, dtype=torch.bool),
        torch.tensor(y_val, dtype=torch.float32)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_params['batch_size'],
        shuffle=False
    )

    # 损失函数与优化器
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=training_params['lr'])

    # 学习率调度：线性warmup + 指数衰减
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch+1) / warmup_epochs
        else:
            return 0.95 ** (epoch - warmup_epochs)
    scheduler = LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = training_params['patience']

    # 记录历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse': [],
        'val_mae': [],
        'val_r2': [],
        'val_pearson': []
    }

    # 训练循环
    for epoch in range(training_params['epochs']):
        model.train()
        train_losses = []
        for Xb, Mb, yb in train_loader:
            Xb, Mb, yb = Xb.to(device), Mb.to(device), yb.to(device)
            pred, _ = model(Xb, Mb)
            pred = pred.squeeze()
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        avg_train = np.mean(train_losses)

        # 验证
        model.eval()
        val_losses, preds, trues = [], [], []
        with torch.no_grad():
            for Xv, Mv, yv in val_loader:
                Xv, Mv, yv = Xv.to(device), Mv.to(device), yv.to(device)
                vp, _ = model(Xv, Mv)
                vp = vp.squeeze()
                loss_v = criterion(vp, yv)
                val_losses.append(loss_v.item())
                preds.append(vp.cpu().numpy())
                trues.append(yv.cpu().numpy())
        avg_val = np.mean(val_losses)
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        # 计算常用回归指标
        metrics = validation.regression_metrics(trues, preds)

        # 学习率更新
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: LR={current_lr:.5f}, TrainLoss={avg_train:.5f}, ValLoss={avg_val:.5f}")

        # 保存历史
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['val_mse'].append(metrics['mse'])
        history['val_mae'].append(metrics['mae'])
        history['val_r2'].append(metrics['r2'])
        history['val_pearson'].append(metrics['pearson'])

        # 早停
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发，{patience} 个 epoch 验证无改善。终止训练。")
                break

    # 加载最佳模型状态
    model.load_state_dict(best_state)
    return model, metrics, history


def train_with_kfold(
    X, masks, y, model_fn, training_params, device, k_folds=5, seed=42
):
    """
    K折交叉验证训练

    参数:
        X, masks, y: 完整数据
        model_fn: 可调用返回新模型的函数（如 lambda: get_model(...)）
        training_params: 同 train_model
        device: 计算设备
        k_folds: 折数
        seed: 随机种子
    返回:
        best_model: 验证MSE最优的模型
        fold_results: 每折结果列表
        all_histories: 各折训练历史
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    best_mse = float('inf')
    best_model = None
    fold_results = []
    all_histories = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"开始第 {fold+1}/{k_folds} 折训练...")
        X_tr, M_tr, y_tr = X[train_idx], masks[train_idx], y[train_idx]
        X_va, M_va, y_va = X[val_idx], masks[val_idx], y[val_idx]
        model = model_fn().to(device)
        model, metrics, history = train_model(
            X_tr, M_tr, y_tr,
            X_va, M_va, y_va,
            model, training_params, device
        )
        fold_results.append(metrics)
        all_histories.append(history)
        if metrics['mse'] < best_mse:
            best_mse = metrics['mse']
            best_model = model
    return best_model, fold_results, all_histories


# -----------------------------
# 基线模型接口
# -----------------------------
def run_baseline_model(
    name, X_train, y_train, X_test, y_test
):
    """
    运行基线分类模型，并返回性能指标
    参数:
        name: 'svm','rf','knn','bayes','lr'
        X_train, y_train, X_test, y_test: numpy数组
    返回:
        result: 字典 {'train_acc', 'train_pre', 'train_rec', 'train_f1', 'train_mcc',
                       'val_acc','val_pre','val_rec','val_f1','val_mcc'}
    """
    name = name.lower()
    if name == 'svm':
        clf = SVC(gamma='auto', class_weight='balanced', probability=True)
    elif name in ('rf','randomforest'):
        clf = RandomForestClassifier()
    elif name == 'knn':
        clf = KNeighborsClassifier()
    elif name == 'bayes':
        clf = GaussianNB()
    elif name in ('lr','logistic','logisticregression'):
        clf = LogisticRegression(max_iter=1000)
    else:
        raise ValueError(f"未知基线模型: {name}")
    # 训练
    clf.fit(X_train, y_train)
    # 训练集性能
    tr_pred = clf.predict(X_train)
    res = {
        'train_acc': accuracy_score(y_train, tr_pred),
        'train_pre': precision_score(y_train, tr_pred),
        'train_rec': recall_score(y_train, tr_pred),
        'train_f1' : f1_score(y_train, tr_pred),
        'train_mcc': matthews_corrcoef(y_train, tr_pred)
    }
    # 验证集性能
    te_pred = clf.predict(X_test)
    res.update({
        'val_acc': accuracy_score(y_test, te_pred),
        'val_pre': precision_score(y_test, te_pred),
        'val_rec': recall_score(y_test, te_pred),
        'val_f1' : f1_score(y_test, te_pred),
        'val_mcc': matthews_corrcoef(y_test, te_pred)
    })
    return res

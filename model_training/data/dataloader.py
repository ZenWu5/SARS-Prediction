import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import KFold
from typing import List, Tuple, Dict, Any

def create_train_val_test_split(dataset, test_size=0.15, seed=42):
    """创建训练、验证和测试集划分"""
    N = len(dataset)
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 随机排列索引
    perm = np.random.permutation(N)
    n_test = int(test_size * N)
    
    test_idx = perm[:n_test]
    train_val_idx = perm[n_test:]
    
    test_set = Subset(dataset, test_idx)
    train_val_set = Subset(dataset, train_val_idx)
    
    return train_val_set, test_set, train_val_idx

def create_kfold_dataloaders(dataset, train_val_idx, batch_size=256, k_folds=5, 
                            seed=42, num_workers=4):
    """创建K折交叉验证的数据加载器"""
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    
    fold_dataloaders = []
    
    for train_subidx, val_subidx in kf.split(train_val_idx):
        # 获取训练和验证数据子集的索引
        train_idx = [train_val_idx[i] for i in train_subidx]
        val_idx = [train_val_idx[i] for i in val_subidx]
        
        # 创建训练和验证数据加载器
        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers//2,
            pin_memory=torch.cuda.is_available()
        )
        
        fold_dataloaders.append((train_loader, val_loader))
    
    return fold_dataloaders

def create_test_dataloader(test_dataset, batch_size=256, num_workers=4):
    """创建测试数据加载器"""
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

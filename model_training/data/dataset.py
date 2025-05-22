import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger("dataset")

class H5ProteinDataset(Dataset):
    def __init__(self, h5_path, max_seq_len=None, normalize=True):
        """
        按需从HDF5文件加载蛋白质序列嵌入数据。
        
        参数:
            h5_path: HDF5文件路径
            max_seq_len: 最大序列长度，默认使用数据集中的最大长度
            normalize: 是否对特征进行标准化
        """
        self.h5_path = h5_path if h5_path.endswith('.h5') else h5_path + '.h5'
        
        # 读取元数据
        with h5py.File(self.h5_path, 'r') as f:
            self.total = int(f.attrs['total_samples'])
            self.emb_dim = int(f.attrs['embedding_dim'])
            self.targets = f['mean_log10Ka'][:].astype(np.float32)
            self.diff_counts = f['diff_count'][:].astype(np.int16)
            self.seq_lengths = np.array([f['embeddings'][f'emb_{i}'].shape[0] 
                                        for i in range(self.total)], dtype=np.int32)
        
        self.max_seq_len = max_seq_len or int(self.seq_lengths.max())
        self._file = None
        
        self.normalize = normalize
        self.feature_mean = None
        self.feature_std = None
        
        if normalize:
            self._compute_normalization_stats()
    
    def __len__(self) -> int:
        return self.total
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 按需加载HDF5文件
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')
            
        # 获取embedding
        emb = self._file['embeddings'][f'emb_{idx}'][:].astype(np.float32)
        
        # 标准化
        if self.normalize and self.feature_mean is not None:
            emb = (emb - self.feature_mean) / self.feature_std
            
        # 处理序列长度
        L = emb.shape[0]
        if L > self.max_seq_len:
            proc = emb[:self.max_seq_len]
            mask = np.ones(self.max_seq_len, dtype=bool)
        else:
            pad_len = self.max_seq_len - L
            proc = np.vstack([emb, np.zeros((pad_len, self.emb_dim), dtype=np.float32)])
            mask = np.concatenate([np.ones(L, dtype=bool), np.zeros(pad_len, dtype=bool)])
            
        target = float(self.targets[idx])
        
        return (
            torch.from_numpy(proc),
            torch.from_numpy(mask),
            torch.tensor(target, dtype=torch.float32)
        )
    
    def _compute_normalization_stats(self):
        """计算所有嵌入向量的均值和标准差，用于标准化"""
        # 计算标准化的统计量
        # [此处省略实现]

    def get_diff_counts(self) -> np.ndarray:
        """返回所有样本的diff_count"""
        return self.diff_counts
        
    def close(self):
        """关闭HDF5文件句柄"""
        if self._file is not None:
            self._file.close()
            self._file = None

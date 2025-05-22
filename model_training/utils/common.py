import numpy as np
import torch
import random
import os
import logging
import time
from typing import Dict, Any, Optional

def set_seed(seed):
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logging(log_dir=None, level=logging.INFO):
    """设置日志记录"""
    # 创建日志目录
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'train_{time.strftime("%Y%m%d_%H%M%S")}.log')
    else:
        log_file = None
        
    # 配置日志格式
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w' if log_file else None
    )
    
    # 添加控制台处理器
    if log_file:
        console = logging.StreamHandler()
        console.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
    
    return logging.getLogger('mhrcnn')

def create_output_dir(base_dir):
    """创建带有时间戳的输出目录"""
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

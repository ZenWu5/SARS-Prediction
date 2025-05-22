import argparse
from dataclasses import dataclass, field
import yaml
import os

@dataclass
class ModelConfig:
    input_dim: int = 1152  
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.2
    conv_channels: list = field(default_factory=lambda: [128, 64])
    conv_kernel_sizes: list = field(default_factory=lambda: [5, 3])
    num_heads: int = 4

@dataclass
class TrainingConfig:
    lr: float = 0.001
    batch_size: int = 256
    epochs: int = 100
    patience: int = 8
    k_folds: int = 10
    seed: int = 1044
    
@dataclass
class DataConfig:
    data_path: str = 'preprocess/ESM/output/sampled_output_esm_embeddings.h5'
    max_seq_len: int = None
    normalize: bool = True
    test_size: float = 0.15
    
@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = 'ModelResults/MultiHeadAttentionRCNN'
    device: str = 'cuda'
    
    def __post_init__(self):
        # 自动检测设备
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

def get_config_from_args():
    """从命令行参数创建配置"""
    parser = argparse.ArgumentParser(description='多头注意力RCNN训练脚本')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, help='h5数据文件路径')
    parser.add_argument('--max_seq_len', type=int, help='最大序列长度')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, help='LSTM层数')
    parser.add_argument('--dropout', type=float, help='Dropout比例')
    parser.add_argument('--num_heads', type=int, help='注意力头数量')
    
    # 训练参数
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--epochs', type=int, help='最大训练轮数')
    parser.add_argument('--patience', type=int, help='早停耐心值')
    parser.add_argument('--k_folds', type=int, help='交叉验证折数')
    parser.add_argument('--seed', type=int, help='随机种子')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, help='结果输出目录')
    parser.add_argument('--config', type=str, help='YAML配置文件路径')
    
    args = parser.parse_args()
    
    # 基准配置
    config = Config()
    
    # 如果提供了YAML配置，从文件加载
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            # 更新配置
            # [此处省略YAML配置转换为对象的代码]
    
    # 命令行参数覆盖配置文件
    for arg, value in vars(args).items():
        if value is not None and arg != 'config':
            # 根据参数名设置相应配置
            if arg in vars(config.data):
                setattr(config.data, arg, value)
            elif arg in vars(config.model):
                setattr(config.model, arg, value)
            elif arg in vars(config.training):
                setattr(config.training, arg, value)
            else:
                setattr(config, arg, value)
                
    return config

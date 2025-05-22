import sys
import argparse

def main():
    """主入口点"""
    parser = argparse.ArgumentParser(description='多头注意力RCNN蛋白质序列分析')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'visualize'], 
                      default='train', help='运行模式')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    
    # 解析命令行中的其他参数
    args, unknown_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown_args
    
    if args.mode == 'train':
        # 导入并运行训练脚本
        from train import main as train_main
        train_main()
    elif args.mode == 'evaluate':
        # 导入并运行评估脚本
        from evaluate import main as evaluate_main
        evaluate_main()
    elif args.mode == 'visualize':
        # 导入并运行可视化脚本
        from visualize import main as visualize_main
        visualize_main()

if __name__ == "__main__":
    main()

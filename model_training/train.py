import os
import torch
import numpy as np
import json

from config.config import get_config_from_args
from data.dataset import H5ProteinDataset
from data.dataloader import (create_train_val_test_split, 
                           create_kfold_dataloaders,
                           create_test_dataloader)
from model_training.models.DLModels import MultiHeadAttentionRCNN  
from trainers.trainer import Trainer
from utils.common import set_seed, setup_logging, create_output_dir
from utils.visualization import (plot_training_history, 
                               plot_predictions,
                               visualize_attention_weights)
from utils.metrics import calculate_regression_metrics, format_metrics

def main():
    """主训练函数"""
    # 1. 解析配置
    config = get_config_from_args()
    
    # 2. 准备输出目录和日志
    output_dir = create_output_dir(config.output_dir)
    logger = setup_logging(output_dir)
    logger.info(f"结果将保存至: {output_dir}")
    
    # 3. 设置随机种子
    set_seed(config.training.seed)
    logger.info(f"设置随机种子: {config.training.seed}")
    
    # 4. 加载数据集
    device = torch.device(config.device)
    dataset = H5ProteinDataset(
        config.data.data_path, 
        max_seq_len=config.data.max_seq_len,
        normalize=config.data.normalize
    )
    N = len(dataset)
    logger.info(f"HDF5 样本总数: {N}, pad长度={dataset.max_seq_len}, emb_dim={dataset.emb_dim}")
    
    # 5. 拆分数据集
    train_val_set, test_set, train_val_idx = create_train_val_test_split(
        dataset, 
        test_size=config.data.test_size,
        seed=config.training.seed
    )
    logger.info(f"切分完成：训练+验证={len(train_val_set)}, 测试={len(test_set)}")
    
    # 6. 创建K折数据加载器
    fold_dataloaders = create_kfold_dataloaders(
        dataset,
        train_val_idx,
        batch_size=config.training.batch_size,
        k_folds=config.training.k_folds,
        seed=config.training.seed
    )
    
    # 7. K折交叉验证训练
    all_histories = []
    fold_results = []
    best_model = None
    best_val_loss = float('inf')
    best_fold = -1
    
    # 确定输入维度
    for x, _, _ in train_val_set:
        input_dim = x.shape[1]
        break
    
    for fold, (train_loader, val_loader) in enumerate(fold_dataloaders):
        logger.info(f"\n======= 训练第 {fold+1}/{config.training.k_folds} 折 =======")
        
        # 初始化模型
        model = MultiHeadAttentionRCNN(
            input_dim=input_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            num_heads=config.model.num_heads
        )
        
        # 创建训练器
        trainer = Trainer(model, config, device)
        
        # 训练模型
        history = trainer.train(train_loader, val_loader)
        all_histories.append(history)
        
        # 验证性能
        val_metrics = trainer.validate(val_loader)
        fold_results.append(val_metrics)
        
        logger.info(f"第 {fold+1} 折验证结果: {format_metrics(val_metrics)}")
        
        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model = model
            best_fold = fold
            
            # 保存折内最佳模型
            fold_model_path = os.path.join(output_dir, f'best_model_fold_{fold+1}.pt')
            trainer.save_model(fold_model_path)
        
    # 8. 可视化训练历史
    history_fig = plot_training_history(
        all_histories, 
        best_fold=best_fold,
        save_path=os.path.join(output_dir, 'training_history.png')
    )
    
    # 9. 在测试集上评估最佳模型
    test_loader = create_test_dataloader(
        test_set,
        batch_size=config.training.batch_size
    )
    
    final_trainer = Trainer(best_model, config, device)
    test_results = final_trainer.evaluate(test_loader)
    
    logger.info(f"测试集评估结果: {format_metrics(test_results)}")
    
    # 10. 可视化预测结果
    plot_predictions(
        test_results['targets'], 
        test_results['predictions'],
        {k: v for k, v in test_results.items() if k not in ['predictions', 'targets']},
        save_path=os.path.join(output_dir, 'predictions.png')
    )
    
    # 11. 可视化注意力权重
    visualize_attention_weights(
        best_model, 
        test_loader, 
        device,
        n_samples=5, 
        output_dir=output_dir
    )
    
    # 12. 保存最终模型和结果
    final_model_path = os.path.join(output_dir, 'best_model.pt')
    final_trainer.save_model(final_model_path)
    
    # 保存折叠结果
    with open(os.path.join(output_dir, 'fold_results.json'), 'w') as f:
        json.dump([{k: float(v) for k, v in res.items()} for res in fold_results], f, indent=2)
    
    # 保存测试结果
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        test_metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                      for k, v in test_results.items() 
                      if k not in ['predictions', 'targets']}
        json.dump(test_metrics, f, indent=2)
    
    # 清理
    dataset.close()
    logger.info("训练完成!")

if __name__ == "__main__":
    main()

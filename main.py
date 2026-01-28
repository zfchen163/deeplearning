#!/usr/bin/env python
"""
生产级深度学习项目启动脚本
Production-grade Deep Learning Project Launcher

使用方法：
    python main.py --config config_example.yaml --mode train
    python main.py --config config_example.yaml --mode eval
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 导入生产级代码
from production_code_examples import *
from advanced_project_example import *


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Production-grade Deep Learning Project"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config_example.yaml',
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'eval', 'inference'],
        default='train',
        help='运行模式'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='检查点路径（用于恢复训练或评估）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='计算设备'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='调试模式'
    )
    
    return parser.parse_args()


def setup_environment(config: ProjectConfig, debug: bool = False):
    """设置运行环境"""
    # 创建日志器
    log_level = logging.DEBUG if debug else logging.INFO
    logger = ProjectLogger('DeepLearning', log_dir=config.logs_dir, level=log_level)
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    logger.info("=" * 60)
    logger.info(f"Project: {config.project_name}")
    logger.info(f"Version: {config.version}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    return logger


def create_dummy_dataset(num_samples: int = 1000, num_classes: int = 10):
    """创建虚拟数据集用于演示"""
    import logging
    logging.getLogger('DeepLearning').info(f"Creating dummy dataset with {num_samples} samples")
    
    X = torch.randn(num_samples, 3, 224, 224)
    y = torch.randint(0, num_classes, (num_samples,))
    
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    test_size = num_samples - train_size - val_size
    
    train_dataset = TensorDataset(X[:train_size], y[:train_size])
    val_dataset = TensorDataset(X[train_size:train_size+val_size], 
                               y[train_size:train_size+val_size])
    test_dataset = TensorDataset(X[train_size+val_size:], 
                                y[train_size+val_size:])
    
    return train_dataset, val_dataset, test_dataset


def train_mode(config: ProjectConfig, logger: ProjectLogger, checkpoint_path: str = None):
    """训练模式"""
    logger.info("Entering TRAIN mode...")
    
    # 创建模型
    model = ModelFactory.create_model(config.model_config, logger)
    logger.info(f"Model created: {config.model_config.architecture}")
    
    # 显示模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # 创建数据集
    train_dataset, val_dataset, test_dataset = create_dummy_dataset(
        num_samples=1000,
        num_classes=config.model_config.num_classes
    )
    
    # 创建数据加载器
    data_module = AdvancedDataModule(config.data_config)
    train_loader = data_module.get_train_loader(train_dataset)
    val_loader = data_module.get_val_loader(val_dataset)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # 创建训练器
    trainer = AdvancedTrainer(config, model, logger)
    
    # 加载检查点（如果提供）
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 开始训练
    try:
        trainer.train(train_loader, val_loader, criterion)
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def eval_mode(config: ProjectConfig, logger: ProjectLogger, checkpoint_path: str):
    """评估模式"""
    logger.info("Entering EVAL mode...")
    
    if not checkpoint_path:
        logger.error("Checkpoint path required for eval mode")
        return
    
    # 加载模型
    model = ModelFactory.create_model(config.model_config, logger)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    model.eval()
    
    logger.info(f"Model loaded from {checkpoint_path}")
    
    # 创建测试数据集
    train_dataset, val_dataset, test_dataset = create_dummy_dataset(
        num_samples=1000,
        num_classes=config.model_config.num_classes
    )
    
    data_module = AdvancedDataModule(config.data_config)
    test_loader = data_module.get_val_loader(test_dataset)
    
    # 评估
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(config.device)
            targets = targets.to(config.device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            _, predicted = outputs.max(1)
            acc = (predicted == targets).float().mean().item()
            
            total_loss += loss.item()
            total_acc += acc
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    logger.info(f"Test Loss: {avg_loss:.4f}")
    logger.info(f"Test Accuracy: {avg_acc:.4f}")


def inference_mode(config: ProjectConfig, logger: ProjectLogger, checkpoint_path: str):
    """推理模式"""
    logger.info("Entering INFERENCE mode...")
    
    if not checkpoint_path:
        logger.error("Checkpoint path required for inference mode")
        return
    
    # 加载模型
    model = ModelFactory.create_model(config.model_config, logger)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    model.eval()
    
    # 创建样本输入
    dummy_input = torch.randn(1, 3, 224, 224).to(config.device)
    
    # 推理
    with torch.no_grad():
        output = model(dummy_input)
    
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Predicted class: {output.argmax(1).item()}")


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 加载配置
    config = ProjectConfig.load(args.config)
    config.device = args.device
    
    # 设置环境
    logger = setup_environment(config, debug=args.debug)
    
    # 根据模式执行
    try:
        if args.mode == 'train':
            train_mode(config, logger, args.checkpoint)
        elif args.mode == 'eval':
            eval_mode(config, logger, args.checkpoint)
        elif args.mode == 'inference':
            inference_mode(config, logger, args.checkpoint)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
        
        logger.info("Process completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    # 添加必要的导入
    import logging
    
    main()

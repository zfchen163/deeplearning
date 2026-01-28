"""
生产级深度学习项目实战示例
Production-grade Deep Learning Project Examples

包含完整的项目结构、配置管理、错误处理和监控
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import pickle
import time
from datetime import datetime
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.models as models


# ==================== 项目配置管理 ====================

@dataclass
class DataConfig:
    """数据配置"""
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    shuffle: bool = True
    augment: bool = True


@dataclass
class ModelConfig:
    """模型配置"""
    architecture: str = 'resnet18'
    pretrained: bool = True
    num_classes: int = 10
    dropout_rate: float = 0.5
    freeze_backbone: bool = False
    freeze_until_layer: int = 0


@dataclass
class TrainingConfig:
    """训练配置"""
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = 'adamw'  # 'sgd', 'adam', 'adamw'
    scheduler: str = 'cosine'  # 'step', 'linear', 'exponential', 'cosine'
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    accumulation_steps: int = 1


@dataclass
class RegularizationConfig:
    """正则化配置"""
    use_dropout: bool = True
    dropout_rate: float = 0.5
    use_batch_norm: bool = True
    use_weight_decay: bool = True
    weight_decay: float = 1e-5
    use_label_smoothing: bool = False
    label_smoothing: float = 0.1


@dataclass
class EarlyStoppingConfig:
    """早停配置"""
    enabled: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    monitor: str = 'val_loss'  # 'val_loss' or 'val_accuracy'


@dataclass
class ProjectConfig:
    """项目总配置"""
    project_name: str = "CV_Project"
    description: str = ""
    version: str = "1.0.0"
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 子配置
    data_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    regularization_config: RegularizationConfig = field(default_factory=RegularizationConfig)
    early_stopping_config: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    
    # 路径配置
    project_root: str = "./projects"
    data_dir: str = "./data"
    checkpoints_dir: str = "./checkpoints"
    logs_dir: str = "./logs"
    results_dir: str = "./results"
    
    def __post_init__(self):
        """初始化后创建必要的目录"""
        for attr in ['project_root', 'data_dir', 'checkpoints_dir', 'logs_dir', 'results_dir']:
            path = getattr(self, attr)
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def save(self, filepath: str):
        """保存配置到YAML文件"""
        config_dict = asdict(self)
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        print(f"Config saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ProjectConfig':
        """从YAML文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 递归构建嵌套配置
        data_config = DataConfig(**config_dict.pop('data_config', {}))
        model_config = ModelConfig(**config_dict.pop('model_config', {}))
        training_config = TrainingConfig(**config_dict.pop('training_config', {}))
        regularization_config = RegularizationConfig(**config_dict.pop('regularization_config', {}))
        early_stopping_config = EarlyStoppingConfig(**config_dict.pop('early_stopping_config', {}))
        
        return cls(
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            regularization_config=regularization_config,
            early_stopping_config=early_stopping_config,
            **config_dict
        )


# ==================== 高级日志管理 ====================

class ProjectLogger:
    """项目级日志管理器"""
    
    def __init__(self, name: str, log_dir: str = './logs', level=10):
        import logging
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 清除已有的handler
        self.logger.handlers.clear()
        
        # 格式化器
        formatter = logging.Formatter(
            '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            Path(log_dir) / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)


# ==================== 性能指标追踪 ====================

class MetricsTracker:
    """性能指标追踪器"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.best_metrics = {}
    
    def update(self, phase: str, **kwargs):
        """更新指标"""
        for key, value in kwargs.items():
            metric_name = f"{phase}/{key}"
            self.metrics[metric_name].append(value)
    
    def get_average(self, phase: str, key: str) -> float:
        """获取平均值"""
        metric_name = f"{phase}/{key}"
        if not self.metrics[metric_name]:
            return 0.0
        return float(np.mean(self.metrics[metric_name]))
    
    def reset(self, phase: str = None):
        """重置指标"""
        if phase:
            keys_to_remove = [k for k in self.metrics if k.startswith(phase)]
            for k in keys_to_remove:
                del self.metrics[k]
        else:
            self.metrics.clear()
    
    def get_best(self, phase: str, key: str) -> Optional[float]:
        """获取最佳值"""
        metric_name = f"{phase}/{key}"
        return self.best_metrics.get(metric_name)
    
    def update_best(self, phase: str, key: str, value: float):
        """更新最佳值"""
        metric_name = f"{phase}/{key}"
        if metric_name not in self.best_metrics or value < self.best_metrics[metric_name]:
            self.best_metrics[metric_name] = value


# ==================== 高级数据加载 ====================

class AdvancedDataModule:
    """高级数据模块"""
    
    def __init__(self, config: DataConfig, transform_train=None, transform_val=None):
        self.config = config
        self.transform_train = transform_train or transforms.ToTensor()
        self.transform_val = transform_val or transforms.ToTensor()
    
    def get_train_loader(self, dataset: Dataset) -> DataLoader:
        """获取训练数据加载器"""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
    
    def get_val_loader(self, dataset: Dataset) -> DataLoader:
        """获取验证数据加载器"""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False
        )


# ==================== 模型工厂 ====================

class ModelFactory:
    """模型工厂类"""
    
    SUPPORTED_MODELS = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'vgg16': models.vgg16,
        'vgg19': models.vgg19,
        'mobilenet_v2': models.mobilenet_v2,
        'efficientnet_b0': models.efficientnet_b0,
    }
    
    @classmethod
    def create_model(cls, config: ModelConfig, logger: Optional[ProjectLogger] = None) -> nn.Module:
        """创建模型"""
        if config.architecture not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported architecture: {config.architecture}")
        
        # 获取基础模型
        model_fn = cls.SUPPORTED_MODELS[config.architecture]
        model = model_fn(pretrained=config.pretrained)
        
        # 冻结backbone（可选）
        if config.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        
        # 修改分类层
        if config.architecture.startswith('resnet'):
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout_rate),
                nn.Linear(512, config.num_classes)
            )
        
        if logger:
            logger.info(f"Model created: {config.architecture}")
        
        return model


# ==================== 高级训练器 ====================

class AdvancedTrainer:
    """高级训练器"""
    
    def __init__(self, config: ProjectConfig, model: nn.Module, logger: ProjectLogger):
        self.config = config
        self.model = model.to(config.device)
        self.logger = logger
        self.metrics_tracker = MetricsTracker()
        self.writer = SummaryWriter(log_dir=config.logs_dir)
        self.global_step = 0
        
        # 初始化优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = torch.cuda.amp.GradScaler() if config.training_config.mixed_precision else None
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        opt_name = self.config.training_config.optimizer.lower()
        lr = self.config.training_config.learning_rate
        wd = self.config.training_config.weight_decay
        
        if opt_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        elif opt_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")
    
    def _create_scheduler(self) -> Any:
        """创建学习率调度器"""
        sched_name = self.config.training_config.scheduler.lower()
        num_epochs = self.config.training_config.num_epochs
        
        if sched_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs
            )
        elif sched_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=num_epochs // 3, gamma=0.1
            )
        elif sched_name == 'linear':
            return optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, total_iters=num_epochs
            )
        else:
            raise ValueError(f"Unsupported scheduler: {sched_name}")
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor], 
                   criterion: nn.Module) -> float:
        """单步训练"""
        self.model.train()
        images, targets = batch
        images = images.to(self.config.device)
        targets = targets.to(self.config.device)
        
        if self.scaler:
            # 混合精度训练
            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                loss = criterion(outputs, targets)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training_config.gradient_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 标准训练
            outputs = self.model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training_config.gradient_clip
            )
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        return loss.item()
    
    def val_step(self, batch: Tuple[torch.Tensor, torch.Tensor], 
                criterion: nn.Module) -> Tuple[float, float]:
        """单步验证"""
        self.model.eval()
        images, targets = batch
        images = images.to(self.config.device)
        targets = targets.to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model(images)
            loss = criterion(outputs, targets)
            
            _, predicted = outputs.max(1)
            accuracy = (predicted == targets).float().mean().item()
        
        return loss.item(), accuracy
    
    def train_epoch(self, train_loader: DataLoader, 
                   criterion: nn.Module, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.metrics_tracker.reset('train')
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            loss = self.train_step((images, targets), criterion)
            self.metrics_tracker.update('train', loss=loss)
            self.global_step += 1
            
            if (batch_idx + 1) % 100 == 0:
                avg_loss = self.metrics_tracker.get_average('train', 'loss')
                self.logger.info(
                    f"Epoch {epoch + 1} | Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f}"
                )
        
        avg_loss = self.metrics_tracker.get_average('train', 'loss')
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        
        return {'train_loss': avg_loss}
    
    def validate(self, val_loader: DataLoader, 
                criterion: nn.Module, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        self.metrics_tracker.reset('val')
        
        for images, targets in val_loader:
            loss, accuracy = self.val_step((images, targets), criterion)
            self.metrics_tracker.update('val', loss=loss, accuracy=accuracy)
        
        avg_loss = self.metrics_tracker.get_average('val', 'loss')
        avg_accuracy = self.metrics_tracker.get_average('val', 'accuracy')
        
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        self.writer.add_scalar('Accuracy/val', avg_accuracy, epoch)
        
        return {'val_loss': avg_loss, 'val_accuracy': avg_accuracy}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             criterion: nn.Module):
        """完整的训练循环"""
        self.logger.info("Starting training...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.training_config.num_epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader, criterion, epoch)
            
            # 验证
            val_metrics = self.validate(val_loader, criterion, epoch)
            
            # 调度器步进
            self.scheduler.step()
            
            # 日志
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.training_config.num_epochs} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Acc: {val_metrics['val_accuracy']:.4f}"
            )
            
            # 早停
            if self.config.early_stopping_config.enabled:
                if val_metrics['val_loss'] < best_val_loss - self.config.early_stopping_config.min_delta:
                    best_val_loss = val_metrics['val_loss']
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_config.patience:
                        self.logger.warning("Early stopping triggered")
                        break
        
        self.writer.close()


# ==================== 完整项目示例 ====================

def run_complete_project():
    """运行完整的项目"""
    
    # 1. 创建配置
    config = ProjectConfig(
        project_name="ImageClassification",
        version="1.0.0",
        data_config=DataConfig(batch_size=32, num_workers=4),
        model_config=ModelConfig(architecture='resnet18', num_classes=10),
        training_config=TrainingConfig(num_epochs=100, learning_rate=1e-3)
    )
    
    # 2. 创建日志器
    logger = ProjectLogger('ImageClassification', log_dir=config.logs_dir)
    logger.info(f"Project: {config.project_name}")
    logger.info(f"Config: {asdict(config)}")
    
    # 3. 保存配置
    config.save(Path(config.project_root) / 'config.yaml')
    
    # 4. 创建模型
    model = ModelFactory.create_model(config.model_config, logger)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. 创建训练器
    trainer = AdvancedTrainer(config, model, logger)
    
    # 6. 创建虚拟数据
    from torch.utils.data import TensorDataset
    X_train = torch.randn(1000, 3, 32, 32)
    y_train = torch.randint(0, 10, (1000,))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    X_val = torch.randn(200, 3, 32, 32)
    y_val = torch.randint(0, 10, (200,))
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 7. 训练
    criterion = nn.CrossEntropyLoss()
    trainer.train(train_loader, val_loader, criterion)
    
    logger.info("Training completed!")


if __name__ == '__main__':
    run_complete_project()

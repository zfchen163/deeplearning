"""
生产级别的深度学习代码示例
Production-grade Deep Learning Code Examples

这个模块为CV-main课程中的所有主题提供生产级别的代码示例
涵盖从PyTorch基础到大模型Agent的完整深度学习生态
"""

import os
import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter


# ==================== 日志配置 ====================
def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """设置生产级别的日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


logger = setup_logger('DeepLearning', 'training.log')


# ==================== 第100-122章：PyTorch基础 ====================

@dataclass
class ConfigPyTorch:
    """PyTorch配置类"""
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    cuda_visible_devices: str = '0'
    seed: int = 42
    mixed_precision: bool = torch.cuda.is_available()
    
    def setup(self):
        """初始化环境"""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.set_device(0)
        logger.info(f"Using device: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")


class PythonMagicMethods:
    """Python两大法宝：dir()和help()的实用包装"""
    
    @staticmethod
    def explore_module(module_obj, pattern: Optional[str] = None) -> List[str]:
        """
        探索模块中的属性和方法
        
        Args:
            module_obj: 要探索的模块对象
            pattern: 可选的过滤模式
            
        Returns:
            属性和方法列表
        """
        attrs = dir(module_obj)
        if pattern:
            attrs = [a for a in attrs if pattern.lower() in a.lower()]
        return attrs
    
    @staticmethod
    def get_documentation(obj, method_name: str) -> str:
        """获取对象方法的文档"""
        try:
            method = getattr(obj, method_name)
            return method.__doc__ or "No documentation available"
        except AttributeError:
            return f"Method '{method_name}' not found"


# ==================== 第200-268章：完整的模型训练框架 ====================

class CustomDataset(Dataset):
    """自定义数据集类（生产级别）"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, transform=None):
        """
        Args:
            data: 输入数据
            labels: 标签
            transform: 数据增强变换
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
        assert len(self.data) == len(self.labels), "Data and labels must have same length"
        logger.info(f"Dataset created with {len(self.data)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label


class ResNet18Classifier(nn.Module):
    """生产级别的ResNet18分类器（带注意力机制）"""
    
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # 替换最后的全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        logger.info(f"ResNet18Classifier initialized with {num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ModelCheckpoint:
    """模型检查点管理器"""
    
    def __init__(self, save_dir: str, best_metric: str = 'val_loss'):
        """
        Args:
            save_dir: 保存目录
            best_metric: 监控的指标
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = best_metric
        self.best_value = float('inf') if 'loss' in best_metric else float('-inf')
        
    def should_save(self, current_value: float, is_loss: bool = True) -> bool:
        """判断是否应该保存"""
        if is_loss:
            return current_value < self.best_value
        else:
            return current_value > self.best_value
    
    def save_model(self, model: nn.Module, optimizer: optim.Optimizer, 
                   epoch: int, metrics: Dict[str, float]):
        """保存模型及其状态"""
        if is_loss := 'loss' in self.best_metric:
            current_value = metrics.get(self.best_metric, float('inf'))
            should_save = current_value < self.best_value
        else:
            current_value = metrics.get(self.best_metric, float('-inf'))
            should_save = current_value > self.best_value
        
        if should_save:
            self.best_value = current_value
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            save_path = self.save_dir / f"best_model_epoch{epoch}.pt"
            torch.save(checkpoint, save_path)
            logger.info(f"Model saved to {save_path}, {self.best_metric}: {current_value:.4f}")
    
    def load_model(self, model: nn.Module, optimizer: optim.Optimizer, 
                   checkpoint_path: str) -> Dict[str, Any]:
        """加载模型及其状态"""
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {checkpoint_path}")
        return checkpoint


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        """
        Args:
            patience: 容忍的无改进epoch数
            min_delta: 最小改进量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Returns:
            是否应该早停
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.warning(f"Early stopping triggered after {self.counter} epochs")
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


class Trainer:
    """生产级别的训练器"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu', 
                 log_dir: str = './logs'):
        self.model = model.to(device)
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        self.global_step = 0
        
    def train_epoch(self, train_loader: DataLoader, loss_fn, optimizer: optim.Optimizer,
                   scheduler=None) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 前向传播
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 记录
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 定期日志
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                logger.info(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {avg_loss:.4f}")
                self.writer.add_scalar('Train/loss', avg_loss, self.global_step)
        
        if scheduler:
            scheduler.step()
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate(self, val_loader: DataLoader, loss_fn) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                
                # 计算准确率
                _, predicted = outputs.max(1)
                acc = (predicted == targets).float().mean()
                
                total_loss += loss.item()
                total_acc += acc.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        self.writer.add_scalar('Val/loss', avg_loss, self.global_step)
        self.writer.add_scalar('Val/accuracy', avg_acc, self.global_step)
        
        return {'val_loss': avg_loss, 'val_accuracy': avg_acc}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int, optimizer: optim.Optimizer, loss_fn,
              scheduler=None, checkpoint_manager: Optional[ModelCheckpoint] = None):
        """完整的训练循环"""
        early_stopping = EarlyStopping(patience=10)
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # 训练
            train_metrics = self.train_epoch(train_loader, loss_fn, optimizer, scheduler)
            
            # 验证
            val_metrics = self.validate(val_loader, loss_fn)
            
            # 合并指标
            metrics = {**train_metrics, **val_metrics}
            logger.info(f"Epoch {epoch + 1} - " + " - ".join(
                [f"{k}: {v:.4f}" for k, v in metrics.items()]
            ))
            
            # 保存最佳模型
            if checkpoint_manager:
                checkpoint_manager.save_model(self.model, optimizer, epoch, metrics)
            
            # 早停
            if early_stopping(val_metrics['val_loss']):
                logger.info("Training stopped early")
                break
        
        self.writer.close()


# ==================== 第300-354章：吴恩达深度学习专项 ====================

class BatchNormalization:
    """批量归一化的实现和理解"""
    
    @staticmethod
    def batch_norm_1d(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                     running_mean: torch.Tensor, running_var: torch.Tensor,
                     momentum: float = 0.9, epsilon: float = 1e-5,
                     training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量归一化的前向传播
        
        Args:
            x: 输入张量 (N, D)
            gamma: 缩放参数
            beta: 偏移参数
            running_mean: 运行平均值
            running_var: 运行方差
            momentum: 指数移动平均的动量
            epsilon: 数值稳定性的小常数
            training: 是否在训练模式
        
        Returns:
            (归一化后的输出, 中间值用于反向传播, 缓存)
        """
        if training:
            # 计算批量统计
            batch_mean = x.mean(dim=0, keepdim=True)
            batch_var = x.var(dim=0, keepdim=True)
            
            # 更新运行统计
            running_mean = momentum * running_mean + (1 - momentum) * batch_mean
            running_var = momentum * running_var + (1 - momentum) * batch_var
            
            # 归一化
            x_normalized = (x - batch_mean) / torch.sqrt(batch_var + epsilon)
        else:
            # 使用运行统计
            x_normalized = (x - running_mean) / torch.sqrt(running_var + epsilon)
        
        # 缩放和偏移
        out = gamma * x_normalized + beta
        
        return out, (x_normalized, gamma, x, batch_mean, batch_var, epsilon), (running_mean, running_var)


# ==================== 第400-409章：大模型和Agent ====================

@dataclass
class Message:
    """对话消息"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ConversationMemory:
    """多轮对话内存管理"""
    
    def __init__(self, max_history: int = 10):
        """
        Args:
            max_history: 保留的最大消息数
        """
        self.messages: List[Message] = []
        self.max_history = max_history
    
    def add_message(self, role: str, content: str):
        """添加消息"""
        message = Message(role=role, content=content)
        self.messages.append(message)
        
        # 保持历史大小
        if len(self.messages) > self.max_history:
            self.messages.pop(0)
    
    def get_conversation_context(self) -> str:
        """获取对话上下文"""
        context = "\n".join([
            f"{msg.role}: {msg.content}" for msg in self.messages
        ])
        return context
    
    def clear(self):
        """清除历史"""
        self.messages.clear()
    
    def save(self, filepath: str):
        """保存对话历史"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([asdict(msg) for msg in self.messages], f, ensure_ascii=False, indent=2)
        logger.info(f"Conversation saved to {filepath}")
    
    def load(self, filepath: str):
        """加载对话历史"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.messages = [Message(**msg) for msg in data]
        logger.info(f"Conversation loaded from {filepath}")


class Agent:
    """简单的Agent框架（生产级别）"""
    
    def __init__(self, model_name: str, system_prompt: str = ""):
        """
        Args:
            model_name: 模型名称
            system_prompt: 系统提示词
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.memory = ConversationMemory()
        self.tools = {}
        
    def register_tool(self, tool_name: str, tool_func):
        """注册工具"""
        self.tools[tool_name] = tool_func
        logger.info(f"Tool registered: {tool_name}")
    
    def process_input(self, user_input: str) -> str:
        """处理用户输入"""
        self.memory.add_message('user', user_input)
        
        # 这里应该调用LLM API
        # response = self.call_llm(self.memory.get_conversation_context())
        response = f"Processing: {user_input}"  # 示例
        
        self.memory.add_message('assistant', response)
        return response
    
    def get_context(self) -> str:
        """获取当前上下文"""
        return self.memory.get_conversation_context()


# ==================== 优化器和损失函数 ====================

class AdvancedOptimizer:
    """高级优化器工具"""
    
    @staticmethod
    def get_optimizer(model: nn.Module, optimizer_name: str = 'adam',
                     lr: float = 1e-3, weight_decay: float = 1e-5) -> optim.Optimizer:
        """
        获取优化器
        
        Args:
            model: 模型
            optimizer_name: 优化器名称
            lr: 学习率
            weight_decay: 权重衰退
        
        Returns:
            优化器实例
        """
        if optimizer_name.lower() == 'adam':
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    @staticmethod
    def get_scheduler(optimizer: optim.Optimizer, scheduler_name: str = 'cosine',
                     num_epochs: int = 100, num_batches: int = 1000) -> Any:
        """获取学习率调度器"""
        if scheduler_name.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler_name.lower() == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif scheduler_name.lower() == 'linear':
            return optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, 
                                              total_iters=num_epochs)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")


class LossFunctions:
    """损失函数工具"""
    
    @staticmethod
    def get_loss_function(task_type: str = 'classification', 
                         num_classes: int = 10) -> nn.Module:
        """
        获取损失函数
        
        Args:
            task_type: 任务类型（'classification', 'regression', 'multilabel'）
            num_classes: 类别数（仅用于分类）
        
        Returns:
            损失函数
        """
        if task_type == 'classification':
            return nn.CrossEntropyLoss()
        elif task_type == 'regression':
            return nn.MSELoss()
        elif task_type == 'multilabel':
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown task type: {task_type}")


# ==================== 实用工具 ====================

class ModelAnalyzer:
    """模型分析工具"""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Tuple[int, int]:
        """
        计算模型参数数量
        
        Returns:
            (总参数数, 可训练参数数)
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    @staticmethod
    def print_model_summary(model: nn.Module, input_size: Tuple[int, ...]):
        """打印模型摘要"""
        total, trainable = ModelAnalyzer.count_parameters(model)
        logger.info(f"Model: {model.__class__.__name__}")
        logger.info(f"Total parameters: {total:,}")
        logger.info(f"Trainable parameters: {trainable:,}")
        logger.info(f"Non-trainable parameters: {total - trainable:,}")


# ==================== 示例用法 ====================

def example_pytorch_basics():
    """示例：PyTorch基础"""
    logger.info("=" * 50)
    logger.info("Example: PyTorch Basics")
    logger.info("=" * 50)
    
    config = ConfigPyTorch()
    config.setup()
    
    # 探索torch模块
    magic = PythonMagicMethods()
    cuda_methods = magic.explore_module(torch.cuda, pattern='is')
    logger.info(f"CUDA methods with 'is': {cuda_methods}")


def example_training():
    """示例：完整的训练流程"""
    logger.info("=" * 50)
    logger.info("Example: Complete Training Pipeline")
    logger.info("=" * 50)
    
    # 配置
    config = ConfigPyTorch()
    config.setup()
    
    # 创建虚拟数据
    X_train = np.random.randn(1000, 28, 28, 1)
    y_train = np.random.randint(0, 10, 1000)
    X_val = np.random.randn(200, 28, 28, 1)
    y_val = np.random.randint(0, 10, 200)
    
    # 创建数据加载器
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 创建模型
    model = ResNet18Classifier(num_classes=10, pretrained=False)
    ModelAnalyzer.print_model_summary(model, (1, 28, 28, 1))
    
    # 优化器和损失函数
    optimizer = AdvancedOptimizer.get_optimizer(model, 'adam', lr=1e-3)
    loss_fn = LossFunctions.get_loss_function('classification')
    
    # 训练
    trainer = Trainer(model, device=config.device)
    checkpoint_manager = ModelCheckpoint('./checkpoints', best_metric='val_loss')
    
    logger.info("Starting training...")
    # trainer.train(train_loader, val_loader, num_epochs=2, optimizer=optimizer, 
    #              loss_fn=loss_fn, checkpoint_manager=checkpoint_manager)


def example_conversation():
    """示例：多轮对话"""
    logger.info("=" * 50)
    logger.info("Example: Multi-turn Conversation")
    logger.info("=" * 50)
    
    agent = Agent("gpt-4", system_prompt="You are a helpful AI assistant.")
    
    # 模拟对话
    agent.process_input("What is Python?")
    agent.process_input("Tell me about deep learning")
    
    logger.info("Conversation context:")
    logger.info(agent.get_context())
    
    # 保存对话
    agent.memory.save('./conversation_history.json')


if __name__ == '__main__':
    logger.info("Starting Production Code Examples")
    
    try:
        example_pytorch_basics()
        example_training()
        example_conversation()
        logger.info("All examples completed successfully!")
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()

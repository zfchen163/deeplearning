# 💼 真实场景案例 - 你会遇到的问题

**这里讲的都是从零开始的人真实会遇到的问题和解决方案。**

---

## 案例1：我有一堆图片，想分类

**真实场景**：你有100个猫的图片，200个狗的图片，想训练个模型区分。

### 步骤1：把数据整理成这样
```
data/
├── cat/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
└── dog/
    ├── 001.jpg
    ├── 002.jpg
    └── ...
```

### 步骤2：写个简单的数据加载器
```python
import os
import cv2
import numpy as np
from pathlib import Path

# 加载所有图片
def load_data():
    X, y = [], []
    labels = {'cat': 0, 'dog': 1}
    
    for label_name, label_id in labels.items():
        folder = f'data/{label_name}'
        for img_name in os.listdir(folder):
            img = cv2.imread(f'{folder}/{img_name}')
            if img is not None:
                img = cv2.resize(img, (224, 224))  # 统一大小
                X.append(img)
                y.append(label_id)
    
    return np.array(X), np.array(y)

X, y = load_data()
print(f"加载了 {len(X)} 张图片")
```

### 步骤3：用我的代码训练
```python
from production_code_examples import CustomDataset, ResNet18Classifier, Trainer, AdvancedOptimizer
from torch.utils.data import DataLoader, random_split
import torch

# 创建数据集
dataset = CustomDataset(X, y)

# 分割成训练和验证
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 创建模型
model = ResNet18Classifier(num_classes=2)

# 创建优化器
optimizer = AdvancedOptimizer.get_optimizer(model, 'adamw', lr=1e-3)

# 训练
trainer = Trainer(model, device='cuda')
trainer.train(train_loader, val_loader, num_epochs=50, optimizer=optimizer, 
              loss_fn=torch.nn.CrossEntropyLoss())
```

### 步骤4：用模型预测
```python
# 测试图片
test_img = cv2.imread('test_cat.jpg')
test_img = cv2.resize(test_img, (224, 224))
test_img = torch.FloatTensor(test_img).unsqueeze(0).to('cuda')

model.eval()
with torch.no_grad():
    output = model(test_img)
    pred = output.argmax(1)
    
print(f"预测结果: {'cat' if pred == 0 else 'dog'}")
```

**为什么这样做**：
- ✅ 数据清晰有序
- ✅ 代码简单易懂
- ✅ 能快速试验

**可能遇到的问题**：
- 图片大小不一样？ → 用 `cv2.resize` 统一
- 内存不够？ → 减小batch_size
- 精度不好？ → 增加数据或训练更久

---

## 案例2：我想用预训练模型，但不知道怎么用

**真实场景**：你听说用ImageNet预训练的模型能更快收敛，但不知道怎么用。

### 错的做法 ❌
```python
# 不要这样做，这样是重新训练
model = ResNet18Classifier(pretrained=False)
```

### 对的做法 ✅
```python
from production_code_examples import ResNet18Classifier

# 用预训练的权重（已经学过很多特征）
model = ResNet18Classifier(num_classes=2, pretrained=True)

# 然后直接训练
# 因为底层已经有了好的特征，只需要微调顶层
```

**为什么这样做**：
- 预训练模型已经学会了识别边缘、纹理等基本特征
- 你只需要让它学会区分你的数据
- 收敛快10倍，精度也更好

**进阶：只微调部分层**
```python
# 冻结底层（不训练）
for name, param in model.backbone.named_parameters():
    if 'layer3' not in name and 'layer4' not in name:
        param.requires_grad = False

# 这样只训练后两层，更快
```

---

## 案例3：我的模型过拟合了

**真实场景**：训练精度99%，测试精度50%。怎么办？

### 第1步：确认真的过拟合了
```python
# 看两条线
# 如果训练曲线持续下降，但验证曲线开始上升 → 过拟合

import matplotlib.pyplot as plt
plt.plot(train_losses, label='train')
plt.plot(val_losses, label='val')
plt.legend()
plt.show()
```

### 第2步：解决方案（按顺序试）

**方案1：加Dropout** ✅ 最简单
```yaml
# 改config_example.yaml
regularization_config:
  use_dropout: true
  dropout_rate: 0.5  # 增加到0.7试试
```

**方案2：加L2正则化**
```yaml
regularization_config:
  use_weight_decay: true
  weight_decay: 1e-4  # 从1e-5改成1e-4
```

**方案3：早停** ✅ 最实用
```yaml
early_stopping_config:
  enabled: true
  patience: 5  # 5个epoch没进步就停
```

**方案4：加数据增强** ✅ 效果最好（但需要代码改）
```python
# 在CustomDataset中加上数据增强
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.RandomRotation(10),  # 随机旋转
    transforms.RandomHorizontalFlip(),  # 随机翻转
    transforms.ColorJitter(brightness=0.2),  # 随机调整亮度
])
```

**方案5：更多数据** ✅ 终极解决
- 如果数据太少（<1000张），增加数据会有明显帮助
- 可以做数据增强、爬虫、或标注

---

## 案例4：我要参加Kaggle竞赛

**真实场景**：你想试试Kaggle的比赛，但从来没参加过。

### 步骤1：下载数据
```bash
# Kaggle提供了命令行工具
kaggle competitions download -c [竞赛名]
```

### 步骤2：快速检查数据
```python
import pandas as pd
import numpy as np

# 看看数据长啥样
train_df = pd.read_csv('train.csv')
print(train_df.head())
print(train_df.shape)
print(train_df.info())

# 看看有没有缺失值
print(train_df.isnull().sum())
```

### 步骤3：用我的代码快速建立baseline
```python
from production_code_examples import *
import torch

# 1. 读取数据
X_train = np.load('train_images.npy')
y_train = np.load('train_labels.npy')

# 2. 创建数据集和DataLoader
dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=32)

# 3. 训练
model = ResNet18Classifier(num_classes=10)  # 改成对应的类别数
optimizer = AdvancedOptimizer.get_optimizer(model, 'adamw')
trainer = Trainer(model)
trainer.train(train_loader, val_loader, num_epochs=20, optimizer=optimizer)

# 4. 预测
X_test = np.load('test_images.npy')
test_dataset = CustomDataset(X_test, np.zeros(len(X_test)))
test_loader = DataLoader(test_dataset, batch_size=32)

predictions = []
model.eval()
with torch.no_grad():
    for images, _ in test_loader:
        outputs = model(images.to('cuda'))
        preds = outputs.argmax(1).cpu().numpy()
        predictions.extend(preds)

# 5. 提交
submission = pd.DataFrame({
    'id': range(len(predictions)),
    'target': predictions
})
submission.to_csv('submission.csv', index=False)
```

**Kaggle技巧**：
1. 先建立baseline（快速跑个版本）
2. 再优化（调参、加特征、集成等）
3. 看排行榜，参考别人的思路

---

## 案例5：我训练了一个模型，要用它做生产预测

**真实场景**：你要把模型部署到服务器或手机上。

### 步骤1：保存最好的模型
```python
# 训练过程中自动保存（用我的Trainer）
# 或手动保存
torch.save(model.state_dict(), 'best_model.pt')
```

### 步骤2：部署到Flask服务器
```python
from flask import Flask, request, jsonify
import torch
from production_code_examples import ResNet18Classifier
import numpy as np
import cv2

app = Flask(__name__)

# 加载模型
model = ResNet18Classifier(num_classes=10)
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    # 获取上传的图片
    file = request.files['image']
    img = cv2.imdecode(np.fromfile(file), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    
    # 预处理
    img = torch.FloatTensor(img).unsqueeze(0).to('cuda')
    
    # 预测
    with torch.no_grad():
        output = model(img)
        pred = output.argmax(1).item()
    
    return jsonify({'prediction': int(pred)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 步骤3：用Python调用
```python
import requests
from PIL import Image

# 上传图片并获得预测
with open('test.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/predict', 
                            files={'image': f})
    print(response.json())
```

---

## 案例6：我的数据是CSV格式，不是图片

**真实场景**：你的数据是表格（CSV），比如股票数据、医疗数据等。

### 步骤1：读取数据
```python
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')

# 分离特征和标签
X = df.drop('target', axis=1).values
y = df['target'].values

print(f"特征数: {X.shape[1]}")
print(f"样本数: {X.shape[0]}")
```

### 步骤2：改数据加载器
```python
from production_code_examples import CustomDataset
from torch.utils.data import DataLoader
import torch

# 归一化（很重要！）
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 创建数据集
dataset = CustomDataset(X.astype(np.float32), y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 步骤3：改模型
```python
import torch.nn as nn

class TabularModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

model = TabularModel(input_size=X.shape[1], num_classes=2)
```

**为什么要改**：
- 表格数据不需要CNN，用全连接层就行
- 要对数据做归一化
- 网络可以更浅

---

## 案例7：我想用多GPU训练

**真实场景**：你有多张显卡，想加速训练。

### 步骤1：改代码（一行解决）
```python
import torch.nn as nn
from torch.nn.parallel import DataParallel

model = ResNet18Classifier(num_classes=10)
model = DataParallel(model)  # 就这一行

# 然后正常训练
trainer = Trainer(model, device='cuda')
```

### 步骤2：看看工作没
```bash
# 查看GPU使用情况
nvidia-smi

# 应该看到两张GPU都在工作
```

**为什么这样做**：
- batch会自动分到多张GPU
- 速度接近线性增长（2张GPU大概快2倍）
- 代码改动最小

---

## 💡 这些案例的规律

**记住这个流程，99%的深度学习问题都能解决**：

```
1. 数据准备
   ↓
2. 数据加载（可能需要改CustomDataset）
   ↓
3. 选模型（ResNet / CNN / 全连接层 等）
   ↓
4. 选优化器（一般用Adam）
   ↓
5. 训练（用我的Trainer或自己写训练循环）
   ↓
6. 评估（看精度、loss等指标）
   ↓
7. 如果不好 → 回到第1或4重新调整
   ↓
8. 部署（保存模型、可能要改成ONNX等）
```

---

**现在你应该能处理大部分实际问题了。** 🎉

*生成时间: 2026-01-27*  
*都是从零开始会遇到的真实场景*

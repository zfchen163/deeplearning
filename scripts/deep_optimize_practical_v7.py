#!/usr/bin/env python3
"""
V7版本: 深度优化 - 为每个函数添加实际应用讲解
重点:
1. 这个函数能解决什么实际问题
2. 为什么要用这个函数(不用会怎样)
3. 生活中的类比
4. 循序渐进的知识展开
5. 初中生都能看懂
"""
import json
import os
import re
from pathlib import Path

# 常见函数的实际应用讲解
FUNCTION_EXPLANATIONS = {
    "transforms.ToTensor": {
        "what": "把图片变成PyTorch能理解的数字",
        "why": """
### 🤔 为什么要用 ToTensor?

**问题: 电脑不认识图片**
- 你看到的图片是彩色的
- 但电脑只认识数字(0和1)
- 需要把图片转换成数字

**就像翻译:**
- 图片 = 中文
- Tensor = 英文
- ToTensor = 翻译器

**不用会怎样?**
❌ 神经网络无法处理图片
❌ 就像让不会中文的外国人读中文书
""",
        "example": """
### 📱 实际应用: 手机拍照识别

**场景: 用手机扫描植物识别**

1. 📸 你拍了一张花的照片
2. 🔄 ToTensor把照片变成数字
3. 🤖 AI读取这些数字
4. 🌸 识别出是"玫瑰花"

```python
# 第1步: 导入工具
from torchvision import transforms
from PIL import Image

# 第2步: 加载图片(假设你拍了一张花的照片)
img = Image.open('flower.jpg')
print(f"原始图片类型: {type(img)}")  # PIL图片

# 第3步: 转换成Tensor
to_tensor = transforms.ToTensor()
tensor_img = to_tensor(img)

print(f"转换后类型: {type(tensor_img)}")  # Tensor
print(f"数字形状: {tensor_img.shape}")    # [3, 高, 宽]
# 3 = RGB三个颜色通道(红绿蓝)

# 第4步: 现在AI可以处理了！
# tensor_img 就是一堆数字,AI能看懂
```

**理解:**
- 原始图片: 人类能看懂
- Tensor: 电脑能看懂
- ToTensor: 翻译工具
""",
        "progressive": """
### 📚 循序渐进理解

**第1层: 小学生理解**
- 把图片变成数字

**第2层: 初中生理解**
- 图片是由像素组成的
- 每个像素有颜色(RGB)
- ToTensor把每个像素的颜色变成0-1之间的数字

**第3层: 高中生理解**
- 原始图片: PIL Image对象,像素值0-255
- ToTensor做了两件事:
  1. 把像素值除以255,变成0-1
  2. 改变维度顺序: (H,W,C) → (C,H,W)
"""
    },
    
    "nn.Conv2d": {
        "what": "用小窗口扫描图片,找特征",
        "why": """
### 🤔 为什么要用 Conv2d?

**问题: 全连接层太笨了**
- 全连接层把整张图片压扁成一条线
- 丢失了空间信息(哪个像素在哪里)
- 参数太多,计算太慢

**就像:**
- ❌ 全连接层 = 把拼图打散,看不出图案
- ✅ 卷积层 = 用放大镜一块一块看,保留位置关系

**不用会怎样?**
❌ 识别图片效果差
❌ 训练速度慢
❌ 需要更多数据
""",
        "example": """
### 📱 实际应用: 人脸识别

**场景: 手机解锁(Face ID)**

```python
import torch
import torch.nn as nn

# 假设这是你的脸部照片(简化版)
face_image = torch.randn(1, 3, 224, 224)
# 1 = 1张图片
# 3 = RGB三个颜色
# 224x224 = 图片大小

# 第1步: 创建卷积层(找脸部特征)
conv = nn.Conv2d(
    in_channels=3,      # 输入RGB图片
    out_channels=32,    # 找32种不同的特征
    kernel_size=3,      # 用3x3的小窗口扫描
    padding=1           # 保持图片大小不变
)

# 第2步: 扫描图片,找特征
features = conv(face_image)
print(f"找到的特征: {features.shape}")
# 输出: [1, 32, 224, 224]
# 32个特征图,每个都是224x224

# 这32个特征可能包括:
# - 眼睛的位置
# - 鼻子的形状
# - 嘴巴的轮廓
# - 脸部的边缘
# ...等等

# 第3步: 后续层会用这些特征判断是不是你
```

**为什么有效?**
1. 卷积核像"特征检测器"
2. 每个卷积核专门找一种特征
3. 32个卷积核 = 32个检测器
4. 找到足够多的特征,就能认出你的脸
""",
        "progressive": """
### 📚 循序渐进理解

**第1层: 小学生理解**
- 用小窗口在图片上滑动
- 找图片里的特征

**第2层: 初中生理解**
- 小窗口 = 卷积核(kernel)
- 滑动 = 从左到右,从上到下扫描
- 找特征 = 计算相似度
- 例子: 找边缘、找角点、找纹理

**第3层: 高中生理解**
- 卷积核是一个小矩阵(如3x3)
- 每次滑动,卷积核和图片对应位置相乘再求和
- 多个卷积核可以找多种特征
- 参数共享: 同一个卷积核扫描整张图片

**第4层: 参数详解**
```python
nn.Conv2d(
    in_channels=3,    # 输入通道数(RGB=3)
    out_channels=32,  # 输出通道数(找32种特征)
    kernel_size=3,    # 卷积核大小(3x3)
    stride=1,         # 步长(每次移动1格)
    padding=1         # 填充(保持大小)
)
```
"""
    },
    
    "nn.ReLU": {
        "what": "把负数变成0,正数不变",
        "why": """
### 🤔 为什么要用 ReLU?

**问题: 神经网络太"线性"了**
- 没有激活函数,神经网络只能画直线
- 现实世界很复杂,不是直线能解决的

**就像:**
- ❌ 没有ReLU = 只能用直尺画图
- ✅ 有ReLU = 可以画曲线、圆圈、各种形状

**不用会怎样?**
❌ 多层神经网络等于一层(退化)
❌ 无法学习复杂模式
❌ 准确率很低
""",
        "example": """
### 📱 实际应用: 判断天气

**场景: 根据温度、湿度、风速判断会不会下雨**

```python
import torch
import torch.nn as nn

# 假设我们有3个输入: 温度、湿度、风速
# 温度=25度, 湿度=80%, 风速=5m/s
weather_data = torch.tensor([[25.0, 80.0, 5.0]])

# 第1步: 线性层(计算)
linear = nn.Linear(3, 1)
score = linear(weather_data)
print(f"计算得分: {score.item():.2f}")  # 可能是负数

# 第2步: ReLU激活(决策)
relu = nn.ReLU()
activated = relu(score)
print(f"激活后: {activated.item():.2f}")

# ReLU的作用:
# - 如果得分 < 0: 变成0(不会下雨)
# - 如果得分 > 0: 保持不变(可能下雨)

# 为什么这样有用?
# 1. 引入非线性(可以学习复杂规律)
# 2. 过滤掉不重要的信号(负数变0)
# 3. 计算速度快(比sigmoid等函数快)
```

**生活类比: 过滤器**
- 想象一个筛子
- 大于0的通过(保留)
- 小于0的被挡住(变成0)
- 就像只让"有用的信息"通过
""",
        "progressive": """
### 📚 循序渐进理解

**第1层: 小学生理解**
- 正数不变
- 负数变成0

**第2层: 初中生理解**
- ReLU = Rectified Linear Unit(修正线性单元)
- 公式: f(x) = max(0, x)
- 作用: 让神经网络能学习复杂模式

**第3层: 高中生理解**
- 为什么需要非线性?
  * 线性函数: y = ax + b(直线)
  * 多层线性 = 还是线性
  * 加入ReLU = 可以拟合任意曲线

**第4层: 对比其他激活函数**
```python
x = torch.tensor([-2, -1, 0, 1, 2])

# ReLU: 负数变0
relu = nn.ReLU()
print(f"ReLU: {relu(x)}")  # [0, 0, 0, 1, 2]

# Sigmoid: 压缩到0-1
sigmoid = nn.Sigmoid()
print(f"Sigmoid: {sigmoid(x)}")  # [0.12, 0.27, 0.5, 0.73, 0.88]

# Tanh: 压缩到-1到1
tanh = nn.Tanh()
print(f"Tanh: {tanh(x)}")  # [-0.96, -0.76, 0, 0.76, 0.96]
```

**为什么ReLU最常用?**
1. 计算简单(速度快)
2. 不会梯度消失(训练效果好)
3. 效果好(大多数情况)
"""
    },
    
    "nn.MaxPool2d": {
        "what": "把图片缩小,保留最重要的信息",
        "why": """
### 🤔 为什么要用 MaxPool2d?

**问题: 图片太大了**
- 高清图片有几百万个像素
- 计算量太大,速度慢
- 内存不够用

**就像:**
- ❌ 原图 = 看一本厚厚的书
- ✅ 池化后 = 看这本书的摘要
- 信息量减少,但重点保留

**不用会怎样?**
❌ 计算太慢
❌ 内存不够
❌ 容易过拟合(记住细节,忘记规律)
""",
        "example": """
### 📱 实际应用: 图片识别

**场景: 识别照片里是猫还是狗**

```python
import torch
import torch.nn as nn

# 假设这是一张猫的照片
cat_image = torch.randn(1, 3, 224, 224)
print(f"原始图片: {cat_image.shape}")  # [1, 3, 224, 224]

# 第1步: 卷积找特征
conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
features = conv(cat_image)
print(f"卷积后: {features.shape}")  # [1, 64, 224, 224]
# 还是很大!

# 第2步: 池化缩小
pool = nn.MaxPool2d(kernel_size=2, stride=2)
pooled = pool(features)
print(f"池化后: {pooled.shape}")  # [1, 64, 112, 112]
# 缩小一半!

# 为什么有效?
# 1. 图片缩小了: 224→112(减少75%的像素)
# 2. 重要特征保留了: 取最大值=保留最强的特征
# 3. 计算加速了: 像素少了,计算快了

# 类比: 看图片缩略图
# - 缩略图很小,但你还是能认出是猫
# - MaxPool就是生成"特征缩略图"
```

**工作原理:**
```python
# 假设有一个2x2的特征区域
feature_region = torch.tensor([
    [1.0, 3.0],
    [2.0, 4.0]
])

# MaxPool2d(2, 2)会取最大值
# 结果: 4.0(保留最强的特征)

# 为什么取最大值?
# - 最大值 = 最强的特征响应
# - 最强的特征 = 最重要的信息
# - 例如: 检测到猫耳朵的强烈信号
```
""",
        "progressive": """
### 📚 循序渐进理解

**第1层: 小学生理解**
- 把图片变小
- 保留重要信息

**第2层: 初中生理解**
- 用一个小窗口(如2x2)扫描图片
- 每个窗口取最大值
- 图片大小减半,但重要特征保留

**第3层: 高中生理解**
- MaxPool vs AvgPool:
  * MaxPool: 取最大值(保留最强特征)
  * AvgPool: 取平均值(保留整体信息)
- 常用MaxPool,因为最强特征最重要

**第4层: 参数详解**
```python
nn.MaxPool2d(
    kernel_size=2,  # 窗口大小2x2
    stride=2,       # 步长2(不重叠)
    padding=0       # 不填充
)

# 输出大小计算:
# output_size = (input_size - kernel_size) / stride + 1
# 例如: (224 - 2) / 2 + 1 = 112
```

**第5层: 为什么有效?**
1. **降维**: 减少计算量
2. **不变性**: 特征位置稍微移动,结果不变
3. **防止过拟合**: 减少参数,提高泛化能力
"""
    },
    
    "DataLoader": {
        "what": "自动分批加载数据,像传送带一样",
        "why": """
### 🤔 为什么要用 DataLoader?

**问题: 数据太多,内存装不下**
- 训练数据有几万张图片
- 一次性加载会爆内存
- 需要分批处理

**就像:**
- ❌ 一次性 = 一口气吃掉一整个蛋糕(撑死)
- ✅ DataLoader = 一口一口吃(舒服)

**不用会怎样?**
❌ 内存不够,程序崩溃
❌ 训练速度慢
❌ 无法利用多核CPU
""",
        "example": """
### 📱 实际应用: 训练图片分类模型

**场景: 训练一个识别猫狗的模型**

```python
import torch
from torch.utils.data import Dataset, DataLoader

# 假设我们有10000张猫狗图片
class CatDogDataset(Dataset):
    def __init__(self):
        # 这里只是示例,实际会加载真实图片
        self.images = torch.randn(10000, 3, 64, 64)  # 10000张图片
        self.labels = torch.randint(0, 2, (10000,))  # 0=猫, 1=狗
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# 第1步: 创建数据集
dataset = CatDogDataset()
print(f"总共有 {len(dataset)} 张图片")

# 第2步: 创建DataLoader(关键!)
dataloader = DataLoader(
    dataset,
    batch_size=32,      # 每次取32张图片
    shuffle=True,       # 打乱顺序(重要!)
    num_workers=2       # 用2个进程加载数据(加速)
)

# 第3步: 训练(DataLoader自动分批)
for epoch in range(3):  # 训练3轮
    for batch_idx, (images, labels) in enumerate(dataloader):
        # images: [32, 3, 64, 64] - 32张图片
        # labels: [32] - 32个标签
        
        # 这里进行训练...
        # model(images)
        # ...
        
        if batch_idx == 0:  # 只打印第一批
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}")
            print(f"  图片: {images.shape}")
            print(f"  标签: {labels.shape}")

# DataLoader的好处:
# 1. 自动分批: 10000张图片分成313批(每批32张)
# 2. 自动打乱: 每个epoch顺序不同(防止记住顺序)
# 3. 并行加载: 一边训练一边加载下一批(不浪费时间)
```

**类比: 餐厅上菜**
- 厨房 = Dataset(存储所有数据)
- 服务员 = DataLoader(分批送菜)
- batch_size = 每次端几盘菜
- shuffle = 随机上菜顺序
- num_workers = 几个服务员
""",
        "progressive": """
### 📚 循序渐进理解

**第1层: 小学生理解**
- 数据太多,分批处理
- 像传送带一样,一批一批送

**第2层: 初中生理解**
- Dataset: 存储所有数据
- DataLoader: 从Dataset中取数据
- batch_size: 每次取多少个
- shuffle: 是否打乱顺序

**第3层: 高中生理解**
- 为什么要shuffle?
  * 防止模型记住数据顺序
  * 提高泛化能力
  * 每个epoch看到的顺序不同

- 为什么要batch?
  * 一次处理多个样本,利用GPU并行
  * 梯度更稳定(多个样本的平均)
  * 内存可控(不会一次性加载所有数据)

**第4层: 参数详解**
```python
DataLoader(
    dataset,              # 数据集
    batch_size=32,        # 批量大小
    shuffle=True,         # 是否打乱
    num_workers=4,        # 加载数据的进程数
    pin_memory=True,      # 加速GPU传输
    drop_last=False       # 最后不足一批是否丢弃
)
```

**第5层: 性能优化**
- num_workers=0: 单进程(慢,但调试方便)
- num_workers=4: 4进程(快,但占用CPU)
- pin_memory=True: 如果用GPU,设置True加速
- persistent_workers=True: 进程不重启(更快)
"""
    }
}

def add_function_explanation(notebook_path):
    """为笔记本中的函数添加详细讲解"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        if not notebook.get('cells'):
            return False
        
        filename = os.path.basename(notebook_path)
        modified = False
        
        # 遍历所有代码单元
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            
            code = ''.join(cell.get('source', []))
            
            # 检查代码中是否包含我们要讲解的函数
            for func_name, explanation in FUNCTION_EXPLANATIONS.items():
                if func_name in code:
                    # 检查前一个cell是否已经有讲解
                    if i > 0 and notebook['cells'][i-1]['cell_type'] == 'markdown':
                        prev_content = ''.join(notebook['cells'][i-1]['source'])
                        if '🤔 为什么要用' in prev_content:
                            continue  # 已经添加过
                    
                    # 创建讲解cell
                    explanation_cell = {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [
                            f"\n## 🔍 深入理解: {func_name}\n\n",
                            f"**{explanation['what']}**\n\n",
                            explanation['why'] + "\n\n",
                            explanation['example'] + "\n\n",
                            explanation['progressive'] + "\n"
                        ]
                    }
                    
                    # 插入到代码cell之前
                    notebook['cells'].insert(i, explanation_cell)
                    modified = True
                    print(f"  ✓ {filename} - 添加 {func_name} 讲解")
                    break  # 每次只处理一个函数,避免索引混乱
        
        if modified:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, ensure_ascii=False, indent=1)
            return True
        
        return False
        
    except Exception as e:
        print(f"  ✗ {filename} - 失败: {e}")
        return False

def main():
    notebooks_dir = Path('/Users/h/practice/CV-main')
    
    print("🚀 深度优化: 添加函数实际应用讲解...\n")
    print("📝 本次优化:")
    print("   - 为每个函数添加'为什么要用'")
    print("   - 添加实际应用场景")
    print("   - 添加生活类比")
    print("   - 循序渐进展开知识点")
    print("   - 确保初中生都能看懂\n")
    
    success_count = 0
    total_count = 0
    
    for file in sorted(os.listdir(notebooks_dir)):
        if file.endswith('.ipynb') and not file.endswith('_backup.ipynb'):
            total_count += 1
            nb_path = notebooks_dir / file
            if add_function_explanation(nb_path):
                success_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ 完成! 成功优化 {success_count} 个笔记本")
    print(f"{'='*60}")
    
    print("\n🎉 现在每个函数都有:")
    print("   ✓ 为什么要用(不用会怎样)")
    print("   ✓ 实际应用场景(真实例子)")
    print("   ✓ 生活类比(容易理解)")
    print("   ✓ 循序渐进讲解(从简单到复杂)")
    print("   ✓ 初中生都能看懂!")

if __name__ == "__main__":
    main()

# 🎓 超实用快速参考 - 说人话版本

**这个文档的目标：你遇到问题时，能在2分钟内找到答案。**

---

## 💻 我要开始学了，怎么开始

### 第0步：装环境（烦人但必须得做）

```bash
# 进目录
cd /Users/h/practice/CV-main

# 装东西（这会花几分钟）
pip install -r requirements.txt
```

**卡住了？** 
- 没有pip？→ Google "pip安装"
- 网太慢？→ 换源（Google "pip换源"）
- 显卡驱动问题？→ 暂时用CPU: `pip install torch --cpu`

### 第1步：看看能干什么（3分钟）

```bash
python production_code_examples.py
```

**你会看到**：环境OK、模型能跑、代码能用

**没看到？** → 查下面的"常见问题"部分

### 第2步：开始学

**选一个符合你的**：

| 你的情况 | 怎么学 |
|--------|--------|
| 我什么都不会 | 按顺序学Notebooks 100-122 |
| 我有编程基础 | 快速过100-122，重点学200-268 |
| 我想快速做项目 | 跳到200，然后直接用我的代码 |
| 我想深入学原理 | 学完200-268再补PRODUCTION_GUIDE |

---

## 🔥 实际问题 & 解决方案

### 问题1：我的显卡没有CUDA
**症状**：运行代码时看到 "CUDA not available"

**秒杀方案**：
```python
python main.py --device cpu  # 改成CPU运行
```

**或者改代码**：
```python
config = ConfigPyTorch()
config.device = 'cpu'
```

**后果**：速度会慢10-20倍，但能跑

---

### 问题2：显存爆炸（OOM）
**症状**：看到 "CUDA out of memory"

**三个解决方案，从快到慢排列**：

1️⃣ **改批大小**（最快）
```yaml
# 改config_example.yaml
batch_size: 16  # 原来是32，改成16
```

2️⃣ **启用混合精度**（快）
```yaml
mixed_precision: true
```

3️⃣ **梯度累积**（次快）
```yaml
accumulation_steps: 2  # 内存减半
```

**还是爆炸？** → 改成CPU，或者等着买更好的显卡😂

---

### 问题3：我不知道怎么调参数
**症状**：改完代码，模型精度还是很差

**通用调参顺序**：

```
1. 先检查数据预处理对不对
   ↓
2. 改学习率（一般从1e-3改起）
   ↓
3. 改batch_size（一般16或32）
   ↓
4. 改模型架构（换个复杂点的）
   ↓
5. 加数据增强
   ↓
6. 如果还是差，那可能是数据本身的问题
```

**快速改学习率**：
```yaml
training_config:
  learning_rate: 0.001  # 改这个
```

---

### 问题4：训练跑不动，太慢了
**症状**：一个epoch要20分钟，疯了

**可能的原因和解决**：

| 症状 | 原因 | 解决办法 |
|------|------|---------|
| GPU占用率不高 | 数据加载慢 | `num_workers: 4` 改大点 |
| 内存满了 | batch太大或模型太大 | 减少batch_size |
| CPU占用率很高 | 数据预处理在CPU | 改用GPU处理，或简化预处理 |
| 就是慢 | 无法避免 | 用GPU或等待 |

**最直接的检查**：
```bash
# 看GPU情况
nvidia-smi

# 看进程
top  # 或 htop
```

---

### 问题5：模型精度上不去
**症状**：训练了很久，精度就是卡在70%上下

**通用排查顺序**：

1️⃣ **数据预处理对不对？**
```python
# 看看你的数据长啥样
print(X[0])  # 看几个样本
print(y[0])
```

2️⃣ **标签对不对？**
```python
print(np.unique(y))  # 看看有多少个类别
print(len(np.unique(y)))  # 跟你的num_classes一样吗
```

3️⃣ **模型参数对不对？**
```python
# 看看模型大小
from production_code_examples import ModelAnalyzer
ModelAnalyzer.print_model_summary(model, input_size=(1, 224, 224, 3))
```

4️⃣ **其他的一般不用改了**

---

### 问题6：怎么保存和加载我训练好的模型
**方法1：简单方法**
```python
# 保存
torch.save(model.state_dict(), 'my_model.pt')

# 加载
model.load_state_dict(torch.load('my_model.pt'))
```

**方法2：完整方法**（我推荐的）
```python
from production_code_examples import ModelCheckpoint

# 训练时自动保存
checkpoint_manager = ModelCheckpoint('./checkpoints', best_metric='val_loss')
checkpoint_manager.save_model(model, optimizer, epoch, metrics)

# 恢复训练
checkpoint_manager.load_model(model, optimizer, 'checkpoints/best_model.pt')
```

---

### 问题7：我的数据不是标准的ImageNet格式
**症状**：你的数据可能是：
- 一堆.npy文件
- Excel表格里的数据
- 乱七八糟的文件夹

**解决方案**：改我的 `CustomDataset` 类

```python
from production_code_examples import CustomDataset

class MyCustomDataset(CustomDataset):
    def __getitem__(self, idx):
        # 你自己的加载逻辑
        # 比如从Excel读，或从.npy读
        # 然后return样本和标签就行
        pass

# 然后用你的Dataset代替原来的
dataset = MyCustomDataset(my_data, my_labels)
```

**最简单的方式**：
```python
# 不管你的数据咋来的，最后转成numpy
X = your_load_function()  # 返回 (N, H, W, C)
y = your_load_labels()     # 返回 (N,)

from production_code_examples import CustomDataset
dataset = CustomDataset(X, y)
```

---

## 📊 代码速查表

### 我要创建一个模型

```python
from production_code_examples import ResNet18Classifier

# 最简单
model = ResNet18Classifier(num_classes=10)

# 改参数
model = ResNet18Classifier(
    num_classes=1000,     # 改类别数
    pretrained=False      # 不用预训练权重
)
```

### 我要选优化器

```python
from production_code_examples import AdvancedOptimizer

# Adam（推荐，一般不会错）
optimizer = AdvancedOptimizer.get_optimizer(model, 'adamw', lr=1e-3)

# SGD（老方法，有时候更稳定）
optimizer = AdvancedOptimizer.get_optimizer(model, 'sgd', lr=1e-2)

# RMSprop（中等选择）
optimizer = AdvancedOptimizer.get_optimizer(model, 'rmsprop', lr=1e-3)
```

**什么时候用哪个**：
- 默认用Adam → 快、稳定、不用调参
- 想要更好效果 → 用SGD，但需要多调参
- 中间方案 → RMSprop

### 我要启用自动混合精度（省显存）

```python
config = ConfigPyTorch()
config.training_config.mixed_precision = True
```

### 我要调学习率

```yaml
# config_example.yaml
training_config:
  learning_rate: 0.001      # 改这个
  scheduler: cosine         # 学习率调度（会逐步降低）
```

### 我要加早停（防止过拟合）

```yaml
early_stopping_config:
  enabled: true
  patience: 10              # 10个epoch没有进步就停止
```

---

## 📈 调参速查表

| 超参数 | 默认值 | 太大会怎样 | 太小会怎样 |
|--------|--------|----------|----------|
| batch_size | 32 | 显存爆炸 | 精度会差 |
| learning_rate | 0.001 | 精度差/爆炸 | 收敛太慢 |
| num_epochs | 100 | 过拟合 | 精度差 |
| dropout_rate | 0.5 | 欠拟合 | 过拟合 |
| weight_decay | 1e-5 | 欠拟合 | 正常 |

**调参金律**：
1. 先用默认值跑一个epoch，看能不能跑
2. 调batch_size让显存用到70-80%
3. 调学习率（看loss曲线，应该要下降）
4. 其他的按需调

---

## 🆘 最常见的报错信息

### 错误：`ModuleNotFoundError: No module named 'torch'`
**原因**：没装PyTorch

**解决**：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### 错误：`RuntimeError: CUDA out of memory`
**原因**：显存不够

**解决**：
```yaml
batch_size: 16  # 改小
mixed_precision: true
```

---

### 错误：`FileNotFoundError: [Errno 2] No such file or directory: ...`
**原因**：路径错了

**解决**：
```python
import os
print(os.getcwd())  # 看看你现在在哪
# 确保数据文件在这个目录下
```

---

### 错误：`ValueError: expected 3D (unbatched) or 4D (batched) input to conv2d`
**原因**：图片维度不对

**解决**：
```python
# 确保你的图片是 (N, C, H, W) 的格式
# N = 数量，C = 通道（RGB=3），H = 高，W = 宽
print(X.shape)  # 应该是 (1000, 3, 224, 224) 这样的
```

---

## 🎯 我想做什么 → 看哪个文件

| 我想... | 看哪个文件 |
|--------|-----------|
| 快速上手 | `00_START_HERE_v2.md` |
| 快速查问题 | 这个文档 |
| 写代码 | `production_code_examples.py` |
| 启动项目 | 运行 `python main.py` |
| 改参数 | 编辑 `config_example.yaml` |
| 深入理解 | `PRODUCTION_GUIDE.md` |
| 学原理 | Notebooks 100-268 |

---

## 📚 学习顺序（别乱学）

**第1周**：
1. 学Notebooks 100-107（数据处理基础）
2. 跑 `python production_code_examples.py`
3. 改参数跑一遍

**第2-3周**：
1. 学Notebooks 108-122（模型和训练）
2. 用我的代码训练一个小模型
3. 自己改数据和参数

**第4-8周**：
1. 学Notebooks 200-226（深度学习和CNN）
2. 参考我的代码，做个小项目
3. 上传到Kaggle试试

**之后**：
1. 学吴恩达或更高级内容
2. 尝试自己手写一些模块
3. 做更大的项目

---

## ⚡ 快速命令速查

```bash
# 查看GPU情况
nvidia-smi

# 运行我的示例代码
python production_code_examples.py

# 启动训练
python main.py --config config_example.yaml --mode train --device cuda

# 用CPU（如果没GPU或显存不够）
python main.py --config config_example.yaml --mode train --device cpu

# 看TensorBoard的训练曲线
tensorboard --logdir ./logs
```

---

## 🎁 最后的话

**记住这几个原则**：

1️⃣ **有现成的就别自己写**
   - 我的代码可以复制用
   
2️⃣ **遇到问题先查这个文档**
   - 99%的问题都在这儿

3️⃣ **改参数前先理解数据**
   - 打印一下 `X.shape`，`y.unique()` 等

4️⃣ **model not work ≠ 代码有bug**
   - 可能是超参数、数据、或其他
   - 改改参数试试

5️⃣ **一步一步来，别急**
   - 先跑通最简单的版本
   - 然后再加功能

---

**好了，开始吧！** 🚀

*生成时间: 2026-01-27*  
*为零基础的人写的，真实解决的都是常见问题*

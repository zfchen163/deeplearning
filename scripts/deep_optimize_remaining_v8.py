
import json
import os
from pathlib import Path

# Mapping of filename keywords to specific, vivid analogies
CONCEPT_MAP = {
    # --- Series 100 & 200: Core Layers & Concepts ---
    "110_卷积层": {
        "title": "卷积层: 寻找特征的手电筒",
        "analogy_title": "🔦 扫描与匹配",
        "analogy_content": """
### 🧪 想象这样一个场景
你在玩"找不同"的游戏，或者在茫茫人海中找一个戴红帽子的人。

你的眼睛就像一个**手电筒 (卷积核/过滤器)**：
1. 你不会一眼看完整个画面，而是**一块一块地扫描**。
2. 当手电筒照到的地方（局部区域）和你要找的目标（特征）很像时，你的大脑就会"叮"一声（激活）。

**卷积 (Convolution)** 就是计算机用无数个小手电筒在图片上扫描，寻找各种特征（线条、圆圈、眼睛、猫耳朵...）。
""",
        "problem": "图片太大了（几百万个像素），如果把每个像素都当成独立的信息，计算机处理不过来，而且容易忽略局部关系。",
        "solution": "使用卷积核提取局部特征，大大减少了参数量，同时保留了空间结构。"
    },
    "216_卷积层": { # Same as above, just mapping 216
        "title": "卷积层: 寻找特征的手电筒",
        "analogy_title": "🔦 扫描与匹配",
        "analogy_content": """
### 🧪 想象这样一个场景
你在玩"找不同"的游戏，或者在茫茫人海中找一个戴红帽子的人。

你的眼睛就像一个**手电筒 (卷积核/过滤器)**：
1. 你不会一眼看完整个画面，而是**一块一块地扫描**。
2. 当手电筒照到的地方（局部区域）和你要找的目标（特征）很像时，你的大脑就会"叮"一声（激活）。

**卷积 (Convolution)** 就是计算机用无数个小手电筒在图片上扫描，寻找各种特征（线条、圆圈、眼睛、猫耳朵...）。
""",
        "problem": "图片太大了（几百万个像素），如果把每个像素都当成独立的信息，计算机处理不过来，而且容易忽略局部关系。",
        "solution": "使用卷积核提取局部特征，大大减少了参数量，同时保留了空间结构。"
    },
    "217_卷积层里的填充和步幅": {
        "title": "填充与步幅: 相框与大长腿",
        "analogy_title": "🖼️ 边缘怎么处理？走多快？",
        "analogy_content": """
### 🧪 想象这样一个场景
你拿着手电筒扫描一张画。

1. **填充 (Padding)**:
   - 画的边缘很难扫描到。
   - **解决**: 给画加一圈白色的**相框**。这样手电筒的中心就能照到画的最边缘了。

2. **步幅 (Stride)**:
   - 如果你时间紧，不想一步一步挪。
   - **解决**: 迈开**大长腿**，一次挪两步（Stride=2）。
   - **结果**: 扫描速度变快了，但输出的图片变小了。
""",
        "problem": "卷积操作会让图片越来越小（边缘丢失），而且有时候我们希望快速降低图片尺寸。",
        "solution": "使用填充(Padding)保持尺寸，使用步幅(Stride)调节降采样率。"
    },
    "218_卷积层里的多输入多输出通道": {
        "title": "多通道: 彩色玻璃叠叠乐",
        "analogy_title": "🌈 RGB与特征叠加",
        "analogy_content": """
### 🧪 想象这样一个场景
图片不仅仅是黑白的。
彩色图片有 **红(R)、绿(G)、蓝(B)** 三层，就像三块彩色玻璃叠在一起。

**多输入通道**:
我们的手电筒（卷积核）也得是三层的！第一层看红，第二层看绿，第三层看蓝。最后把看到的结果加起来。

**多输出通道**:
为了看清图片的方方面面，我们不能只用一个手电筒。
我们要用 **64个不同的手电筒**（一组卷积核），有的找线条，有的找颜色，有的找形状。
最后输出 64 张特征图。
""",
        "problem": "单层灰度图无法表达丰富的色彩和纹理信息。",
        "solution": "增加通道数(Channels)，让神经网络在每一层都能学习到多种不同的特征模式。"
    },
    "219_池化层": {
        "title": "池化层: 选代表",
        "analogy_title": "🗳️ 谁最强谁上",
        "analogy_content": """
### 🧪 想象这样一个场景
你有一张很高清的照片，但你只想知道"照片里有没有猫"，不需要知道猫毛有几根。
为了减少计算量，我们需要把图片**变小（压缩）**。

**最大池化 (Max Pooling)**:
- 把图片分成 2x2 的小方块。
- 在每个方块里，**选一个数值最大（最强）的**作为代表。
- 其他弱的都丢掉。

就像全班选班长，选最优秀的那个代表全班。图片变小了，但最关键的特征（猫的耳朵）留下了。
""",
        "problem": "特征图太大，计算量大，且容易过拟合（太关注细节）。",
        "solution": "通过池化层降低分辨率，提取主要特征，提高模型的鲁棒性。"
    },
    "112_非线性激活": {
        "title": "激活函数: 神经元的开关",
        "analogy_title": "⚡️ 只有超过阈值才放行",
        "analogy_content": """
### 🧪 想象这样一个场景
你的大脑神经元不是什么信号都传的。
只有当刺激足够强（比如被针扎了一下），神经元才会**兴奋（激活）**。

**ReLU 激活函数**:
- 就像一个单向闸门。
- **正数**: "请进！"（保持原样）
- **负数**: "禁止通行！"（变成0）

如果没有这个非线性的开关，神经网络就算叠100层，也和1层没有区别（都是线性变换），变不出花样来。
""",
        "problem": "线性模型无法拟合复杂的曲线和现实世界的问题。",
        "solution": "引入非线性激活函数（如ReLU, Sigmoid），赋予神经网络强大的拟合能力。"
    },
    "113_线性层及其他层": {
        "title": "线性层: 全连接的社交网络",
        "analogy_title": "🤝 每个人都认识每个人",
        "analogy_content": """
### 🧪 想象这样一个场景
**线性层 (Linear Layer)** 也叫 **全连接层 (Fully Connected Layer)**。

想象两个班级联谊：
- A班的**每一个**同学，都和B班的**每一个**同学握了手。
- 没有任何遗漏。

这就是"全连接"。所有的信息都混合在了一起，这是神经网络最基础的变换方式。
""",
        "problem": "如何将上一层的所有特征综合起来进行最终的决策（比如分类）？",
        "solution": "使用全连接层，通过矩阵乘法将输入特征映射到输出空间。"
    },
    "267_优化算法": {
        "title": "优化算法: 下山指南",
        "analogy_title": "🏔️ 寻找最低谷",
        "analogy_content": """
### 🧪 想象这样一个场景
你被困在了一座漆黑的高山上（高Loss），你的目标是下到最低的山谷里（低Loss）。

你看不见路，只能用脚试探。
- **SGD (随机梯度下降)**: 像个醉汉，跌跌撞撞，但这步往左，下步往右，虽然慢但也能下山。
- **Momentum (动量)**: 像个滑雪者，利用惯性，冲过小坑，加速下山。
- **Adam**: 像个专业的登山向导，不仅利用惯性，还懂得根据地形调整步子大小（自适应学习率）。
""",
        "problem": "如何快速、稳定地找到损失函数的最小值？",
        "solution": "使用改进的优化算法（如Adam），结合动量和自适应学习率，加速收敛。"
    },
    "315_课程2_第2周_优化算法": { # Mapping similar file
         "title": "优化算法: 下山指南",
         "analogy_title": "🏔️ 寻找最低谷",
         "analogy_content": """
### 🧪 想象这样一个场景
你被困在了一座漆黑的高山上（高Loss），你的目标是下到最低的山谷里（低Loss）。

你看不见路，只能用脚试探。
- **SGD (随机梯度下降)**: 像个醉汉，跌跌撞撞，但这步往左，下步往右，虽然慢但也能下山。
- **Momentum (动量)**: 像个滑雪者，利用惯性，冲过小坑，加速下山。
- **Adam**: 像个专业的登山向导，不仅利用惯性，还懂得根据地形调整步子大小（自适应学习率）。
""",
        "problem": "如何快速、稳定地找到损失函数的最小值？",
        "solution": "使用改进的优化算法（如Adam），结合动量和自适应学习率，加速收敛。"
    },
    
    # --- Series 400: LLM & API ---
    "401_核心概念与流程": {
        "title": "大模型核心: 鹦鹉还是天才？",
        "analogy_title": "🦜 下一个词猜猜看",
        "analogy_content": """
### 🧪 想象这样一个场景
ChatGPT 到底是怎么工作的？

它其实在玩一个超级复杂的 **"成语接龙"** 游戏。
你给它："床前明月__"
它接："光"。

它读过人类历史上几乎所有的书。当你问它问题时，它不是在"思考"，而是在根据概率**预测下一个字是什么**。
但是，当书读得足够多，量变引起质变，它似乎就有了"理解"和"逻辑"。
""",
        "problem": "机器如何理解和生成人类语言？",
        "solution": "通过在大规模文本上进行预训练，学习语言的概率分布（Next Token Prediction）。"
    },
    "408_API基本参数": {
        "title": "API参数: 调教机器人的旋钮",
        "analogy_title": "🎛️ 严谨 vs 奔放",
        "analogy_content": """
### 🧪 想象这样一个场景
你面前有一个控制大模型的控制台，上面有很多旋钮。

最重要的是 **Temperature (温度)**：
- **温度 = 0 (严谨模式)**: 机器人变成老学究。每次问一样的问题，答案都一模一样。适合做数学题、写代码。
- **温度 = 1 (奔放模式)**: 机器人变成浪漫诗人。每次回答都不一样，充满想象力，但有时会胡说八道。适合写小说、头脑风暴。

学会调节这个旋钮，才能让AI按你的意愿工作。
""",
        "problem": "同一个模型，如何适应不同的任务需求（有的需要准确，有的需要创意）？",
        "solution": "通过调整Temperature、Top-p等参数，控制模型输出的随机性和多样性。"
    },
    "409_多轮对话": {
        "title": "多轮对话: 只有7秒记忆？",
        "analogy_title": "🧠 给AI装个记事本",
        "analogy_content": """
### 🧪 想象这样一个场景
默认情况下，API 是**没有记忆**的。
- 你问："我也一样。"
- AI懵了："你也一样什么？"

因为它不记得上一句你说过"我喜欢吃苹果"。

**多轮对话** 就是每次聊天时，把你之前的聊天记录（Context）**打包一起发给AI**。
"（之前他说喜欢苹果），我现在说：我也一样。"
这样AI就听懂了。
""",
        "problem": "HTTP请求是无状态的，模型默认无法记住之前的对话。",
        "solution": "将历史对话记录作为一个列表（Messages List），每次请求都完整地发给模型。"
    },
    
    # --- Series 200 & Others: Practical & Quiz ---
    "213_Kaggle房价预测": {
        "title": "房价预测: AI房产中介",
        "analogy_title": "🏠 这房子值多少钱？",
        "analogy_content": """
### 🧪 想象这样一个场景
你是个新手房产中介，你要给一套房子估价。

你会看：
- 面积多大？ (特征1)
- 几室几厅？ (特征2)
- 哪年建的？ (特征3)
- 离地铁近吗？ (特征4)

**房价预测 (回归问题)** 就是让AI通过学习几千套房子的成交记录，总结出一套"估价公式"。
下次来一套新房子，输进去这些特征，价格就算出来了。
""",
        "problem": "如何根据多个特征预测一个连续的数值（如价格、温度、销量）？",
        "solution": "使用线性回归或集成模型（如XGBoost），学习特征与目标值之间的映射关系。"
    },
    "122_查看开源项目": {
        "title": "开源项目: 偷师学艺",
        "analogy_title": "👨‍🍳 看看米其林大厨怎么做菜",
        "analogy_content": """
### 🧪 想象这样一个场景
你想成为顶级大厨。光看菜谱（理论）是不够的。
最快的方法是**直接去米其林大厨的厨房（GitHub）参观**。

看他是怎么：
- 组织食材的（项目结构）
- 怎么切菜的（代码规范）
- 怎么处理突发情况的（错误处理）

**查看开源项目** 就是去阅读高手的代码。刚开始会很难，但这是从"学徒"变成"大厨"的必经之路。
""",
        "problem": "教程里的代码通常太简单，无法应对复杂的真实项目。",
        "solution": "阅读优秀的开源项目源码，学习工程化结构、设计模式和最佳实践。"
    },
    "215_使用购买GPU": {
        "title": "云端GPU: 租辆法拉利",
        "analogy_title": "🏎️ 不买车也能飙车",
        "analogy_content": """
### 🧪 想象这样一个场景
你想体验开法拉利的感觉（训练大模型）。
但法拉利太贵了，买不起（高端显卡几万一块）。

**云端GPU (Colab/Kaggle/AutoDL)** 就像**租车公司**。
- 你按小时付租金（甚至免费）。
- 远程连上去开几个小时。
- 跑完比赛（训练完），把车还回去。

既省钱，又能用上最顶级的设备。
""",
        "problem": "深度学习对硬件要求极高，个人电脑往往跑不动。",
        "solution": "利用云计算平台，按需租用高性能GPU，低成本完成高算力任务。"
    },
    
    # --- Generic fallback for Exam/Quiz files ---
    "DEFAULT_QUIZ": {
        "title": "测验题: 驾照科目一",
        "analogy_title": "📝 检验你的车技",
        "analogy_content": """
### 🧪 想象这样一个场景
你已经学了交通规则（理论课）。
现在要进行**科目一考试**。

这不仅仅是为了分数，而是为了确认：
- 真的懂了吗？
- 遇到红灯知道停吗？

**测验题** 帮你查漏补缺。做错了没关系，那是为了让你上路（实战）时不翻车。
""",
        "problem": "光听课容易产生\"我懂了\"的错觉。",
        "solution": "通过做题强制回顾知识点，发现盲区，巩固记忆。"
    }
}

# Standard template for High School Friendly intro
TEMPLATE = """
# {title}

**分类:** {category}

**{analogy_title}**

---

{analogy_content}

### 🎯 为什么需要这个技术?

**问题:** {problem}

**解决:** {solution}

### 📚 循序渐进学习

**第一步: 理解问题** (你现在在这里)
- 为什么需要这个技术?
- 它解决什么问题?

**第二步: 学习原理** (接下来)
- 这个技术如何工作?
- 核心思想是什么?

**第三步: 实际应用** (最后)
- 如何应用到实际项目?
- 如何解决实际问题?
"""

def optimize_notebook(file_path):
    # Determine filename key
    filename = os.path.basename(file_path).replace('.ipynb', '')
    
    # Logic to find config
    config = None
    
    # 1. Exact or partial match in CONCEPT_MAP
    for key in CONCEPT_MAP:
        if key in filename:
            config = CONCEPT_MAP[key]
            break
            
    # 2. If no match, check if it's a Quiz/Homework file
    if not config:
        if "测验" in filename or "作业" in filename or "题" in filename:
            config = CONCEPT_MAP["DEFAULT_QUIZ"]
            # Customize title for quiz
            config['title'] = filename.split('_')[-1] if '_' in filename else filename
        else:
            print(f"Skipping {filename} - No specific or quiz config found.")
            return

    print(f"Processing {file_path} with config: {config['title']}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Determine category
    category = "深度学习基础"
    if "40" in filename: category = "大模型 (LLM)"
    elif "3" in filename: category = "吴恩达深度学习课程"
    elif "2" in filename: category = "深度学习原理"
    elif "1" in filename: category = "PyTorch工具箱"

    # Create header
    new_header_source = TEMPLATE.format(
        title=filename.split('_', 1)[1] if '_' in filename else filename,
        category=category,
        analogy_title=config['analogy_title'],
        analogy_content=config['analogy_content'].strip(),
        problem=config['problem'],
        solution=config['solution']
    )

    new_header_lines = [line + '\n' for line in new_header_source.split('\n')]
    if new_header_lines and new_header_lines[-1] == '\n':
        new_header_lines.pop()

    # Update Notebook
    # 1. Header (First Cell)
    if notebook['cells'] and notebook['cells'][0]['cell_type'] == 'markdown':
        notebook['cells'][0]['source'] = new_header_lines
    else:
        notebook['cells'].insert(0, {
            "cell_type": "markdown",
            "metadata": {},
            "source": new_header_lines
        })
        
    # 2. Beginner Tips (Second Cell) - Ensure it exists
    # We assume if it's there, it's fine. If not, we add it.
    # But for "Generic Header" files, the "Beginner Tips" might already be there (as we found out).
    # So we just leave the Beginner Tips alone if it exists.
    
    # Save back
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"✅ Optimized {filename}")

def main():
    base_dir = Path("/Users/h/practice/CV-main")
    
    # Find all generic files again (or just iterate all and check match)
    # We will iterate all files, and if they match our new CONCEPT_MAP, we update them.
    # This ensures we overwrite the "Generic Header" with the "Specific Header".
    
    all_notebooks = list(base_dir.glob("**/*.ipynb"))
    
    for file_path in all_notebooks:
        optimize_notebook(str(file_path))

if __name__ == "__main__":
    main()

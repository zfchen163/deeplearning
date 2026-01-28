#!/usr/bin/env python3
"""
批量优化所有Jupyter笔记本,使其更适合高中生学习
"""
import json
import os
import re
from pathlib import Path

# 笔记本分类和优化规则
NOTEBOOK_CATEGORIES = {
    "基础入门": {
        "keywords": ["配置", "安装", "Python", "Pytorch", "START"],
        "intro": "🚀 从零开始,搭建你的深度学习环境"
    },
    "数据处理": {
        "keywords": ["数据", "Dataloader", "Transforms", "预处理", "增广"],
        "intro": "📊 数据是AI的燃料,学会处理数据是第一步"
    },
    "神经网络基础": {
        "keywords": ["感知机", "线性", "激活", "损失", "优化器", "反向传播"],
        "intro": "🧠 理解神经网络的基本组件"
    },
    "卷积神经网络": {
        "keywords": ["卷积", "池化", "LeNet", "AlexNet", "VGG", "ResNet", "GoogLeNet"],
        "intro": "🖼️ 让计算机\"看懂\"图片的秘密武器"
    },
    "循环神经网络": {
        "keywords": ["RNN", "LSTM", "GRU", "序列", "循环"],
        "intro": "🔄 处理时间序列和文本的神经网络"
    },
    "注意力机制": {
        "keywords": ["注意力", "Transformer", "BERT", "seq2seq"],
        "intro": "👀 让AI学会\"关注重点\""
    },
    "计算机视觉": {
        "keywords": ["检测", "分割", "识别", "风格迁移", "目标检测"],
        "intro": "👁️ 图像识别、物体检测等视觉任务"
    },
    "实战项目": {
        "keywords": ["Kaggle", "竞赛", "实战", "项目"],
        "intro": "💪 真实项目实战,检验学习成果"
    },
    "高级主题": {
        "keywords": ["分布式", "GPU", "TPU", "微调", "RAG", "大模型"],
        "intro": "🚀 进阶技术和前沿应用"
    }
}

def categorize_notebook(filename):
    """根据文件名判断笔记本类别"""
    filename_lower = filename.lower()
    
    for category, info in NOTEBOOK_CATEGORIES.items():
        for keyword in info["keywords"]:
            if keyword.lower() in filename_lower:
                return category
    
    return "其他"

def extract_title_from_filename(filename):
    """从文件名提取标题"""
    # 移除.ipynb后缀
    name = filename.replace('.ipynb', '')
    # 移除数字前缀(如 109_)
    name = re.sub(r'^\d+_', '', name)
    return name

def add_friendly_intro(notebook_path):
    """为笔记本添加友好的引言"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        if not notebook.get('cells'):
            return False
        
        filename = os.path.basename(notebook_path)
        title = extract_title_from_filename(filename)
        category = categorize_notebook(filename)
        category_intro = NOTEBOOK_CATEGORIES.get(category, {}).get("intro", "")
        
        # 检查第一个cell是否已经是友好的引言
        first_cell = notebook['cells'][0]
        if first_cell['cell_type'] == 'markdown':
            content = ''.join(first_cell['source'])
            if '🎯' in content or '开始之前' in content:
                print(f"  ✓ {filename} 已经优化过")
                return False
        
        # 创建新的引言cell
        intro_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {title}\n",
                "\n",
                f"**分类:** {category}\n",
                "\n",
                f"**简介:** {category_intro}\n",
                "\n",
                "---\n",
                "\n",
                "## 🎯 学习目标\n",
                "\n",
                "通过本节课,你将学会:\n",
                "- 理解核心概念和原理\n",
                "- 掌握实际代码实现\n",
                "- 能够应用到实际项目中\n",
                "\n",
                "## 💡 学习建议\n",
                "\n",
                "1. **先看懂原理** - 不要急着运行代码\n",
                "2. **动手实践** - 每个代码块都运行一遍\n",
                "3. **修改参数** - 试试改变参数会发生什么\n",
                "4. **做笔记** - 记录你的理解和疑问\n",
                "\n",
                "---\n",
                "\n"
            ]
        }
        
        # 在开头插入引言
        notebook['cells'].insert(0, intro_cell)
        
        # 在结尾添加总结(如果没有的话)
        last_cell = notebook['cells'][-1]
        last_content = ''.join(last_cell['source']) if last_cell.get('source') else ''
        
        if '总结' not in last_content and '小结' not in last_content:
            summary_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## 📚 本节小结\n",
                    "\n",
                    "恭喜你完成了本节学习!让我们回顾一下:\n",
                    "\n",
                    "### ✅ 你学到了什么?\n",
                    "- 请在这里写下你的收获...\n",
                    "\n",
                    "### 🤔 还有疑问?\n",
                    "- 请记录下你不理解的地方...\n",
                    "\n",
                    "### 🚀 下一步\n",
                    "- 继续学习相关主题\n",
                    "- 尝试做一些练习题\n",
                    "- 应用到实际项目中\n",
                    "\n",
                    "---\n",
                    "\n",
                    "**记住:** 学习是一个循序渐进的过程,不要着急,慢慢来! 💪\n"
                ]
            }
            notebook['cells'].append(summary_cell)
        
        # 保存修改
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, ensure_ascii=False, indent=1)
        
        print(f"  ✓ {filename} 优化完成")
        return True
        
    except Exception as e:
        print(f"  ✗ {filename} 优化失败: {e}")
        return False

def generate_index(notebooks_dir):
    """生成课程索引"""
    notebooks = []
    
    for file in sorted(os.listdir(notebooks_dir)):
        if file.endswith('.ipynb') and not file.endswith('_backup.ipynb'):
            category = categorize_notebook(file)
            title = extract_title_from_filename(file)
            
            # 提取数字前缀作为顺序
            match = re.match(r'^(\d+)_', file)
            order = int(match.group(1)) if match else 999
            
            notebooks.append({
                "filename": file,
                "title": title,
                "category": category,
                "order": order
            })
    
    # 按类别和顺序分组
    categorized = {}
    for nb in notebooks:
        cat = nb['category']
        if cat not in categorized:
            categorized[cat] = []
        categorized[cat].append(nb)
    
    # 生成Markdown索引
    index_md = "# 深度学习课程索引\n\n"
    index_md += "## 📖 课程大纲\n\n"
    
    for category in NOTEBOOK_CATEGORIES.keys():
        if category in categorized:
            index_md += f"\n### {category}\n\n"
            index_md += f"*{NOTEBOOK_CATEGORIES[category]['intro']}*\n\n"
            
            for nb in sorted(categorized[category], key=lambda x: x['order']):
                index_md += f"- [{nb['title']}]({nb['filename']})\n"
    
    # 其他类别
    if "其他" in categorized:
        index_md += f"\n### 其他\n\n"
        for nb in sorted(categorized["其他"], key=lambda x: x['order']):
            index_md += f"- [{nb['title']}]({nb['filename']})\n"
    
    return index_md, categorized

def main():
    """主函数"""
    notebooks_dir = Path('/Users/h/practice/CV-main')
    
    print("🚀 开始批量优化笔记本...\n")
    
    # 获取所有笔记本文件
    notebook_files = [f for f in os.listdir(notebooks_dir) 
                     if f.endswith('.ipynb') and not f.endswith('_backup.ipynb')]
    
    print(f"找到 {len(notebook_files)} 个笔记本文件\n")
    
    # 优化每个笔记本
    success_count = 0
    for notebook_file in sorted(notebook_files):
        notebook_path = notebooks_dir / notebook_file
        if add_friendly_intro(notebook_path):
            success_count += 1
    
    print(f"\n✅ 优化完成! 成功优化 {success_count} 个笔记本")
    
    # 生成索引
    print("\n📚 生成课程索引...")
    index_md, categorized = generate_index(notebooks_dir)
    
    index_path = notebooks_dir / "COURSE_INDEX.md"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_md)
    
    print(f"✅ 索引已生成: {index_path}")
    
    # 生成JSON格式的索引(供前端使用)
    index_json = {
        "categories": list(NOTEBOOK_CATEGORIES.keys()),
        "notebooks": categorized
    }
    
    json_path = notebooks_dir / "course_index.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(index_json, f, ensure_ascii=False, indent=2)
    
    print(f"✅ JSON索引已生成: {json_path}")
    
    print("\n🎉 所有任务完成!")

if __name__ == "__main__":
    main()

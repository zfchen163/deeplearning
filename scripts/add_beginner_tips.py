#!/usr/bin/env python3
"""
简化版: 为所有笔记本添加入门友好的提示
"""
import json
import os
from pathlib import Path

def add_beginner_tips(notebook_path):
    """为笔记本添加入门提示"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        if not notebook.get('cells') or len(notebook['cells']) < 2:
            return False
        
        filename = os.path.basename(notebook_path)
        
        # 检查是否已经添加过
        if len(notebook['cells']) > 1:
            second_cell = notebook['cells'][1]
            if second_cell['cell_type'] == 'markdown':
                content = ''.join(second_cell['source'])
                if '🔰 新手必看' in content:
                    return False
        
        # 创建通用的入门提示
        tips_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": """
## 🔰 新手必看

**第一次学？这些提示能帮到你！**

### 💡 学习建议

1. **不要急** - 慢慢看，不懂的多看几遍
2. **动手做** - 每个代码都运行一遍
3. **改参数** - 试着改改数字，看看会怎样
4. **记笔记** - 把重点记下来

### ⚠️ 常见问题

**Q: 代码报错怎么办？**
- 先看错误提示（红色的那行）
- 检查是否有拼写错误
- 确认缩进是否正确（Python对空格很敏感）
- 复制错误信息搜索一下

**Q: 看不懂怎么办？**
- 跳过难的部分，先学简单的
- 看看前面的课程有没有遗漏
- 多看几遍，理解需要时间

**Q: 需要什么基础？**
- 会用电脑就行
- Python基础最好有，没有也能学
- 数学不好也没关系，我们用例子讲

### 📌 学习技巧

- 🎯 **目标明确**: 知道这节课要学什么
- 📝 **做笔记**: 重点内容记下来
- 💻 **多练习**: 代码要自己敲一遍
- 🤔 **多思考**: 想想为什么这样做
- 🔄 **多复习**: 学完了回头再看看

---
""".split('\n')
        }
        
        # 插入到第二个位置（第一个是标题）
        notebook['cells'].insert(1, tips_cell)
        
        # 保存
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, ensure_ascii=False, indent=1)
        
        print(f"  ✓ {filename}")
        return True
        
    except Exception as e:
        print(f"  ✗ {filename} - 失败: {e}")
        return False

def main():
    notebooks_dir = Path('/Users/h/practice/CV-main')
    
    print("🚀 为所有笔记本添加入门提示...\n")
    
    success_count = 0
    total_count = 0
    
    for file in sorted(os.listdir(notebooks_dir)):
        if file.endswith('.ipynb') and not file.endswith('_backup.ipynb'):
            total_count += 1
            nb_path = notebooks_dir / file
            if add_beginner_tips(nb_path):
                success_count += 1
    
    print(f"\n✅ 完成! 成功优化 {success_count}/{total_count} 个笔记本")
    print("\n🎉 现在每个笔记本都有:")
    print("   ✓ 新手必看提示")
    print("   ✓ 学习建议")
    print("   ✓ 常见问题解答")
    print("   ✓ 学习技巧")

if __name__ == "__main__":
    main()

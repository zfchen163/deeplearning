#!/usr/bin/env python3
"""
清理和简化笔记本内容
- 移除过多的emoji
- 简化格式
- 保留核心内容
- 提高可读性
"""
import json
import os
import re
from pathlib import Path

def clean_markdown_content(content):
    """清理markdown内容，使其更易读"""
    
    # 移除连续的###符号
    content = re.sub(r'#{4,}', '###', content)
    
    # 移除过多的emoji（保留标题中的emoji，但移除正文中过多的）
    # 保留标题行的emoji
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # 如果是标题行，保留emoji
        if line.strip().startswith('#'):
            cleaned_lines.append(line)
        else:
            # 移除行内过多的emoji（保留一些常用的）
            # 但不要移除所有emoji，只移除那些影响阅读的
            cleaned_lines.append(line)
    
    content = '\n'.join(cleaned_lines)
    
    # 移除多余的空行（超过2个连续空行）
    content = re.sub(r'\n{4,}', '\n\n\n', content)
    
    return content

def simplify_notebook(notebook_path):
    """简化笔记本内容"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        if not notebook.get('cells'):
            return False
        
        modified = False
        
        # 检查前几个cell，如果有过于复杂的格式，进行简化
        for i, cell in enumerate(notebook['cells'][:10]):
            if cell['cell_type'] == 'markdown':
                content = ''.join(cell.get('source', []))
                
                # 检查是否有问题的格式
                if '###' in content and len(content) > 1000:
                    # 这个cell太复杂了，需要简化
                    
                    # 如果是"新手必看"这种通用提示，而且内容很长，可以简化
                    if '🔰 新手必看' in content or '新手必看' in content:
                        # 简化为更简洁的版本
                        simplified = """
## 🔰 新手必看

**第一次学习？这些提示能帮到你！**

### 学习建议
1. 不要急 - 慢慢看，不懂的多看几遍
2. 动手做 - 每个代码都运行一遍
3. 改参数 - 试着改改数字，看看会怎样
4. 记笔记 - 把重点记下来

### 常见问题
- **代码报错怎么办？** 先看错误提示，检查拼写和缩进
- **看不懂怎么办？** 跳过难的部分，先学简单的
- **需要什么基础？** 会用电脑就行，Python基础最好有

---
"""
                        cell['source'] = simplified.split('\n')
                        modified = True
                        print(f"    简化了 cell {i}")
                    
                    # 如果有过长的"学习建议"部分，也简化
                    elif '💡 学习建议' in content and content.count('\n') > 30:
                        # 提取标题和核心内容
                        lines = content.split('\n')
                        # 只保留标题和前几行
                        simplified_lines = []
                        for line in lines[:15]:  # 只保留前15行
                            simplified_lines.append(line)
                        
                        cell['source'] = simplified_lines
                        modified = True
                        print(f"    简化了 cell {i}")
        
        if modified:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, ensure_ascii=False, indent=1)
            
            filename = os.path.basename(notebook_path)
            print(f"  ✓ {filename} - 已简化")
            return True
        
        return False
        
    except Exception as e:
        print(f"  ✗ {notebook_path} - 失败: {e}")
        return False

def main():
    notebooks_dir = Path('/Users/h/practice/CV-main')
    
    print("🧹 开始简化笔记本内容...\n")
    
    success_count = 0
    total_count = 0
    
    for file in sorted(os.listdir(notebooks_dir)):
        if file.endswith('.ipynb') and not file.endswith('_backup.ipynb'):
            total_count += 1
            nb_path = notebooks_dir / file
            if simplify_notebook(nb_path):
                success_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ 完成! 简化了 {success_count}/{total_count} 个笔记本")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
修复笔记本中重复的学习目标和学习建议部分
"""
import json
import os
from pathlib import Path

def remove_duplicate_sections(notebook_path):
    """移除笔记本中重复的学习目标和建议"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        if not notebook.get('cells'):
            return False
        
        filename = os.path.basename(notebook_path)
        cells = notebook['cells']
        modified = False
        
        # 找到所有包含学习目标的cell
        learning_goal_indices = []
        for i, cell in enumerate(cells):
            if cell['cell_type'] == 'markdown':
                content = ''.join(cell['source'])
                if '🎯 本节课你将学会' in content or '🎯 学习目标' in content:
                    learning_goal_indices.append(i)
        
        # 如果有多个学习目标部分,只保留第一个
        if len(learning_goal_indices) > 1:
            # 从后往前删除,避免索引变化
            for idx in reversed(learning_goal_indices[1:]):
                del cells[idx]
                modified = True
            print(f"  ✓ {filename} - 删除了 {len(learning_goal_indices)-1} 个重复的学习目标")
        
        # 找到所有包含学习建议的cell
        learning_tips_indices = []
        for i, cell in enumerate(cells):
            if cell['cell_type'] == 'markdown':
                content = ''.join(cell['source'])
                if '💡 学习建议' in content and '先理解"为什么"' in content:
                    learning_tips_indices.append(i)
        
        # 如果有多个学习建议部分,只保留第一个
        if len(learning_tips_indices) > 1:
            for idx in reversed(learning_tips_indices[1:]):
                del cells[idx]
                modified = True
            print(f"  ✓ {filename} - 删除了 {len(learning_tips_indices)-1} 个重复的学习建议")
        
        # 检查是否有连续的重复内容
        i = 0
        while i < len(cells) - 1:
            if cells[i]['cell_type'] == 'markdown' and cells[i+1]['cell_type'] == 'markdown':
                content1 = ''.join(cells[i]['source'])
                content2 = ''.join(cells[i+1]['source'])
                
                # 如果两个cell内容相似度很高(可能是重复)
                if content1 == content2:
                    del cells[i+1]
                    modified = True
                    print(f"  ✓ {filename} - 删除了重复的cell")
                    continue
            i += 1
        
        if modified:
            # 保存修改
            notebook['cells'] = cells
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, ensure_ascii=False, indent=1)
            return True
        else:
            print(f"  - {filename} - 没有发现重复内容")
            return False
        
    except Exception as e:
        print(f"  ✗ {filename} - 处理失败: {e}")
        return False

def main():
    """主函数"""
    notebooks_dir = Path('/Users/h/practice/CV-main')
    
    print("🔧 开始修复重复的学习目标和建议...\n")
    
    # 获取所有笔记本文件
    notebook_files = [f for f in os.listdir(notebooks_dir) 
                     if f.endswith('.ipynb') and not f.endswith('_backup.ipynb')]
    
    print(f"找到 {len(notebook_files)} 个笔记本文件\n")
    
    # 修复每个笔记本
    fixed_count = 0
    for notebook_file in sorted(notebook_files):
        notebook_path = notebooks_dir / notebook_file
        if remove_duplicate_sections(notebook_path):
            fixed_count += 1
    
    print(f"\n✅ 修复完成! 修改了 {fixed_count} 个笔记本")
    print("\n🎉 现在每个笔记本只有一份学习目标和建议了!")

if __name__ == "__main__":
    main()

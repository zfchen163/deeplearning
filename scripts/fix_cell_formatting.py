#!/usr/bin/env python3
"""
修复cell格式问题
- 确保每个cell的source是正确的列表格式
- 每行文字单独一个元素
- 添加适当的换行
"""
import json
import os
from pathlib import Path

def fix_cell_source(cell):
    """修复cell的source格式"""
    if cell['cell_type'] != 'markdown':
        return False
    
    source = cell.get('source', [])
    if not source:
        return False
    
    # 如果source是字符串，转换为列表
    if isinstance(source, str):
        cell['source'] = source.split('\n')
        return True
    
    # 如果source是列表，检查每个元素
    if isinstance(source, list):
        # 检查是否有元素没有换行符
        needs_fix = False
        for item in source:
            if isinstance(item, str) and '\n' not in item and len(item) > 100:
                needs_fix = True
                break
        
        if needs_fix:
            # 重新格式化
            full_text = ''.join(source)
            # 按行分割，每行末尾加\n
            lines = full_text.split('\n')
            cell['source'] = [line + '\n' if i < len(lines)-1 else line 
                            for i, line in enumerate(lines)]
            return True
    
    return False

def fix_notebook(notebook_path):
    """修复笔记本格式"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        if not notebook.get('cells'):
            return False
        
        modified = False
        
        for i, cell in enumerate(notebook['cells']):
            if fix_cell_source(cell):
                modified = True
                print(f"    修复了 cell {i}")
        
        if modified:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, ensure_ascii=False, indent=1)
            
            filename = os.path.basename(notebook_path)
            print(f"  ✓ {filename}")
            return True
        
        return False
        
    except Exception as e:
        print(f"  ✗ {notebook_path} - 失败: {e}")
        return False

def main():
    notebooks_dir = Path('/Users/h/practice/CV-main')
    
    print("🔧 修复笔记本格式...\n")
    
    success_count = 0
    total_count = 0
    
    for file in sorted(os.listdir(notebooks_dir)):
        if file.endswith('.ipynb') and not file.endswith('_backup.ipynb'):
            total_count += 1
            nb_path = notebooks_dir / file
            if fix_notebook(nb_path):
                success_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ 完成! 修复了 {success_count}/{total_count} 个笔记本")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

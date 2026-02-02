#!/usr/bin/env python3
"""
删除笔记本中重复的cell
"""
import json
import os
from pathlib import Path

def remove_duplicate_cells(notebook_path):
    """删除笔记本中重复的markdown cells"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        if not notebook.get('cells'):
            return False
        
        cells = notebook['cells']
        seen_content = {}
        cells_to_remove = []
        
        # 找出重复的cells
        for i, cell in enumerate(cells):
            if cell['cell_type'] == 'markdown':
                content = ''.join(cell.get('source', []))
                
                # 跳过空cell
                if len(content.strip()) < 10:
                    continue
                
                # 使用内容的前500字符作为唯一标识
                content_key = content[:500].strip()
                
                if content_key in seen_content:
                    # 发现重复，标记删除
                    cells_to_remove.append(i)
                    print(f"    发现重复cell {i}: {content[:50]}...")
                else:
                    seen_content[content_key] = i
        
        if not cells_to_remove:
            return False
        
        # 从后往前删除，避免索引变化
        for i in reversed(cells_to_remove):
            del notebook['cells'][i]
        
        # 保存
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, ensure_ascii=False, indent=1)
        
        filename = os.path.basename(notebook_path)
        print(f"  ✓ {filename} - 删除了 {len(cells_to_remove)} 个重复cell")
        return True
        
    except Exception as e:
        print(f"  ✗ {notebook_path} - 失败: {e}")
        return False

def main():
    notebooks_dir = Path('/Users/h/practice/CV-main')
    
    print("🧹 开始清理重复的cell...\n")
    
    success_count = 0
    total_count = 0
    
    for file in sorted(os.listdir(notebooks_dir)):
        if file.endswith('.ipynb') and not file.endswith('_backup.ipynb'):
            total_count += 1
            nb_path = notebooks_dir / file
            if remove_duplicate_cells(nb_path):
                success_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ 完成! 清理了 {success_count}/{total_count} 个笔记本")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

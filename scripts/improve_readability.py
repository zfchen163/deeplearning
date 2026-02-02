#!/usr/bin/env python3
"""
改善笔记本阅读体验
- 修复文字挤在一起的问题
- 优化段落间距
- 清理过度复杂的格式
"""

import json
import os
import re
from pathlib import Path

def clean_markdown_content(text):
    """清理和优化markdown内容"""
    if not text or not text.strip():
        return text
    
    # 1. 移除过多的emoji和符号
    # 保留必要的emoji，但删除过度使用的
    text = re.sub(r'(#{3,})\s*', r'\1 ', text)  # 确保###后有空格
    
    # 2. 优化"新手必看"部分 - 简化格式
    if '新手必看' in text and len(text) > 500:
        # 如果内容太长太复杂，简化它
        lines = text.split('\n')
        new_lines = []
        skip_complex = False
        
        for line in lines:
            # 跳过过度复杂的格式行
            if '💡 学习建议1.' in line or '不要急 - 慢慢看' in line:
                skip_complex = True
                # 添加简化版本
                new_lines.append('\n## 🔰 新手必看\n')
                new_lines.append('\n**第一次学习？这些提示很重要：**\n')
                new_lines.append('\n')
                new_lines.append('### 学习方法\n')
                new_lines.append('\n')
                new_lines.append('1. **慢慢学** - 不懂的地方多看几遍\n')
                new_lines.append('2. **动手做** - 每段代码都运行一遍\n')
                new_lines.append('3. **改参数** - 试着修改数字看效果\n')
                new_lines.append('4. **记笔记** - 记录重点内容\n')
                new_lines.append('\n')
                new_lines.append('### 遇到问题怎么办\n')
                new_lines.append('\n')
                new_lines.append('- **代码报错**: 看红色错误提示，检查拼写和缩进\n')
                new_lines.append('- **看不懂**: 先跳过，学简单的，再回来看\n')
                new_lines.append('- **需要基础**: 会用电脑就行，Python基础更好\n')
                new_lines.append('\n')
                new_lines.append('---\n')
                new_lines.append('\n')
                continue
            
            if skip_complex:
                # 跳过旧的复杂内容，直到遇到分隔线
                if line.strip() == '---' or line.strip().startswith('# '):
                    skip_complex = False
                    if line.strip() == '---':
                        continue  # 跳过这个分隔线，我们已经加过了
                continue
            
            new_lines.append(line)
        
        text = '\n'.join(new_lines)
    
    # 3. 确保标题后有空行
    text = re.sub(r'(^#{1,6}\s+.+)$\n(?!\n)', r'\1\n\n', text, flags=re.MULTILINE)
    
    # 4. 确保列表项之间有适当间距
    text = re.sub(r'(\n[-*]\s+.+)\n(?=[-*]\s+)', r'\1\n', text)
    
    # 5. 清理多余的空行（超过2个连续空行）
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    
    # 6. 确保分隔线前后有空行
    text = re.sub(r'(?<!\n)\n(---+)\n(?!\n)', r'\n\n\1\n\n', text)
    
    return text

def improve_notebook_readability(notebook_path):
    """改善单个笔记本的阅读体验"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        modified = False
        
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'markdown':
                source = cell.get('source', [])
                
                # 合并source为文本
                if isinstance(source, list):
                    original_text = ''.join(source)
                else:
                    original_text = source
                
                # 清理和优化内容
                cleaned_text = clean_markdown_content(original_text)
                
                if cleaned_text != original_text:
                    # 转换回列表格式，每行一个元素
                    lines = cleaned_text.split('\n')
                    cell['source'] = [line + '\n' if i < len(lines) - 1 else line 
                                     for i, line in enumerate(lines)]
                    modified = True
        
        if modified:
            # 保存修改
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, ensure_ascii=False, indent=1)
            return True
        
        return False
    
    except Exception as e:
        print(f"❌ 处理失败 {notebook_path}: {e}")
        return False

def main():
    """主函数"""
    # 获取所有笔记本
    notebooks = list(Path('.').glob('*.ipynb'))
    
    print(f"📚 找到 {len(notebooks)} 个笔记本")
    print("🔧 开始改善阅读体验...\n")
    
    improved_count = 0
    
    for notebook_path in sorted(notebooks):
        if improve_notebook_readability(notebook_path):
            improved_count += 1
            print(f"✅ {notebook_path.name}")
    
    print(f"\n🎉 完成！")
    print(f"📊 改善了 {improved_count} 个笔记本")
    
    if improved_count > 0:
        print("\n💡 建议:")
        print("   1. 刷新浏览器查看效果（Cmd+Shift+R）")
        print("   2. 检查几个笔记本确认格式正确")
        print("   3. 如果满意，提交更改")

if __name__ == '__main__':
    main()

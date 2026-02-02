
import os
import json
from pathlib import Path

def check_notebooks():
    base_dir = Path("/Users/h/practice/CV-main")
    all_notebooks = list(base_dir.glob("**/*.ipynb"))
    
    unoptimized_files = []
    
    for file_path in all_notebooks:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                
            has_tips = False
            # Check first few cells for "新手必看"
            for cell in content.get('cells', [])[:3]:
                if cell['cell_type'] == 'markdown':
                    source = "".join(cell.get('source', []))
                    if "新手必看" in source:
                        has_tips = True
                        break
            
            if not has_tips:
                unoptimized_files.append(str(file_path))
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print(f"Total notebooks: {len(all_notebooks)}")
    print(f"Unoptimized notebooks: {len(unoptimized_files)}")
    print("\nList of unoptimized files:")
    for f in unoptimized_files:
        print(f)

if __name__ == "__main__":
    check_notebooks()

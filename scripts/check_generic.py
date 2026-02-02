
import os
import json
from pathlib import Path

def check_generic_headers():
    base_dir = Path("/Users/h/practice/CV-main")
    all_notebooks = list(base_dir.glob("**/*.ipynb"))
    
    generic_files = []
    
    for file_path in all_notebooks:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            is_generic = False
            # Check first cell for generic phrases
            if content.get('cells') and content['cells'][0]['cell_type'] == 'markdown':
                source = "".join(content['cells'][0]['source'])
                if "让我们从实际问题出发" in source or "Generic Placeholder" in source:
                    is_generic = True
            
            if is_generic:
                generic_files.append(str(file_path))
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print(f"Total notebooks: {len(all_notebooks)}")
    print(f"Notebooks with Generic Headers: {len(generic_files)}")
    print("\nList of files needing specific analogies:")
    for f in generic_files:
        print(f)

if __name__ == "__main__":
    check_generic_headers()

import os
from pathlib import Path
from typing import List

def load_markdown(file_path: Path) -> str:
    """Load markdown file content"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
    
    
def find_markdown_files(directory: Path) -> List[Path]:
    """
    Recursively find all markdown files in the given directory
    
    Args:
        directory: The directory to search in
        
    Returns:
        List of paths to markdown files
    """
    markdown_files = []
        
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.md'):
                markdown_files.append(Path(os.path.join(root, file)))
    
    return markdown_files
import re
from typing import Dict, List


def extract_page_map(content: str) -> Dict[int, str]:
    """Extract page number to content mapping"""
    # Split by page markers like {19}------------------------------------------------
    pages = re.split(r'\{(\d+)\}-+', content)
    
    page_map = {}
    if len(pages) > 1:  # If we found page markers
        for i in range(1, len(pages), 2):
            if i+1 < len(pages):  # Ensure we don't go out of bounds
                page_num = int(pages[i])
                page_content = pages[i+1].strip()
                page_map[page_num] = page_content
    else:
        # No page markers found, treat as a single page
        page_map[1] = content
        
    return page_map

def merge_pages(content: str) -> str:
    """Merge all pages into single content by removing pagination markers"""
    # Remove page markers and merge
    merged = re.sub(r'\{(\d+)\}-+', ' ', content)
    return merged.strip()



def get_chunk_page_numbers(chunk_text: str, page_map: Dict[int, str]) -> List[int]:
    """
    Determine which page numbers a chunk belongs to
    
    Args:
        chunk_text: The text of the chunk
        page_map: Dictionary mapping page numbers to page content
        
    Returns:
        List of page numbers this chunk belongs to
    """
    page_numbers = []
    
    # For each page, check if a significant portion of the chunk is in that page
    for page_num, page_content in page_map.items():
        # If the chunk text is found in the page content, add the page number
        if chunk_text in page_content:
            page_numbers.append(page_num)
            continue
            
        # Check for partial matches (at least 50 characters in sequence)
        min_match_length = min(50, len(chunk_text) // 2)
        
        # Check for smaller sequences from the chunk in the page
        for i in range(len(chunk_text) - min_match_length + 1):
            sequence = chunk_text[i:i+min_match_length]
            if sequence in page_content:
                page_numbers.append(page_num)
                break
    
    return page_numbers

"""
Utility functions for PDF content chunking.
"""

import re
from typing import List, Dict, Any
import tiktoken
from constants import ENCODING_NAME

def extract_text_from_html(html: str) -> str:
    """
    Extract clean text from HTML content.
    
    Args:
        html: HTML content string
        
    Returns:
        Clean text with preserved whitespace
    """
    if not html:
        return ""
        
    # Remove content-ref tags and their contents
    html = re.sub(r'<content-ref[^>]*>.*?</content-ref>', '', html)
    
    # Replace <br/> and <br> with newlines
    html = html.replace('<br/>', '\n').replace('<br>', '\n')
    
    # Replace </p>, </div>, </tr> with newlines
    html = re.sub(r'</(?:p|div|tr)>', '\n', html)
    
    # Remove all other HTML tags
    html = re.sub(r'<[^>]+>', '', html)
    
    # Decode HTML entities
    html = html.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', html)
    
    return text.strip()

def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    magnitude1 = sum(a * a for a in embedding1) ** 0.5
    magnitude2 = sum(b * b for b in embedding2) ** 0.5
    if magnitude1 * magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text using the tiktoken encoder.
    
    Args:
        text: Text to tokenize
        
    Returns:
        Number of tokens
    """
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    tokens = encoding.encode(text)
    return len(tokens)

def find_block_by_id(node: Dict[str, Any], target_id: str) -> Dict[str, Any]:
    """
    Recursively search for a block with the given ID in the document tree.
    
    Args:
        node: Current document node
        target_id: ID to search for
        
    Returns:
        Block if found, None otherwise
    """
    if node.get("id") == target_id:
        return node
    
    if "children" in node and node["children"]:
        for child in node["children"]:
            result = find_block_by_id(child, target_id)
            if result:
                return result
    
    return None

def extract_page_id(block_id: str) -> str:
    """
    Extract the page ID from a block ID.
    
    Args:
        block_id: Block ID string (e.g., "/page/0/Text/3")
        
    Returns:
        Page ID or None if not found
    """
    if "/page/" in block_id:
        return block_id.split("/")[2]
    return None 
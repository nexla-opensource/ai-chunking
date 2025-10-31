import tiktoken
from ai_chunking.utils.logging import logger


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string using tiktoken.
    
    Args:
        text: The text to count tokens for
        
    Returns:
        int: The number of tokens in the text
    """
    try:
        # For newer models like GPT-4
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        # Fallback to a simple approximation if tiktoken fails
        logger.warning(f"Error counting tokens while using tiktoken: {e}")
        return int(round(len(text.split()) * 1.3))  # Rough approximation as integer

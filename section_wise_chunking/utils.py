from difflib import SequenceMatcher
import re
from pathlib import Path
from typing import Dict, List, Tuple
import asyncio
import tiktoken
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from litellm import AsyncOpenAI

from constants import MODEL_NAME
from json_utils import parse_json_response
from prompts import SYSTEM_PROMPT



client = AsyncOpenAI()

async def process_with_llm(prompt: str) -> Tuple[str, Dict]:
    """Process text with LLM and return summary and metadata"""
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        response_format={ "type": "json_object" }  # Ensure JSON response
    )
    
    # Parse structured response
    response_json = parse_json_response(response.choices[0].message.content)
    return response_json


def load_markdown(file_path: Path) -> str:
    """Load markdown file content"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_page_map(content: str) -> Dict[int, str]:
    """Extract page number to content mapping"""
    # Split by page markers like {19}------------------------------------------------
    pages = re.split(r'\{(\d+)\}-+', content)
    
    page_map = {}
    for i in range(1, len(pages), 2):
        page_num = int(pages[i])
        page_content = pages[i+1].strip()
        page_map[page_num] = page_content
        
    return page_map

def merge_pages(content: str) -> str:
    """Merge all pages into single content"""
    # Remove page markers and merge
    merged = re.sub(r'\{(\d+)\}-+', ' ', content)
    return merged.strip()

def extract_image_urls(text: str) -> List[str]:
    """Extract markdown image URLs from text"""
    # Match markdown image syntax ![alt](url)
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    matches = re.findall(image_pattern, text)
    return [url for _, url in matches]

async def run_concurrent_tasks(tasks: List[asyncio.Task], max_concurrent: int = 5):
    """Run tasks concurrently with semaphore"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def wrapped_task(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(
        *(wrapped_task(task) for task in tasks),
        return_exceptions=True
    )

def create_semantic_splitter(min_chunk_size: int) -> SemanticChunker:
    """Create semantic text splitter"""
    return SemanticChunker(
        embeddings=OpenAIEmbeddings(model="text-embedding-3-large"),
        min_chunk_size=min_chunk_size
    )

def text_similarity_ratio(text1: str, text2: str) -> float:
    return SequenceMatcher(None, text1, text2).ratio()

def count_tokens(text: str, model_name: str = MODEL_NAME) -> int:
    """
    Count the number of tokens in a text string using tiktoken.
    
    Args:
        text: The text to count tokens for
        model_name: The model name to use for tokenization (defaults to the model in constants)
        
    Returns:
        int: The number of tokens in the text
    """
    try:
        # For newer models like gpt-3.5-turbo and gpt-4
        if "gpt-4" in model_name or "gpt-3.5" in model_name:
            encoding_name = "cl100k_base"
        # For older models like davinci
        elif "davinci" in model_name or "curie" in model_name or "babbage" in model_name or "ada" in model_name:
            encoding_name = "p50k_base"
        else:
            # Default to cl100k_base for newer models
            encoding_name = "cl100k_base"
            
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        # Fallback to a simple approximation if tiktoken fails
        print(f"Error counting tokens: {e}")
        return len(text.split()) * 1.3  # Rough approximation

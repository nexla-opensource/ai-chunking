from difflib import SequenceMatcher
import re
from pathlib import Path
from typing import Dict, List, Tuple
import asyncio
import tiktoken
from litellm import AsyncOpenAI

from ai_chunking.utils.json_utils import parse_json_response
from ai_chunking.chunkers.section_based_semantic_chunker.prompts import SYSTEM_PROMPT


MODEL_NAME = "gpt-4o-mini"
client = AsyncOpenAI()

async def process_with_llm(prompt: str, model_name: str = MODEL_NAME) -> Tuple[str, Dict]:
    """Process text with LLM and return summary and metadata"""
    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        response_format={ "type": "json_object" }  # Ensure JSON response
    )
    
    # Parse structured response
    response_json = parse_json_response(response.choices[0].message.content)
    return response_json


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




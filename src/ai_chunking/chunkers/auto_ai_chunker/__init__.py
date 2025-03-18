import os
from typing import List
import asyncio
import concurrent.futures

from ai_chunking.chunkers.auto_ai_chunker.models.document import Page
from ai_chunking.chunkers.auto_ai_chunker.processor import DocumentProcessor
from ai_chunking.llm.base import LLMConfig
from ai_chunking.llm.factory import LLMFactory, LLMProvider
from ai_chunking.llm.models import OpenAIModels
from ai_chunking.models.chunk import Chunk
from ai_chunking.utils.markdown_utils import load_markdown


class AutoAIChunker:    
    def __init__(self): 
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        
        # Initialize LLM clients
        small_llm = LLMFactory.create(
            provider=LLMProvider.OPENAI,
            api_key=api_key,
            config=LLMConfig(model=OpenAIModels.GPT_4O_MINI, temperature=0.0)
        )
        
        large_llm = LLMFactory.create(
            provider=LLMProvider.OPENAI,
            api_key=api_key,
            config=LLMConfig(model=OpenAIModels.GPT_4O, temperature=0.0)
        )

        # Initialize processor
        self.processor = DocumentProcessor(
            llm_client=large_llm,
            small_llm_client=small_llm,
            large_llm_client=large_llm
        )
    
    def _run_async(self, coro):
        """Safely run a coroutine in the appropriate event loop"""
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If the loop is already running, create a new loop in a new thread
                import threading
                import functools
                
                def run_in_new_loop(coro):
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    return new_loop.run_until_complete(coro)
                
                # Run in a new thread with a new event loop
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(run_in_new_loop, coro)
                    return future.result()
            else:
                # If the loop exists but isn't running, use it
                return loop.run_until_complete(coro)
        except RuntimeError:
            # If no event loop exists in this thread
            return asyncio.run(coro)
    
    def chunk_documents(self, documents: List[str]) -> List[Chunk]:
        chunks = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks
    
    def chunk_document(self, document: str) -> List[Chunk]:
        content = load_markdown(document)
        chunks = self._run_async(self.processor.process_document(
            pages=[Page(text=content, page_number=1)],
            table_data=[],
            metadata={},
            source=document
        ))
        return chunks
    
    
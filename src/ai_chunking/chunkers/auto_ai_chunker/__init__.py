import os
from typing import List

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
    
    async def chunk_documents(self, documents: List[str]) -> List[Chunk]:
        chunks = []
        for document in documents:
            chunks.extend(await self.chunk_document(document))
        return chunks
    
    async def chunk_document(self, document: str) -> List[Chunk]:
        content = load_markdown(document)
        chunks = await self.processor.process_document(
            pages=[Page(text=content, page_number=1)],
            table_data=[],
            metadata={},
            source=document
        )
        return chunks
    
    
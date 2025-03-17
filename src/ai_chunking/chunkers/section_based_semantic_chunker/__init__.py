import asyncio
from pathlib import Path
from typing import List
import time
import concurrent.futures
from functools import partial
import logging

from ai_chunking.models.chunk import Chunk

from ai_chunking.chunkers.base_chunker import BaseChunker
from ai_chunking.chunkers.section_based_semantic_chunker.models import Document
from ai_chunking.chunkers.section_based_semantic_chunker.processors import (
    assign_page_numbers_to_chunks,
    process_sections,
    create_chunks,
    process_chunks,
    process_document_summary
)
from ai_chunking.utils.markdown_utils import load_markdown
from ai_chunking.utils.count_tokens import count_tokens


PAGE_CONTENT_SIMILARITY_THRESHOLD = 0.8
MIN_SECTION_CHUNK_SIZE = 5000
MAX_SECTION_CHUNK_SIZE = 50000
MIN_CHUNK_SIZE = 128
MAX_CHUNK_SIZE = 1000
MODEL_NAME = "gpt-4o-mini"
logger = logging.getLogger(__name__)


class SectionBasedSemanticChunker(BaseChunker):
    def __init__(self, 
                 page_content_similarity_threshold: float = PAGE_CONTENT_SIMILARITY_THRESHOLD,
                 min_section_chunk_size: int = MIN_SECTION_CHUNK_SIZE,
                 max_section_chunk_size: int = MAX_SECTION_CHUNK_SIZE,
                 min_chunk_size: int = MIN_CHUNK_SIZE,
                 max_chunk_size: int = MAX_CHUNK_SIZE):
        self.page_content_similarity_threshold = page_content_similarity_threshold
        self.min_section_chunk_size = min_section_chunk_size
        self.max_section_chunk_size = max_section_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self._executor = concurrent.futures.ThreadPoolExecutor()

    def chunk_text(self, text: str) -> List[Chunk]:
        raise NotImplementedError("SectionBasedSemanticChunker does not support chunking text")

    def chunk_documents(self, file_paths: List[str]) -> List[Chunk]:
        all_chunks = []
        for file_path in file_paths:
            chunks = self.chunk_document(file_path)
            all_chunks.extend(chunks)
        return all_chunks
    
    def _run_async(self, coro):
        """Safely run a coroutine in the appropriate event loop"""
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If the loop is already running, use asyncio.run_coroutine_threadsafe
                # or a new thread with asyncio.run()
                future = concurrent.futures.ThreadPoolExecutor().submit(asyncio.run, coro)
                return future.result()
            else:
                # If the loop exists but isn't running, use it
                return loop.run_until_complete(coro)
        except RuntimeError:
            # If no event loop exists in this thread
            return asyncio.run(coro)

    def chunk_document(self, file_path: str) -> List[Chunk]:
        """Process a single document synchronously"""
        start_time = time.perf_counter()
        
        # Load and prepare document
        content = load_markdown(file_path)
        initial_tokens_count = count_tokens(content)
        
        document = Document(
            name=Path(file_path).stem,
            text=content,
            tokens_count=initial_tokens_count, 
        )
        
        prep_time = time.perf_counter() - start_time
        logger.info(f"Document preparation took {prep_time:.2f} seconds")
        
        # Create sections
        section_start = time.perf_counter()
        logger.info("Creating and processing sections...")
        sections = create_chunks(document.text, self.min_section_chunk_size, self.max_section_chunk_size)
        logger.info(f"Created {len(sections)} sections")

        # Process sections concurrently using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create chunks for all sections concurrently
            chunk_creator = partial(create_chunks, min_chunk_size=self.min_chunk_size, max_chunk_size=self.max_chunk_size)
            section_chunks_futures = [
                executor.submit(chunk_creator, section_text)
                for section_text in sections
            ]
            all_section_chunks = [future.result() for future in concurrent.futures.as_completed(section_chunks_futures)]

        # Process sections using our safe async runner
        processed_sections = self._run_async(process_sections(sections))
        document.sections = processed_sections
        section_time = time.perf_counter() - section_start
        logger.info(f"Section processing took {section_time:.2f} seconds")
        
        # Create document summary
        summary_start = time.perf_counter()
        document.summary = self._run_async(
            process_document_summary([section.summary for section in document.sections])
        )
        summary_time = time.perf_counter() - summary_start
        logger.info(f"Document summary creation took {summary_time:.2f} seconds")

        # Process chunks for all sections
        chunk_start = time.perf_counter()
        logger.info("Processing chunks for all sections...")
        
        all_processed_chunks = []
        for i, section in enumerate(document.sections):
            chunks = all_section_chunks[i]
            processed_chunks = self._run_async(
                process_chunks(chunks, section.summary, document.summary)
            )
            if not hasattr(section, 'chunks'):
                section.chunks = []
            section.chunks.extend(processed_chunks)
            all_processed_chunks.extend(processed_chunks)

        chunk_time = time.perf_counter() - chunk_start
        logger.info(f"Chunk processing took {chunk_time:.2f} seconds")
        
        # Update chunk relationships
        relation_start = time.perf_counter()
        logger.info("Updating chunk relationships...")
        
        for i, chunk in enumerate(all_processed_chunks):
            if i > 0:
                chunk.previous_chunk_summary = all_processed_chunks[i-1].summary.text
            if i < len(all_processed_chunks) - 1:
                chunk.next_chunk_summary = all_processed_chunks[i+1].summary.text
        
        relation_time = time.perf_counter() - relation_start
        logger.info(f"Chunk relationship updates took {relation_time:.2f} seconds")
        
        # Efficiently assign page numbers to chunks
        # page_start = time.perf_counter()
        # logger.info("Updating the page numbers to the chunks")
        # assign_page_numbers_to_chunks(document)
        # page_time = time.perf_counter() - page_start
        # logger.info(f"Page number assignment took {page_time:.2f} seconds")
        
        total_time = time.perf_counter() - start_time
        logger.info(f"Document processed: {document.name}, {document.total_pages} pages, {document.tokens_count} tokens")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        chunks = [Chunk(text=chunk.text, metadata={
            "tokens_count": chunk.tokens_count,
            "page_numbers": chunk.page_numbers,
            "summary": chunk.summary,
            "global_context": chunk.global_context,
            "section_summary": chunk.section_summary,
            "document_summary": chunk.document_summary,
            "previous_chunk_summary": chunk.previous_chunk_summary,
            "next_chunk_summary": chunk.next_chunk_summary,
            "questions_this_excerpt_can_answer": chunk.questions_this_excerpt_can_answer    
        }) for chunk in all_processed_chunks]

        return chunks

    def __del__(self):
        self._executor.shutdown(wait=True)

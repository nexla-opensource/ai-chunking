import asyncio
from typing import List
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
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

    def chunk_documents(self, file_paths: List[str]) -> List[Chunk]:
        async def process_documents():
            tasks = []
            for file_path in file_paths:
                tasks.append(self.chunk_document(file_path))
            return await asyncio.gather(*tasks)

        all_chunks = []
        results = asyncio.run(process_documents())
        for document_chunks in results:
            all_chunks.extend(document_chunks)
        return all_chunks

    def chunk_document(self, file_path: str) -> List[Chunk]:
        return asyncio.run(self.chunk_document(file_path))

    async def chunk_document(self, file_path: str) -> List[Chunk]:
        """
        Main function to create semantic chunks from markdown document with parallel processing
        """
        start_time = time.perf_counter()
        
        # Load and prepare document
        content = load_markdown(file_path)
        initial_tokens_count = count_tokens(content)
        
        document = Document(
            name=file_path.stem,
            text=content,
            tokens_count=initial_tokens_count, 
        )
        
        prep_time = time.perf_counter() - start_time
        logger.info(f"Document preparation took {prep_time:.2f} seconds")
        
        # Create sections
        section_start = time.perf_counter()
        logger.info("Creating and processing sections...")
        sections = create_chunks(document.text, MIN_SECTION_CHUNK_SIZE, MAX_SECTION_CHUNK_SIZE)
        logger.info(f"Created {len(sections)} sections")

        # Start section processing task
        sections_task = process_sections(sections)
        
        # Create chunks for all sections using multiprocessing
        loop = asyncio.get_event_loop()
        max_workers = min(multiprocessing.cpu_count(), len(sections))  # Don't create more processes than sections
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Create a partial function with the fixed min_chunk_size parameter
            chunk_creator = partial(create_chunks, min_chunk_size=MIN_CHUNK_SIZE, max_chunk_size=MAX_CHUNK_SIZE)
            # Submit all tasks to the process pool
            futures = [
                loop.run_in_executor(executor, chunk_creator, section_text)
                for idx, section_text in enumerate(sections)
            ]
            all_section_chunks = await asyncio.gather(*futures)
        
        # Wait for section processing to complete
        processed_sections = await sections_task
        document.sections = processed_sections
        section_time = time.perf_counter() - section_start
        logger.info(f"Section processing took {section_time:.2f} seconds")
        
        # Create document summary
        summary_start = time.perf_counter()
        document.summary = await process_document_summary([section.summary for section in document.sections])
        summary_time = time.perf_counter() - summary_start
        logger.info(f"Document summary creation took {summary_time:.2f} seconds")

        # Process chunks for all sections in parallel
        chunk_start = time.perf_counter()
        logger.info("Processing chunks for all sections...")
        chunk_tasks = []
        
        # Launch all chunk processing tasks at once
        for i, section in enumerate(document.sections):
            # We already pre-computed chunks while waiting for sections
            chunks = all_section_chunks[i]
            # Process all chunks for a section at once
            task = process_chunks(chunks, section.summary, document.summary)
            chunk_tasks.append((section, task))
        
        # Process all chunks in parallel
        chunk_results = await asyncio.gather(*(task for _, task in chunk_tasks))
        
        # Assign chunks back to their sections
        for (section, _), processed_chunks in zip(chunk_tasks, chunk_results):
            if not hasattr(section, 'chunks'):
                section.chunks = []
            section.chunks.extend(processed_chunks)
        chunk_time = time.perf_counter() - chunk_start
        logger.info(f"Chunk processing took {chunk_time:.2f} seconds")
        
        # Update chunk relationships
        relation_start = time.perf_counter()
        logger.info("Updating chunk relationships...")
        
        # Create a flat list of all chunks across all sections
        all_chunks = []
        for section in document.sections:
            all_chunks.extend(section.chunks)
        
        # Update relationships across all chunks
        for i, chunk in enumerate(all_chunks):
            if i > 0:
                chunk.previous_chunk_summary = all_chunks[i-1].summary.text
            if i < len(all_chunks) - 1:
                chunk.next_chunk_summary = all_chunks[i+1].summary.text
        
        relation_time = time.perf_counter() - relation_start
        logger.info(f"Chunk relationship updates took {relation_time:.2f} seconds")
        
        # Efficiently assign page numbers to chunks
        page_start = time.perf_counter()
        logger.info("Updating the page numbers to the chunks")
        assign_page_numbers_to_chunks(document)
        page_time = time.perf_counter() - page_start
        logger.info(f"Page number assignment took {page_time:.2f} seconds")
        
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
        }) for chunk in all_chunks]

        return chunks

import logging
import asyncio
import time
from typing import List, Optional
from .models import Document, Section, Chunk, Summary
from .utils import process_with_llm
from ai_chunking.utils.count_tokens import count_tokens
from ai_chunking.chunkers.section_based_semantic_chunker.semantic_chunker import create_semantic_splitter
from .prompts import (
    DOCUMENT_PROMPT,
    SECTION_PROMPT,
    CHUNK_PROMPT
)


logger = logging.getLogger(__name__)


async def process_sections(sections: List[str]) -> List[Section]:
    """Process sections in parallel"""
    async def process_section(text: str, idx: int) -> Section:
        logger.info(f"Processing section {idx}")
        prompt = SECTION_PROMPT.format(
            text=text,
        )
        response_json = await process_with_llm(prompt)
        title = response_json.get("title", "")
        summary_text = response_json.get("summary", "")
        metadata = response_json.get("metadata", {})
        tokens_count = count_tokens(text)

        return Section(
            section_id=idx,
            title=title,
            text=text,
            tokens_count=tokens_count,
            summary=Summary(text=summary_text, metadata=metadata)
        )
    
    tasks = [process_section(text, idx) for idx, text in enumerate(sections)]
    processed_sections = await asyncio.gather(*tasks)
    logger.info(f"Processed {len(processed_sections)} sections")
    return processed_sections

def create_chunks(text: str, min_chunk_size: int, max_chunk_size: Optional[int]) -> List[str]:
    """
    Create chunks using semantic chunking with optional size constraints.
    
    Args:
        text: The text to chunk
        min_chunk_size: Minimum chunk size in characters
        max_chunk_size: Maximum chunk size in tokens. If None, only semantic chunking 
                      is performed without size constraints.
                      
    Returns:
        List of text chunks
    """
    start_time = time.perf_counter()
    splitter = create_semantic_splitter(min_chunk_size=min_chunk_size, max_tokens=max_chunk_size)
    chunks = splitter.split_text(text)
    end_time = time.perf_counter()
    logger.info(f"Created {len(chunks)} chunks in {end_time - start_time:.2f} seconds")
    logger.info(f"Chunk sizes (tokens): {[count_tokens(chunk) for chunk in chunks]}")
    return chunks

async def process_chunks(
    chunks: List[str],
    section_summary: Summary,
    document_summary: Summary
) -> List[Chunk]:
    """Process chunks in parallel"""
    async def process_chunk(text: str, idx: int) -> Chunk:
        logger.info(f"Processing chunk {idx}")
        doc_summary_text = str(document_summary)
        prompt = CHUNK_PROMPT.format(
            text=text,
            section_summary=section_summary.text,
            document_summary=doc_summary_text
        )
        response_json = await process_with_llm(prompt)
        global_context = response_json.get("global_context", "")
        summary_text = response_json.get("summary", "")
        metadata = response_json.get("metadata", {})
        questions_this_excerpt_can_answer = response_json.get("questions_this_excerpt_can_answer", [])
        tokens_count = count_tokens(text)
        
        return Chunk(
            chunk_id=idx,
            text=text,
            tokens_count=tokens_count,
            page_numbers=[],  # This needs to be mapped based on content
            summary=Summary(text=summary_text, metadata=metadata),
            global_context=global_context,
            section_summary=section_summary,
            document_summary=document_summary,
            questions_this_excerpt_can_answer=questions_this_excerpt_can_answer
        )
    
    tasks = [process_chunk(text, idx) for idx, text in enumerate(chunks)]
    processed_chunks = await asyncio.gather(*tasks)
    logger.info(f"Processed {len(processed_chunks)} chunks")
    return processed_chunks

async def process_document_summary(section_summaries: List[Summary]) -> Summary:
    """Process document summary from section summaries"""
    logger.info("Processing complete document summary")
    # Combine all section summaries
    combined_summaries = "\n\n".join(summary.text for summary in section_summaries)
    
    # Process through LLM to get unified summary
    prompt = DOCUMENT_PROMPT.format(text=combined_summaries)
    response_json = await process_with_llm(prompt)
    summary_text = response_json.get("summary", "")
    metadata = response_json.get("metadata", {})
    
    return Summary(text=summary_text, metadata=metadata)


def assign_page_numbers_to_chunks(document: Document):
    """
    Efficiently assign page numbers to chunks using a positional index approach
    """
    # Step 1: Create a mapping of page boundaries in the merged document
    page_boundaries = {}
    current_position = 0
    merged_content = document.text
    
    # Find the position of each page in the merged content
    for page_num, page_text in sorted(document.page_map.items()):
        start_pos = merged_content.find(page_text, current_position)
        if start_pos != -1:
            end_pos = start_pos + len(page_text)
            page_boundaries[page_num] = (start_pos, end_pos)
            current_position = end_pos
    
    # Step 2: Create a mapping of chunks to their positions in the merged document
    chunk_positions = []
    for section in document.sections:
        for chunk in section.chunks:
            # Find the position of this chunk in the document
            # We use a fuzzy match since the chunk text might have been modified
            start_pos = find_chunk_position(chunk.text, merged_content)
            if start_pos != -1:
                chunk_positions.append((chunk, start_pos, start_pos + len(chunk.text)))
    
    # Step 3: Assign page numbers to chunks based on position overlap
    for chunk, chunk_start, chunk_end in chunk_positions:
        chunk.page_numbers = []
        for page_num, (page_start, page_end) in page_boundaries.items():
            # Check if chunk overlaps with this page
            if (chunk_start <= page_end and chunk_end >= page_start):
                chunk.page_numbers.append(page_num)
        
        # Sort page numbers
        chunk.page_numbers.sort()

def find_chunk_position(chunk_text: str, document_text: str, max_samples: int = 3) -> int:
    """
    Find the approximate position of a chunk in the document text.
    Uses a sampling approach to improve performance.
    """
    # If chunk is too short, just use a direct search
    if len(chunk_text) < 100:
        return document_text.find(chunk_text)
    
    # For longer chunks, sample a few distinctive phrases
    samples = []
    chunk_lines = [line for line in chunk_text.split('\n') if len(line.strip()) > 30]
    
    if not chunk_lines:
        return document_text.find(chunk_text[:100])
    
    # Take up to max_samples distinctive lines from the chunk
    sample_lines = []
    step = max(1, len(chunk_lines) // max_samples)
    for i in range(0, len(chunk_lines), step):
        if len(sample_lines) < max_samples:
            sample_lines.append(chunk_lines[i])
    
    # Find the position of each sample in the document
    positions = []
    for line in sample_lines:
        # Use a substring of the line to improve matching chances
        sample = line[:min(80, len(line))]
        pos = document_text.find(sample)
        if pos != -1:
            positions.append(pos)
    
    # Return the earliest position found, or -1 if none found
    return min(positions) if positions else -1

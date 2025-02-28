import asyncio
from typing import List, Dict
from pathlib import Path
import json
from dotenv import load_dotenv

load_dotenv()

from constants import MIN_CHUNK_SIZE, MIN_SECTION_CHUNK_SIZE, PAGE_CONTENT_SIMILARITY_THRESHOLD
from models import Document, Section, Chunk, Summary
from processors import (
    assign_page_numbers_to_chunks,
    process_sections,
    create_chunks,
    process_chunks,
    process_document_summary
)
from utils import load_markdown, merge_pages, extract_page_map, text_similarity_ratio, count_tokens
from logger_config import logger


async def create_document_chunks(markdown_path: Path) -> Document:
    """
    Main function to create semantic chunks from markdown document with parallel processing
    """
    # Load and prepare document
    content = load_markdown(markdown_path)
    page_map = extract_page_map(content)
    merged_content = merge_pages(content)
    initial_tokens_count = count_tokens(merged_content)
    
    document = Document(
        name=markdown_path.stem,
        text=merged_content,
        page_map=page_map,
        total_pages=len(page_map),
        tokens_count=initial_tokens_count, 
    )
    
    # Create and process sections
    logger.info("Creating and processing sections...")
    sections = create_chunks(document.text, MIN_SECTION_CHUNK_SIZE)
    processed_sections = await process_sections(sections)
    document.sections = processed_sections
    
    # Create document summary
    logger.info("Creating document summary...")
    document.summary = await process_document_summary([section.summary for section in document.sections])

    # Process chunks for all sections in parallel
    logger.info("Processing chunks for all sections...")
    chunk_tasks = []
    for section in document.sections:
        chunks = create_chunks(section.text, MIN_CHUNK_SIZE)
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
    
    # Update chunk relationships
    logger.info("Updating chunk relationships...")
    for section in document.sections:
        for i, chunk in enumerate(section.chunks):
            if i > 0:
                chunk.previous_chunk_summary = section.chunks[i-1].summary.text
            if i < len(section.chunks) - 1:
                chunk.next_chunk_summary = section.chunks[i+1].summary.text
    
    # Efficiently assign page numbers to chunks
    logger.info("Updating the page numbers to the chunks")
    assign_page_numbers_to_chunks(document)
    logger.info(f"Document processed: {document.name}, {document.total_pages} pages, {document.tokens_count} tokens")
    
    return document

async def main():
    input_path = Path("/Users/sakshammittal/Documents/GitHub/temp/pdf-parser/output/marker/oaktree/ocr/Summary of Fund Terms_Power Opportunities Fund VI_LP/Summary of Fund Terms_Power Opportunities Fund VI_LP.md")
    output_path = Path("/Users/sakshammittal/Documents/GitHub/temp/pdf-parser/output/processed_chunks/oaktree/Summary of Fund Terms_Power Opportunities Fund VI_LP.json")
    
    logger.info(f"Processing {input_path}...")
    document = await create_document_chunks(input_path)
    
    logger.info(f"Saving results to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save_json(output_path)
    
    logger.info(f"Chunks saved at {output_path}")

if __name__ == "__main__":
    asyncio.run(main())

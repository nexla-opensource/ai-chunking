import asyncio
from typing import List, Dict
from pathlib import Path
import json
from dotenv import load_dotenv
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

load_dotenv()

from constants import MAX_CHUNK_SIZE, MAX_SECTION_CHUNK_SIZE, MIN_CHUNK_SIZE, MIN_SECTION_CHUNK_SIZE
from models import Document
from processors import (
    assign_page_numbers_to_chunks,
    process_sections,
    create_chunks,
    process_chunks,
    process_document_summary
)
from utils import load_markdown, merge_pages, extract_page_map, count_tokens
from logger_config import logger


async def create_document_chunks(markdown_path: Path) -> Document:
    """
    Main function to create semantic chunks from markdown document with parallel processing
    """
    start_time = time.perf_counter()
    
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
    
    return document

async def process_single_folder(folder: Path, output_dir: Path) -> tuple[int, int, int]:
    """
    Process a single folder and return processing status (processed, skipped, failed)
    """
    start_time = time.perf_counter()
    
    # Find markdown file in the folder
    markdown_files = list(folder.glob("*.md"))
    if not markdown_files:
        logger.warning(f"No markdown file found in {folder}")
        return 0, 0, 1
            
    markdown_file = markdown_files[0]  # Take the first markdown file
    output_file = output_dir / f"{folder.name}.json"
    
    # Check if file has already been processed
    if output_file.exists():
        logger.info(f"Skipping {markdown_file} as output already exists at {output_file}")
        return 0, 1, 0
            
    logger.info(f"Processing {markdown_file}...")
    
    try:
        # Process the document
        document = await create_document_chunks(markdown_file)
        
        # Save the document
        logger.info(f"Saving results to {output_file}...")
        document.save_json(output_file)
        
        process_time = time.perf_counter() - start_time
        logger.info(f"Successfully processed {markdown_file} in {process_time:.2f} seconds")
        return 1, 0, 0
    except Exception as e:
        logger.error(f"Error processing {markdown_file}: {str(e)}")
        return 0, 0, 1

async def process_directory(input_dir: Path, output_dir: Path):
    """
    Process all markdown files in the input directory and save results in the output directory.
    Processes folders in parallel with a limit of 5 at a time.
    """
    start_time = time.perf_counter()
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all subdirectories in input directory
    input_folders = [d for d in input_dir.iterdir() if d.is_dir()]
    total_files = len(input_folders)
    
    # Initialize counters
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    # Create a counter for progress tracking and a lock to safely update it
    completed_count = 0
    progress_lock = asyncio.Lock()
    
    # Create a semaphore to limit concurrent processing to 5 folders at a time
    semaphore = asyncio.Semaphore(5)
    
    async def process_folder_with_semaphore(folder):
        nonlocal completed_count
        
        async with semaphore:
            try:
                logger.info(f"\nProcessing folder: {folder.name}")
                result = await process_single_folder(folder, output_dir)
                
                # Update progress counter with lock to avoid race conditions
                async with progress_lock:
                    completed_count += 1
                    logger.info(f"Progress: {completed_count}/{total_files} folders completed")
                    
                return result
            except Exception as e:
                logger.error(f"Unexpected error processing folder {folder.name}: {str(e)}")
                
                # Update progress counter even if there's an error
                async with progress_lock:
                    completed_count += 1
                    logger.info(f"Progress: {completed_count}/{total_files} folders completed (with error)")
                
                # Return failed status
                return 0, 0, 1
    
    # Create tasks for all folders
    tasks = [process_folder_with_semaphore(folder) for folder in input_folders]
    
    # Process all tasks and collect results
    results = await asyncio.gather(*tasks)
    
    # Aggregate results
    for processed, skipped, failed in results:
        processed_count += processed
        skipped_count += skipped
        failed_count += failed
    
    # Log final statistics
    logger.info(f"\nProcessing complete! Summary:")
    total_time = time.perf_counter() - start_time
    logger.info(f"Total files: {total_files}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Skipped (already existed): {skipped_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Total directory processing time: {total_time:.2f} seconds")

async def main():
    input_dir = Path("/Users/sakshammittal/Documents/Nexla OpenSource GitHub/ai-chunking/temp/input/financebench")
    output_dir = Path("/Users/sakshammittal/Documents/Nexla OpenSource GitHub/ai-chunking/temp/output/financebench")
    
    logger.info(f"Starting to process files from {input_dir}")
    await process_directory(input_dir, output_dir)
    logger.info("Processing complete!")

if __name__ == "__main__":
    asyncio.run(main())

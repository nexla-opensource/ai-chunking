"""Main document processor for AI-powered document chunking."""

import asyncio
import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple, Any

from ai_chunking.chunkers.auto_ai_chunker.chunk_builder import ChunkBuilder
from ai_chunking.chunkers.auto_ai_chunker.text_enricher import TextEnricher
from ai_chunking.llm.base import StructuredLLMClient
from ai_chunking.chunkers.auto_ai_chunker.models.document import Page, ProcessedChunk, TableData, EnrichedPageData, ContentType, Heading, HeadingType
from ai_chunking.models.chunk import Chunk
from ai_chunking.utils.json_utils import parse_json_response

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Main class for processing documents into semantic chunks."""

    def __init__(
        self,
        llm_client: StructuredLLMClient,
        small_llm_client: Optional[StructuredLLMClient] = None,
        large_llm_client: Optional[StructuredLLMClient] = None,
        token_counter = None
    ):
        """Initialize document processor.
        
        Args:
            llm_client: Default LLM client for text processing (used if specific clients not provided)
            small_llm_client: Optional smaller/faster LLM client for simpler tasks
            large_llm_client: Optional larger/smarter LLM client for complex tasks
            token_counter: Optional function to count tokens (if not provided, uses llm_client)
        """
        self.small_llm_client = small_llm_client or llm_client
        self.large_llm_client = large_llm_client or llm_client
        self.llm_client = llm_client
        self.token_counter = token_counter or llm_client.count_tokens
        
        self.enricher = TextEnricher(self.small_llm_client)
        self.chunk_builder = ChunkBuilder(self.large_llm_client)

    async def _process_page(
        self,
        page: Page,
        table_data_map: Dict,
        customer_tags: List[str],
        source: str
    ) -> EnrichedPageData:
        """Process a single page asynchronously.
        
        Args:
            page: Page to process
            table_data_map: Map of page numbers to table data
            customer_tags: Customer-specific tags
            source: Source document identifier
            
        Returns:
            Enriched page data
        """
        try:
            # Get table data for this page
            page_tables = table_data_map.get(page.page_number, [])
            
            # Convert rows to TableData objects if they aren't already
            table_data_objects = []
            for table_rows in page_tables:
                if isinstance(table_rows, list):
                    table_data_objects.append(TableData(
                        page_number=page.page_number,
                        rows=table_rows,
                        caption=""
                    ))
                else:
                    table_data_objects.append(table_rows)
            
            # Enrich page with tables
            enriched_page = await self.enricher.enrich_page(
                page,
                table_data_objects
            )
            logger.debug(f"Enriched page {page.page_number}")
            return enriched_page
            
        except Exception as e:
            logger.error(f"Error enriching page {page.page_number}: {str(e)}")
            raise

    async def process_document(
        self,
        pages: List[Page],
        table_data: Optional[List[TableData]] = None,
        metadata: Optional[Dict] = None,
        customer_tags: Optional[List[str]] = None,
        source: str = ""
    ) -> List[Chunk]:
        """Process a document into semantic chunks using async processing.
        
        Args:
            pages: Document pages to process
            table_data: Optional table data to incorporate
            customer_tags: Optional customer-specific tags
            source: Source document identifier
            
        Returns:
            List of processed chunks
            
        Raises:
            ValueError: If pages list is empty
        """
        if not pages:
            raise ValueError("No pages provided")
            
        logger.info(f"Processing document with {len(pages)} pages")

        # Prepare table data map
        table_data_map = {}
        if table_data:
            for table in table_data:
                page_number = table.page_number
                if page_number not in table_data_map:
                    table_data_map[page_number] = []
                table_data_map[page_number].append(table)
        
        # Process pages concurrently using asyncio
        page_tasks = [
            self._process_page(page, table_data_map, customer_tags or [], source)
            for page in pages
        ]
        enriched_pages = await asyncio.gather(*page_tasks)
        logger.info(f"Completed enriching {len(enriched_pages)} pages")

        # Extract universal headings (single LLM call)
        candidate_headings = self._get_candidate_headings(enriched_pages)
        universal_headings = await self._get_universal_headings(candidate_headings, source)

        # Process bridge headings (batched LLM calls)
        enriched_pages = await self._process_bridge_headings_batch(enriched_pages, universal_headings)
        
        # Build chunks concurrently
        chunk_tasks = []
        for enriched_page in enriched_pages:
            task = self.chunk_builder.build_chunks(
                enriched_page,
                customer_tags,
                universal_headings
            )
            chunk_tasks.append(task)
            
        # Wait for all chunk tasks
        chunk_results = await asyncio.gather(*chunk_tasks)
        
        # Process chunks and count tokens in parallel
        async def process_chunk(chunk, enriched_page):
            # Get chunk text considering gap ranges
            chunk_text = self._get_chunk_text(chunk, enriched_page.enriched_text_lines_map)
            
            # Format chunk text with proper structure
            formatted_text = self._format_chunk_text(
                content=chunk_text,
                title=chunk.title,
                universal_headings=universal_headings,
                parent_title=chunk.parent_title,
                sub_titles=chunk.sub_titles,
                summary=chunk.summary,
                customer_specific_tags=customer_tags
            )
            
            # Count tokens
            token_count = await self.token_counter(formatted_text)
            
            return ProcessedChunk(
                text=formatted_text,
                page_number=enriched_page.page_number,
                content_type=chunk.content_type,
                tokens=token_count,
                summary=chunk.summary,
                parent_heading=chunk.parent_title,
                headings=[h for h in enriched_page.headings if chunk.start_index <= h.line_number <= chunk.end_index],
                tables=enriched_page.tables,
                metadata=self._build_chunk_metadata(chunk, customer_tags, universal_headings, metadata)
            )
        
        # Process all chunks in parallel batches
        BATCH_SIZE = 50  # Process 50 chunks at a time
        all_chunks = []
        
        for page_chunks, enriched_page in zip(chunk_results, enriched_pages):
            for i in range(0, len(page_chunks), BATCH_SIZE):
                batch = page_chunks[i:i + BATCH_SIZE]
                processed_chunks = await asyncio.gather(*[
                    process_chunk(chunk, enriched_page)
                    for chunk in batch
                ])
                all_chunks.extend(processed_chunks)
        
        # Post-process: merge small chunks if needed
        try:
            processed_chunks = await self._merge_small_chunks(all_chunks)
            logger.info(
                f"Successfully merged chunks. Final count: {len(processed_chunks)} chunks"
            )
        except Exception as e:
            logger.error(f"Error merging small chunks: {str(e)}")
            raise RuntimeError(f"Failed to merge chunks: {str(e)}") from e

        logger.info(
            f"Document processing complete. Created {len(processed_chunks)} chunks"
        )
        chunks = [Chunk(
            text=chunk.text,
            metadata={
                "page_number": chunk.page_number,
                "content_type": chunk.content_type,
                "tokens": chunk.tokens,
                "summary": chunk.summary,
                "parent_heading": chunk.parent_heading,
                "headings": chunk.headings,
                "tables": chunk.tables
            }
        ) for chunk in processed_chunks]
        return chunks

    def _get_chunk_text(self, chunk, enriched_text_lines_map: Dict[int, str]) -> str:
        """Get chunk text considering gap ranges."""
        excluded_lines = set()
        for gap_start, gap_end in chunk.gap_index_range:
            for ln in range(gap_start, gap_end + 1):
                excluded_lines.add(ln)

        lines = []
        for ln in range(chunk.start_index, chunk.end_index + 1):
            if ln not in excluded_lines and ln in enriched_text_lines_map:
                lines.append(enriched_text_lines_map[ln])

        return "\n".join(lines)

    def _build_chunk_metadata(
        self,
        chunk,
        customer_tags: Optional[List[str]] = None,
        universal_headings: Optional[List[str]] = None,
        metadata: Dict = None
    ) -> Dict[str, str]:
        """Build chunk metadata."""
        metadata = {} if metadata is None else metadata
        
        if universal_headings:
            metadata["universal_headings"] = ", ".join(str(h) for h in universal_headings)
            
        if customer_tags:
            metadata["customer_tags"] = ", ".join(str(t) for t in customer_tags)
            
        if chunk.parent_title:
            metadata["parent_section"] = str(chunk.parent_title)
            
        if chunk.sub_titles:
            metadata["sub_sections"] = ", ".join(str(t) for t in filter(None, chunk.sub_titles))
            
        return metadata

    def _get_candidate_headings(self, enriched_pages: List[EnrichedPageData]) -> Dict:
        """Extract candidate headings from enriched pages."""
        all_headings = [
            heading
            for page in enriched_pages
            for heading in page.headings
        ]
        return {
            "global_headings": [
                h for h in all_headings
                if hasattr(h, 'heading_type') and h.heading_type == "global_heading"
            ],
            "footer_headings": [
                h for h in all_headings
                if hasattr(h, 'heading_type') and h.heading_type == "footer_heading"
            ]
        }

    async def _get_universal_headings(self, candidate_headings: Dict, source: str) -> List[str]:
        """Get universal headings from candidate headings."""
        from .prompts import global_headings_prompt
        
        # Convert lists to tuples for caching
        global_headings_tuple = tuple(candidate_headings.get("global_headings", []))
        footer_headings_tuple = tuple(candidate_headings.get("footer_headings", []))
        
        universal_headings_prompt = global_headings_prompt(
            global_headings_tuple,
            footer_headings_tuple,
            source
        )
        universal_headings_response = await self.large_llm_client.complete(universal_headings_prompt)
        if not universal_headings_response:
            return []
            
        try:
            return [
                uh.get("title", None)
                for uh in parse_json_response(universal_headings_response, key="universal_headings", default=[])
            ]
        except Exception:
            logger.warning("Failed to parse universal headings response")
            return []

    async def _process_bridge_headings_batch(
        self,
        enriched_pages: List[EnrichedPageData],
        universal_headings: List[str]
    ) -> List[EnrichedPageData]:
        """Process bridge headings in batches."""
        from .prompts import bridge_headings_prompt
        
        # Find candidate pairs
        pairs = []
        for i in range(len(enriched_pages) - 1):
            curr_page = enriched_pages[i]
            next_page = enriched_pages[i + 1]
            
            if next_page.headings and next_page.headings[0].line_number > 1:
                pairs.append((curr_page, next_page))
                
        if not pairs:
            return enriched_pages
            
        # Prepare prompts for all pairs
        prompts = []
        for curr_page, next_page in pairs:
            if not curr_page.headings or not next_page.headings:
                continue
                
            curr_last_heading = curr_page.headings[-1]
            next_first_heading = next_page.headings[0]
            
            # Get text under last heading
            curr_last_text = {
                k: v
                for k, v in curr_page.enriched_text_lines_map.items()
                if k >= curr_last_heading.line_number
            }
            
            # Get text before first heading
            next_first_text = {
                k: v
                for k, v in next_page.enriched_text_lines_map.items()
                if k < next_first_heading.line_number
            }
            
            prompt = bridge_headings_prompt(
                curr_last_heading.text,
                curr_last_text,
                next_first_text,
                curr_page.headings,
                universal_headings
            )
            prompts.append((prompt, curr_page, next_page))
            
        # Batch process prompts
        responses = await asyncio.gather(*[
            self.small_llm_client.complete(prompt)
            for prompt, _, _ in prompts
        ])
        
        # Process responses
        for response, (_, curr_page, next_page) in zip(responses, prompts):
            try:
                bridge_data = parse_json_response(response, default={})
                
                if bridge_data.get("bridging") == "true":
                    # Use current page's last heading
                    curr_last_heading = curr_page.headings[-1]
                    next_page.headings.insert(0, Heading(
                        text=curr_last_heading.text,
                        level=curr_last_heading.level,
                        line_number=1,
                        is_bridge=True,
                        heading_type=HeadingType.BRIDGE,
                        page_number=next_page.page_number
                    ))
                elif bridge_data.get("bridging") == "relevant":
                    # Use most relevant headings
                    next_page.headings.insert(0, Heading(
                        text=", ".join(bridge_data["most_relevant_headings"]),
                        level=curr_page.headings[-1].level,
                        line_number=1,
                        is_bridge=True,
                        heading_type=HeadingType.BRIDGE,
                        page_number=next_page.page_number
                    ))
            except Exception:
                logger.warning(f"Failed to process bridge response for page {next_page.page_number}")
                
        return enriched_pages

    async def _merge_small_chunks(
        self,
        chunks: List[ProcessedChunk]
    ) -> List[ProcessedChunk]:
        """Merge small chunks based on token count."""
        if not chunks:
            logger.debug("No chunks to merge")
            return []
            
        if len(chunks) == 1:
            return chunks
            
        logger.debug(f"Starting merge of {len(chunks)} chunks")
        merged_chunks = []
        idx = 0
        
        while idx < len(chunks):
            current_chunk = chunks[idx]
            
            try:
                logger.debug(f"Processing chunk {idx}: current(tokens={current_chunk.tokens}, type={current_chunk.content_type})")
                
                if current_chunk.tokens < 10:
                    if idx == 0 and len(chunks) > 1:
                        # Merge first and second chunk
                        next_chunk = chunks[idx + 1]
                        merged_text = f"{current_chunk.text}\n{next_chunk.text}"
                        merged_tokens = await self.token_counter(merged_text)
                        
                        merged_chunk = ProcessedChunk(
                            text=merged_text,
                            page_number=current_chunk.page_number,
                            content_type=current_chunk.content_type,
                            tokens=merged_tokens,
                            summary=f"{current_chunk.summary}\n{next_chunk.summary}".strip(),
                            parent_heading=current_chunk.parent_heading,
                            headings=current_chunk.headings + next_chunk.headings,
                            tables=current_chunk.tables + next_chunk.tables,
                            metadata=self._merge_metadata(current_chunk.metadata, next_chunk.metadata)
                        )
                        merged_chunks.append(merged_chunk)
                        idx += 2  # Skip next chunk as it's merged
                        
                    elif idx == len(chunks) - 1 and len(merged_chunks) > 0:
                        # Merge last chunk with previous chunk
                        prev_chunk = merged_chunks.pop()  # Remove last added chunk
                        merged_text = f"{prev_chunk.text}\n{current_chunk.text}"
                        merged_tokens = await self.token_counter(merged_text)
                        
                        merged_chunk = ProcessedChunk(
                            text=merged_text,
                            page_number=prev_chunk.page_number,
                            content_type=prev_chunk.content_type,
                            tokens=merged_tokens,
                            summary=f"{prev_chunk.summary}\n{current_chunk.summary}".strip(),
                            parent_heading=prev_chunk.parent_heading,
                            headings=prev_chunk.headings + current_chunk.headings,
                            tables=prev_chunk.tables + current_chunk.tables,
                            metadata=self._merge_metadata(prev_chunk.metadata, current_chunk.metadata)
                        )
                        merged_chunks.append(merged_chunk)
                        idx += 1
                        
                    elif idx < len(chunks) - 1:
                        # Merge current with next chunk
                        next_chunk = chunks[idx + 1]
                        merged_text = f"{current_chunk.text}\n{next_chunk.text}"
                        merged_tokens = await self.token_counter(merged_text)
                        
                        merged_chunk = ProcessedChunk(
                            text=merged_text,
                            page_number=current_chunk.page_number,
                            content_type=current_chunk.content_type,
                            tokens=merged_tokens,
                            summary=f"{current_chunk.summary}\n{next_chunk.summary}".strip(),
                            parent_heading=current_chunk.parent_heading,
                            headings=current_chunk.headings + next_chunk.headings,
                            tables=current_chunk.tables + next_chunk.tables,
                            metadata=self._merge_metadata(current_chunk.metadata, next_chunk.metadata)
                        )
                        merged_chunks.append(merged_chunk)
                        idx += 2  # Skip next chunk as it's merged
                    else:
                        merged_chunks.append(current_chunk)
                        idx += 1
                else:
                    merged_chunks.append(current_chunk)
                    idx += 1
                    
            except Exception as e:
                logger.error(f"Error merging chunks {idx}: {str(e)}")
                # On error, add current chunk and continue
                merged_chunks.append(current_chunk)
                idx += 1
        
        return merged_chunks

    def _merge_metadata(self, metadata1: Dict[str, str], metadata2: Dict[str, str]) -> Dict[str, str]:
        """Merge metadata from two chunks safely."""
        merged = {}
        
        # Helper function to safely convert metadata values to strings
        def safe_str_convert(value) -> str:
            if isinstance(value, (list, tuple)):
                return ", ".join(str(v) for v in value)
            return str(value)
        
        # Merge first chunk metadata
        for key, value in metadata1.items():
            merged[str(key)] = safe_str_convert(value)
        
        # Merge second chunk metadata
        for key, value in metadata2.items():
            key = str(key)
            value = safe_str_convert(value)
            if key in merged:
                merged[key] = f"{merged[key]}, {value}"
            else:
                merged[key] = value
                
        return merged

    def _format_chunk_text(
        self,
        content: str,
        title: str,
        universal_headings: List[str],
        parent_title: Optional[str] = None,
        sub_titles: Optional[List[str]] = None,
        summary: Optional[str] = None,
        customer_specific_tags: Optional[List[str]] = None,
    ) -> str:
        """
        Formats chunk text with structural context for RAG.

        Args:
            content: The main text content of the chunk
            title: The title of the current section
            universal_headings: List of universal headings for the document
            parent_title: Optional parent section title
            sub_titles: Optional list of sub-section titles
            summary: Optional summary text
            customer_specific_tags: Optional customer-specific tags

        Returns:
            Formatted chunk text with structural context
        """
        # Clean and validate inputs
        cleaned_universal = [
            h for h in universal_headings 
            if h and str(h).strip() and str(h) not in ("None", "N/A")
        ]
        cleaned_subs = [
            s for s in (sub_titles or []) 
            if s and str(s).strip() and str(s) not in ("None", "N/A")
        ]

        # Build the formatted text
        formatted_parts = []

        # Add universal headings if present
        if cleaned_universal:
            formatted_parts.append(f"UNIVERSAL: {' | '.join(cleaned_universal)}")

        # Add parent section if present
        if parent_title and str(parent_title) not in ("None", "N/A"):
            formatted_parts.append(f"SECTION: {parent_title}")

        # Add current topic
        formatted_parts.append(f"TOPIC: {title}")

        # Add sub-topics if present
        if cleaned_subs:
            formatted_parts.append(f"SUB-TOPICS: {', '.join(cleaned_subs)}")

        # Add customer specific tags
        if customer_specific_tags:
            for tag in customer_specific_tags:
                formatted_parts.append(str(tag))

        # Add content
        formatted_parts.append("\nCONTENT:")
        formatted_parts.append(content)

        # Add summary if present
        if summary and summary.strip():
            formatted_parts.append(f"\nSUMMARY: {summary}")

        # Join all parts with newlines
        return "\n".join(formatted_parts)

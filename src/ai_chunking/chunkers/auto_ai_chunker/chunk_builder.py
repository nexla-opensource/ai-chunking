import logging
import asyncio
from typing import Dict, List, Optional

from ai_chunking.llm.base import StructuredLLMClient, LLMError
from ai_chunking.chunkers.auto_ai_chunker.models.document import Chunk, ContentType, EnrichedPageData
from ai_chunking.chunkers.auto_ai_chunker.models.llm_responses import ContentAnalysis, Section, SemanticGroupingResponse
from ai_chunking.chunkers.auto_ai_chunker.prompts import semantic_grouping_prompt

logger = logging.getLogger(__name__)

def must_be_separate_chunk(content_type: str, start_line: int, end_line: int) -> bool:
    """
    Decide if a sub-section must be a separate chunk based on rules:
      - If content_type ∈ [Metrics, Instructions, Concept Definition] => forced separate
      - If content_type ∈ [Narrative, Comparisons, Legal] and line_count > 10 => forced separate
    """
    if content_type in [ContentType.METRICS, ContentType.INSTRUCTIONS, ContentType.CONCEPT]:
        return True
    elif content_type in [ContentType.NARRATIVE, ContentType.COMPARISONS, ContentType.LEGAL]:
        line_count = end_line - start_line + 1
        if line_count > 10:
            return True
    return False

def carve_out_subsections(section: Section, section_start: int, section_end: int) -> List[Chunk]:
    """
    Create chunks for a section with subsections, handling forced separations.
    """
    chunks = []
    sub_sections = section.sub_sections
    
    if not sub_sections:
        return chunks
        
    # Sort subsections by start index
    sub_sections.sort(key=lambda s: s.start_index)
    
    # Handle gap between section start and first subsection
    first_sub = sub_sections[0]
    if first_sub.start_index > section_start:
        # Create chunk for content before first subsection
        if must_be_separate_chunk(section.content_type, section_start, first_sub.start_index - 1):
            chunks.append(
                Chunk(
                    parent_title=None,
                    title=section.title,
                    sub_titles=[],
                    content_type=section.content_type,
                    start_index=section_start,
                    end_index=first_sub.start_index - 1,
                    summary=section.summary,
                    page_number=getattr(section, 'page_number', 1)
                )
            )
    
    # Process each subsection
    for i, sub in enumerate(sub_sections):
        sub_start = sub.start_index
        
        # Determine end index for this subsection
        if i < len(sub_sections) - 1:
            sub_end = sub_sections[i + 1].start_index - 1
        else:
            sub_end = section_end
            
        if sub_end < sub_start:
            sub_end = sub_start
            
        # Create chunk for this subsection
        chunks.append(
            Chunk(
                parent_title=section.title,
                title=sub.title,
                sub_titles=[],
                content_type=sub.content_type,
                start_index=sub_start,
                end_index=sub_end,
                summary=sub.summary,
                page_number=getattr(section, 'page_number', 1)
            )
        )
    
    return chunks

def merge_single_line_chunks(chunks: List[Chunk], total_number_of_lines: int) -> List[Chunk]:
    """
    Merge any chunk that covers exactly line 1..1 with the next chunk,
    and any chunk that covers exactly line total_number_of_lines..total_number_of_lines
    with the previous chunk.
    """
    if not chunks:
        return chunks

    chunks.sort(key=lambda c: c.start_index)
    merged: List[Chunk] = []
    i = 0
    while i < len(chunks):
        ch = chunks[i]
        single_line = ch.start_index == ch.end_index

        # if chunk is exactly 1..1, merge with next if possible
        if single_line and ch.start_index == 1 and i < len(chunks) - 1:
            nxt = chunks[i + 1]
            # unify coverage from 1..nxt.end_index
            new_chunk = Chunk(
                parent_title=nxt.parent_title,
                title=nxt.title,
                sub_titles=nxt.sub_titles,
                content_type=nxt.content_type,
                start_index=1,
                end_index=nxt.end_index,
                summary=nxt.summary,
                page_number=nxt.page_number
            )
            merged.append(new_chunk)
            i += 2
            continue

        # if chunk is exactly last_line..last_line => merge with prev
        if (
            single_line
            and ch.start_index == total_number_of_lines == ch.end_index
            and len(merged) > 0
        ):
            prev = merged[-1]
            new_chunk = Chunk(
                parent_title=prev.parent_title,
                title=prev.title,
                sub_titles=prev.sub_titles,
                content_type=prev.content_type,
                start_index=prev.start_index,
                end_index=total_number_of_lines,
                summary=prev.summary,
                page_number=prev.page_number
            )
            merged[-1] = new_chunk
            i += 1
            continue

        merged.append(ch)
        i += 1

    return merged

def fill_in_end_indices(sections: List[Section], total_number_of_lines: int) -> None:
    """
    Computes and assigns 'end_index' for each top-level section and each sub-section.
    This modifies the input 'sections' in place.
    """
    sections.sort(key=lambda s: s.start_index)
    num_sections = len(sections)

    for i, section in enumerate(sections):
        sec_start = section.start_index

        # Compute the top-level section end
        if i < num_sections - 1:
            next_start = sections[i + 1].start_index
            sec_end = max(sec_start, next_start - 1)
        else:
            sec_end = total_number_of_lines

        if sec_end < sec_start:
            sec_end = sec_start

        section.end_index = sec_end

        # Sort sub-sections by start_index if present
        sub_secs = section.sub_sections
        sub_secs.sort(key=lambda s: s.start_index)

        for j, sub_sec in enumerate(sub_secs):
            sub_start = sub_sec.start_index
            if j < len(sub_secs) - 1:
                nxt_sub_start = sub_secs[j + 1].start_index
                sub_end = max(sub_start, nxt_sub_start - 1)
            else:
                sub_end = sec_end

            if sub_end < sub_start:
                sub_end = sub_start

            sub_sec.end_index = sub_end

def create_chunks_from_structured_document(sections: List[Section], total_number_of_lines: int) -> List[Chunk]:
    """
    Create chunks from a structured document representation.
    Following the simpler chunking strategy:
    1. Fill in end indices for sections
    2. Create chunks based on content type and line count
    3. Handle merging of small chunks
    """
    fill_in_end_indices(sections, total_number_of_lines)
    all_chunks: List[Chunk] = []

    for sec in sections:
        section_start = sec.start_index
        section_end = getattr(sec, 'end_index', total_number_of_lines)
        sub_secs = sec.sub_sections
        
        if not sub_secs:
            # If no sub-sections, create a single chunk for the entire section
            c = Chunk(
                parent_title=None,
                title=sec.title,
                sub_titles=[],
                content_type=sec.content_type,
                start_index=section_start,
                end_index=section_end,
                summary=sec.summary,
                page_number=getattr(sec, 'page_number', 1)
            )
            all_chunks.append(c)
            continue

        # Carve out sub-sections and handle forced separations
        carved_chunks = carve_out_subsections(sec, section_start, section_end)
        all_chunks.extend(carved_chunks)

    # Merge single-line chunks at first line or last line
    merged_chunks = merge_single_line_chunks(all_chunks, total_number_of_lines)
    
    # Further merge small chunks based on token count
    final_chunks = merge_chunks_by_tokens(merged_chunks)
    
    # Sort final result by start_index
    final_chunks.sort(key=lambda c: c.start_index)

    return final_chunks

def merge_chunks_by_tokens(chunks: List[Chunk]) -> List[Chunk]:
    """
    Merge small chunks based on token count and content type.
    More aggressive merging strategy that considers both line count and content type.
    """
    if not chunks:
        return chunks
        
    chunks.sort(key=lambda c: c.start_index)
    merged: List[Chunk] = []
    i = 0
    
    while i < len(chunks):
        current = chunks[i]
        line_count = current.end_index - current.start_index + 1
        
        # If current chunk is small (< 10 lines) and there's a next chunk
        if line_count < 10 and i < len(chunks) - 1:
            next_chunk = chunks[i + 1]
            
            # Only merge if same content type
            if current.content_type == next_chunk.content_type:
                # Create merged chunk
                merged_chunk = Chunk(
                    parent_title=current.parent_title,
                    title=f"{current.title} & {next_chunk.title}",
                    content_type=current.content_type,
                    start_index=current.start_index,
                    end_index=next_chunk.end_index,
                    summary=f"{current.summary}\n{next_chunk.summary}".strip(),
                    sub_titles=current.sub_titles + next_chunk.sub_titles,
                    gap_index_range=current.gap_index_range + next_chunk.gap_index_range,
                    page_number=current.page_number
                )
                merged.append(merged_chunk)
                i += 2  # Skip next chunk since we merged it
                continue
            elif i == 0 and len(chunks) > 1:
                # Merge first and second chunk even if different content types
                merged_chunk = Chunk(
                    parent_title=next_chunk.parent_title,
                    title=f"{current.title} & {next_chunk.title}",
                    content_type=next_chunk.content_type,
                    start_index=current.start_index,
                    end_index=next_chunk.end_index,
                    summary=f"{current.summary}\n{next_chunk.summary}".strip(),
                    sub_titles=current.sub_titles + next_chunk.sub_titles,
                    gap_index_range=current.gap_index_range + next_chunk.gap_index_range,
                    page_number=current.page_number
                )
                merged.append(merged_chunk)
                i += 2  # Skip next chunk since we merged it
                continue
            elif i == len(chunks) - 1 and len(merged) > 0:
                # Merge last chunk with previous chunk even if different content types
                prev_chunk = merged.pop()
                merged_chunk = Chunk(
                    parent_title=prev_chunk.parent_title,
                    title=f"{prev_chunk.title} & {current.title}",
                    content_type=prev_chunk.content_type,
                    start_index=prev_chunk.start_index,
                    end_index=current.end_index,
                    summary=f"{prev_chunk.summary}\n{current.summary}".strip(),
                    sub_titles=prev_chunk.sub_titles + current.sub_titles,
                    gap_index_range=prev_chunk.gap_index_range + current.gap_index_range,
                    page_number=prev_chunk.page_number
                )
                merged.append(merged_chunk)
                i += 1
                continue
                
        merged.append(current)
        i += 1
    
    return merged

def get_chunk_text(chunk: Chunk, enriched_text_lines_map: Dict[int, str]) -> str:
    """
    Returns the concatenated text for the chunk from chunk.start_index..chunk.end_index,
    EXCLUDING any line ranges in chunk.gap_index_range.
    Ensures all non-gap lines are included and properly formatted.
    """
    excluded_lines = set()
    # Convert each gap range (gap_start..gap_end) to line numbers
    for gap_start, gap_end in chunk.gap_index_range:
        for ln in range(gap_start, gap_end + 1):
            excluded_lines.add(ln)

    # Build text lines skipping excluded lines
    lines = []
    for ln in range(chunk.start_index, chunk.end_index + 1):
        if ln not in excluded_lines:
            # If the line exists in enriched_text_lines_map, append it
            if ln in enriched_text_lines_map:
                line_text = enriched_text_lines_map[ln].strip()
                if line_text:  # Only add non-empty lines
                    lines.append(line_text)
            else:
                # Log missing line for debugging
                logger.debug(f"Missing line {ln} in chunk {chunk.title}")

    # Join lines with proper spacing
    return "\n".join(lines)

class ChunkBuilder:
    """Builds semantic chunks from enriched document pages."""

    def __init__(self, llm_client: StructuredLLMClient):
        """Initialize chunk builder.
        
        Args:
            llm_client: LLM client for semantic analysis
        """
        self.llm_client = llm_client

    async def build_chunks(
        self,
        page: EnrichedPageData,
        customer_tags: Optional[List[str]] = None,
        universal_headings: Optional[List[str]] = None
    ) -> List[Chunk]:
        """Build semantic chunks from an enriched page using LLM-based semantic grouping.

        Args:
            page: Enriched page data
            customer_tags: Optional customer-specific tags
            universal_headings: Optional universal headings

        Returns:
            List of semantic chunks
        """
        try:
            logger.info(f"Building chunks for page {page.page_number}")
            total_lines = len(page.enriched_text_lines_map)

            # Build enriched text with line numbers
            enriched_text_with_line_numbers = ""
            for i in range(1, total_lines + 1):
                line = page.enriched_text_lines_map.get(i, "")
                enriched_text_with_line_numbers += f"{i}. {line}\n"

            # Convert headings to the expected format
            headings_list = []
            for heading in page.headings:
                heading_dict = {
                    "title": heading.text,
                    "start_index": heading.line_number,
                    "heading_type": heading.heading_type,
                    "page_number": page.page_number
                }
                headings_list.append(heading_dict)

            # Generate semantic grouping prompt
            prompt = semantic_grouping_prompt(
                enriched_text_with_line_numbers, 
                str(headings_list), 
                total_lines
            )

            # Call LLM to get structured document
            try:
                grouping_result = await self.llm_client.structured_generate(prompt, SemanticGroupingResponse)
                sections = grouping_result.sections
                if not sections:
                    raise ValueError("No sections found in semantic grouping response")
            except Exception as json_err:
                logger.warning(f"Failed to parse semantic grouping response; falling back to rule-based sections. Error: {json_err}")
                sections = self._rule_based_sections(page)

            # Create chunks from the structured document
            chunks = create_chunks_from_structured_document(sections, total_lines)
            
            # Merge small chunks if applicable
            chunks = merge_chunks_by_tokens(chunks)
            
            # Add summaries to chunks
            await self._add_summaries_batch(chunks, page)
            
            # Add headings to chunks based on line ranges
            for chunk in chunks:
                chunk_headings = [
                    h for h in page.headings 
                    if chunk.start_index <= h.line_number <= chunk.end_index
                ]
                if chunk_headings:
                    # Update chunk title and parent title based on headings if needed
                    main_heading = chunk_headings[0]
                    if not chunk.title:
                        chunk.title = main_heading.text
                    if not chunk.parent_title and len(chunk_headings) > 1:
                        chunk.parent_title = main_heading.text
                        chunk.title = chunk_headings[1].text
                    # Add remaining headings as sub_titles
                    chunk.sub_titles.extend([h.text for h in chunk_headings[1:]])
            
            logger.info(f"Created {len(chunks)} chunks for page {page.page_number}")
            return chunks
        except Exception as e:
            logger.error(f"Error building chunks for page {page.page_number}: {e}")
            raise

    def _rule_based_sections(self, page: EnrichedPageData) -> List[Dict]:
        """Create sections using rule-based approach."""
        sections = []
        current_section = None
        
        for heading in page.headings:
            # Start new section
            if current_section:
                sections.append(current_section)
                
            current_section = {
                "title": heading.text,
                "start_index": heading.line_number,
                "content_type": self._infer_content_type(heading.text),
                "page_number": page.page_number,
                "sub_sections": []
            }
            
        # Add final section
        if current_section:
            sections.append(current_section)
            
        return sections

    def _infer_content_type(self, text: str) -> ContentType:
        """Infer content type from text using rules."""
        text = text.lower()
        
        if any(word in text for word in ["metrics", "statistics", "numbers", "data"]):
            return ContentType.METRICS
        elif any(word in text for word in ["instruction", "guide", "how to", "steps"]):
            return ContentType.INSTRUCTIONS
        elif any(word in text for word in ["definition", "glossary", "terms"]):
            return ContentType.CONCEPT
        elif any(word in text for word in ["vs", "versus", "compared", "comparison"]):
            return ContentType.COMPARISONS
        elif any(word in text for word in ["disclaimer", "legal", "notice"]):
            return ContentType.LEGAL
        else:
            return ContentType.NARRATIVE

    async def _add_summaries_batch(self, chunks: List[Chunk], page: EnrichedPageData) -> None:
        """Add summaries to chunks in batch."""
        if not chunks:
            return
            
        # Prepare prompts for all chunks in parallel
        async def get_chunk_text_and_prompt(chunk):
            # Get chunk text
            chunk_text = get_chunk_text(chunk, page.enriched_text_lines_map)
            
            prompt = f"""Summarize this text in 1-2 sentences:

{chunk_text}

Focus on key points and main ideas. Keep it concise."""
            return prompt

        # Generate prompts concurrently
        prompts = await asyncio.gather(*[
            get_chunk_text_and_prompt(chunk)
            for chunk in chunks
        ])
            
        # Get summaries in parallel batches of 5 to avoid rate limits
        BATCH_SIZE = 50
        summaries = []
        
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[i:i + BATCH_SIZE]
            batch_summaries = await asyncio.gather(*[
                self.llm_client.complete(prompt)
                for prompt in batch_prompts
            ])
            summaries.extend(batch_summaries)
            
        # Add summaries to chunks
        for chunk, summary in zip(chunks, summaries):
            chunk.summary = summary.strip() 
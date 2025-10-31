"""
Markdown document processor for semantic chunking.

This module handles the core processing logic for markdown documents:
- Regex-based heading extraction
- Semantic chunking with LLM
- Chunk formatting and metadata generation
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional

from ai_chunking.llm.base import StructuredLLMClient
from ai_chunking.chunkers.auto_ai_chunker.models.document import (
    ProcessedChunk, Heading, HeadingLevel, HeadingType
)
from ai_chunking.chunkers.auto_ai_chunker.models.llm_responses import (
    SemanticGroupingResponse, FooterDetectionResponse
)
from ai_chunking.chunkers.auto_ai_chunker.prompts import semantic_grouping_prompt
from ai_chunking.chunkers.markdown_ai_chunker.prompts import footer_detection_prompt
from ai_chunking.chunkers.auto_ai_chunker.chunk_builder import (
    create_chunks_from_structured_document, merge_chunks_by_tokens
)
from ai_chunking.models.chunk import Chunk


logger = logging.getLogger(__name__)


class MarkdownProcessor:
    """
    Processor for markdown documents.
    
    Handles the complete processing pipeline:
    1. Regex-based heading extraction
    2. Semantic chunking via LLM
    3. Chunk formatting and metadata generation
    
    Args:
        llm_client: LLM client for semantic chunking
        token_counter: Optional function to count tokens in text
    """

    def __init__(
        self,
        llm_client: StructuredLLMClient,
        token_counter=None,
        enrich_with_questions: bool = False,
        exclude_footer: bool = False
    ):
        """
        Initialize markdown processor.
        
        Args:
            llm_client: LLM for semantic chunking and summaries.
            token_counter: Optional function to count tokens.
            enrich_with_questions: If True, generate potential questions for each chunk.
            exclude_footer: If True, detect and remove footer content from the last chunk.
        """
        self.llm_client = llm_client
        self.token_counter = token_counter or llm_client.count_tokens
        self.enrich_with_questions = enrich_with_questions
        self.exclude_footer = exclude_footer

    async def process_markdown(
        self,
        text: str,
        source: str = "",
        customer_tags: Optional[List[str]] = None
    ) -> List[Chunk]:
        """
        Process a markdown document into semantic chunks.
        
        Processing flow:
        1. Extract headings using regex patterns
        2. Perform semantic chunking with LLM-generated summaries
        
        Args:
            text: Markdown text content.
            source: Source document identifier.
            customer_tags: Optional customer-specific tags.
            
        Returns:
            List of processed chunks with metadata.
        """
        logger.info(f"Processing markdown document: {source}")
        
        # Step 1: Extract headings from markdown using regex
        headings = self._extract_headings_regex(text)
        logger.info(f"Found {len(headings)} headings")
        
        # Create text lines map
        text_lines = text.split('\n')
        text_lines_map = {i+1: line for i, line in enumerate(text_lines)}
        total_lines = len(text_lines)
        
        # Step 2: Semantic chunking with LLM
        chunks = await self._build_chunks(text_lines_map, headings, total_lines)
        logger.info(f"Created {len(chunks)} semantic chunks (with summaries from LLM)")
        
        # Log document title if available
        if chunks and hasattr(chunks[0], 'document_title'):
            logger.info(f"Document title: {chunks[0].document_title}")
        
        # Step 3: Format and convert to ProcessedChunk
        processed_chunks = []
        for chunk in chunks:
            chunk_text = self._get_chunk_text(chunk, text_lines_map)
            
            # Get document title from chunk
            document_title = getattr(chunk, 'document_title', None)
            
            # Format chunk text
            formatted_text = self._format_chunk_text(
                content=chunk_text,
                title=chunk.title,
                parent_title=chunk.parent_title,
                sub_titles=chunk.sub_titles,
                summary=chunk.summary,
                questions=chunk.questions,
                document_title=document_title,
                customer_specific_tags=customer_tags
            )
            
            # Count tokens
            token_count = await self.token_counter(formatted_text)
            
            processed_chunks.append(ProcessedChunk(
                text=formatted_text,
                page_number=1,  # Single page for markdown
                content_type=chunk.content_type,
                tokens=token_count,
                summary=chunk.summary,
                parent_heading=chunk.parent_title,
                headings=[h for h in headings if chunk.start_index <= h.line_number <= chunk.end_index],
                tables=[],  # No tables for markdown processor
                metadata=self._build_chunk_metadata(chunk, customer_tags, source)
            ))
        
        logger.info(f"Markdown processing complete. Created {len(processed_chunks)} chunks")
        
        # Convert to Chunk objects with proper metadata
        final_chunks = []
        for chunk in processed_chunks:
            # Build metadata dict with proper types (not all strings)
            metadata = {
                "page_number": chunk.page_number,
                "content_type": chunk.content_type,
                "tokens": chunk.tokens,
                "summary": chunk.summary,
                "parent_heading": chunk.parent_heading,
                "source": source
            }
            
            # Add document_title if available in chunk.metadata
            if "document_title" in chunk.metadata:
                metadata["document_title"] = chunk.metadata["document_title"]
            
            # Extract questions from the string metadata back to list
            # (ProcessedChunk.metadata converts everything to strings)
            if "questions" in chunk.metadata:
                import ast
                try:
                    # Parse string representation back to list
                    questions_str = chunk.metadata["questions"]
                    if questions_str.startswith('['):
                        metadata["questions"] = ast.literal_eval(questions_str)
                    else:
                        metadata["questions"] = []
                except:
                    metadata["questions"] = []
            else:
                metadata["questions"] = []
                
            final_chunks.append(Chunk(
                text=chunk.text,
                metadata=metadata
            ))
        
        return final_chunks

    def _extract_headings_regex(self, text: str) -> List[Heading]:
        """
        Extract headings from markdown using regex.
        
        Parses standard markdown headings: # ## ### #### ##### ######
        
        Args:
            text: Markdown text.
            
        Returns:
            List of identified headings with line numbers and types.
        """
        lines = text.split("\n")
        headings = []
        
        # Regex to match markdown headings: ^(#{1,6})\s+(.+)$
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        
        for line_num, line in enumerate(lines, start=1):
            match = heading_pattern.match(line.strip())
            if match:
                hashes = match.group(1)
                title = match.group(2).strip()
                
                # Determine heading level
                level_map = {
                    1: HeadingLevel.H1,
                    2: HeadingLevel.H2,
                    3: HeadingLevel.H3,
                    4: HeadingLevel.H4,
                    5: HeadingLevel.H5,
                    6: HeadingLevel.H6
                }
                level = level_map.get(len(hashes), HeadingLevel.H2)
                
                # Determine heading type based on level
                if len(hashes) == 1:
                    heading_type = HeadingType.GLOBAL  # H1 is usually document title
                elif len(hashes) == 2:
                    heading_type = HeadingType.PAGE  # H2 is main sections
                else:
                    heading_type = HeadingType.SECTION  # H3+ are subsections
                
                heading = Heading(
                    text=title,
                    level=level,
                    line_number=line_num,
                    is_bridge=False,
                    heading_type=heading_type,
                    page_number=1
                )
                headings.append(heading)
        
        logger.info(f"Extracted {len(headings)} headings")
        return headings

    async def _build_chunks(
        self,
        text_lines_map: Dict[int, str],
        headings: List[Heading],
        total_lines: int
    ):
        """
        Build semantic chunks using LLM-based semantic grouping.
        
        Args:
            text_lines_map: Map of line numbers to text.
            headings: Detected headings.
            total_lines: Total number of lines.
            
        Returns:
            List of chunks with summaries and document_title in metadata.
        """
        # Build text with line numbers
        enriched_text_with_line_numbers = ""
        for i in range(1, total_lines + 1):
            line = text_lines_map.get(i, "")
            enriched_text_with_line_numbers += f"{i}. {line}\n"

        # Convert headings to expected format
        headings_list = []
        for heading in headings:
            heading_dict = {
                "title": heading.text,
                "start_index": heading.line_number,
                "heading_type": heading.heading_type,
                "page_number": 1
            }
            headings_list.append(heading_dict)

        # Generate semantic grouping prompt
        prompt = semantic_grouping_prompt(
            enriched_text_with_line_numbers,
            str(headings_list),
            total_lines,
            self.enrich_with_questions
        )

        # Call LLM to get structured document
        try:
            grouping_result = await self.llm_client.structured_generate(
                prompt, SemanticGroupingResponse
            )
            sections = grouping_result.sections
            document_title = grouping_result.document_title
            if not sections:
                raise ValueError("No sections found in semantic grouping response")
        except Exception as e:
            logger.warning(f"Failed semantic grouping: {e}, using rule-based fallback")
            sections = self._rule_based_sections(headings)
            document_title = "Document"  # Fallback title

        # Create chunks from structured document
        chunks = create_chunks_from_structured_document(sections, total_lines)
        
        # Merge small chunks
        chunks = merge_chunks_by_tokens(chunks)
        
        # Detect and remove footer from last chunk if enabled
        if self.exclude_footer and chunks:
            chunks = await self._remove_footer_from_last_chunk(chunks, text_lines_map)
        
        # Add document_title to each chunk as an attribute
        for chunk in chunks:
            chunk.document_title = document_title
        
        
        return chunks

    async def _remove_footer_from_last_chunk(self, chunks, text_lines_map):
        """
        Detect and remove footer content from the last chunk using LLM.
        
        Args:
            chunks: List of chunks after merging.
            text_lines_map: Dictionary mapping line numbers to text lines.
            
        Returns:
            Updated list of chunks with footer removed from last chunk.
        """
        if not chunks:
            return chunks
        
        last_chunk = chunks[-1]
        
        # Get the text of the last chunk with line numbers
        chunk_lines = []
        for line_num in range(last_chunk.start_index, last_chunk.end_index + 1):
            if line_num in text_lines_map:
                chunk_lines.append(f"{line_num}| {text_lines_map[line_num]}")
        
        chunk_text_with_line_numbers = "\n".join(chunk_lines)
        
        # Ask LLM to detect footer
        try:
            prompt = footer_detection_prompt(chunk_text_with_line_numbers)
            footer_result = await self.llm_client.structured_generate(
                prompt, FooterDetectionResponse
            )
            
            if footer_result.footer_start_line is not None:
                # Footer detected - trim the last chunk
                original_end = last_chunk.end_index
                new_end = footer_result.footer_start_line - 1
                
                if new_end >= last_chunk.start_index:
                    logger.info(f"Footer detected at line {footer_result.footer_start_line}. Trimming last chunk from {original_end} to {new_end}")
                    last_chunk.end_index = new_end
                else:
                    logger.warning(f"Footer start line {footer_result.footer_start_line} is before chunk start {last_chunk.start_index}. Not trimming.")
            else:
                logger.info("No footer content detected in last chunk")
                
        except Exception as e:
            logger.warning(f"Failed to detect footer: {e}. Keeping last chunk as-is.")
        
        return chunks
    
    def _rule_based_sections(self, headings: List[Heading]):
        """
        Create sections using rule-based approach as fallback.
        
        Used when LLM-based semantic grouping fails.
        
        Args:
            headings: List of detected headings.
            
        Returns:
            List of Section objects.
        """
        from ai_chunking.chunkers.auto_ai_chunker.models.llm_responses import Section
        from ai_chunking.chunkers.auto_ai_chunker.models.document import ContentType
        
        sections = []
        for i, heading in enumerate(headings):
            section = Section(
                title=heading.text,
                start_index=heading.line_number,
                content_type=ContentType.NARRATIVE,
                summary=f"Section covering {heading.text}",
                sub_sections=[]
            )
            sections.append(section)
        
        return sections if sections else [Section(
            title="Document Content",
            start_index=1,
            content_type=ContentType.NARRATIVE,
            summary="Document content",
            sub_sections=[]
        )]

    def _get_chunk_text(self, chunk, text_lines_map: Dict[int, str]) -> str:
        """
        Get chunk text considering gap ranges.
        
        Excludes lines marked as gaps (e.g., tables, images).
        
        Args:
            chunk: Chunk object with start/end indices.
            text_lines_map: Map of line numbers to text.
            
        Returns:
            Chunk text as string.
        """
        excluded_lines = set()
        for gap_start, gap_end in chunk.gap_index_range:
            for ln in range(gap_start, gap_end + 1):
                excluded_lines.add(ln)

        lines = []
        for ln in range(chunk.start_index, chunk.end_index + 1):
            if ln not in excluded_lines and ln in text_lines_map:
                lines.append(text_lines_map[ln])

        return "\n".join(lines)

    def _format_chunk_text(
        self,
        content: str,
        title: str,
        parent_title: Optional[str] = None,
        sub_titles: Optional[List[str]] = None,
        summary: Optional[str] = None,
        questions: Optional[List[str]] = None,
        document_title: Optional[str] = None,
        customer_specific_tags: Optional[List[str]] = None,
    ) -> str:
        """
        Format chunk text with structural context.
        
        Creates a well-structured chunk with metadata headers.
        
        Args:
            content: Raw chunk content.
            title: Chunk title.
            parent_title: Optional parent section title.
            sub_titles: Optional list of sub-section titles.
            summary: Optional chunk summary.
            questions: Optional list of potential questions this chunk answers.
            document_title: Optional main document title.
            customer_specific_tags: Optional custom tags.
            
        Returns:
            Formatted chunk text ready for embedding/RAG.
        """
        formatted_parts = []

        # Add document title if present (main # heading)
        if document_title and str(document_title).strip() and str(document_title) not in ("None", "N/A"):
            formatted_parts.append(f"DOCUMENT: {document_title}")

        # Add parent section if present
        if parent_title and str(parent_title) not in ("None", "N/A"):
            formatted_parts.append(f"SECTION: {parent_title}")

        # Add current topic
        formatted_parts.append(f"TOPIC: {title}")

        # Add sub-topics if present
        cleaned_subs = [
            s for s in (sub_titles or []) 
            if s and str(s).strip() and str(s) not in ("None", "N/A")
        ]
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

        # Add questions if present
        if questions and len(questions) > 0:
            formatted_parts.append("\nPOTENTIAL QUESTIONS:")
            for question in questions:
                if question and question.strip():
                    formatted_parts.append(f"  • {question}")

        return "\n".join(formatted_parts)

    def _build_chunk_metadata(
        self,
        chunk,
        customer_tags: Optional[List[str]],
        source: str
    ) -> Dict[str, any]:
        """
        Build chunk metadata dictionary.
        
        Args:
            chunk: Chunk object.
            customer_tags: Optional custom tags.
            source: Source document path.
            
        Returns:
            Metadata dictionary.
        """
        metadata = {"source": source}
        
        if customer_tags:
            metadata["customer_tags"] = ", ".join(str(t) for t in customer_tags)
            
        if chunk.parent_title:
            metadata["parent_section"] = str(chunk.parent_title)
            
        if chunk.sub_titles:
            metadata["sub_sections"] = ", ".join(str(t) for t in filter(None, chunk.sub_titles))
        
        # Add questions to metadata (keep as list, not string)
        if hasattr(chunk, 'questions') and chunk.questions:
            metadata["questions"] = chunk.questions
        
        # Add document_title to metadata
        if hasattr(chunk, 'document_title') and chunk.document_title:
            metadata["document_title"] = chunk.document_title
            
        return metadata


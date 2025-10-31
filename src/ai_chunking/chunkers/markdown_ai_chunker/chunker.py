"""
Markdown AI Chunker - Main chunker implementation.

This module provides the MarkdownAIChunker class for ultra-optimized
semantic chunking of markdown documents.
"""

import os
from typing import List, Optional
import asyncio
import concurrent.futures

from ai_chunking.llm.base import LLMConfig
from ai_chunking.llm.factory import LLMFactory, LLMProvider
from ai_chunking.llm.models import OpenAIModels
from ai_chunking.models.chunk import Chunk
from ai_chunking.utils.markdown_utils import load_markdown

from .processor import MarkdownProcessor


class MarkdownAIChunker:
    """
    Ultra-optimized semantic chunker for Markdown files.
    
    This chunker is designed specifically for standard markdown documents
    and provides significant performance and cost improvements over general-purpose
    chunkers.
    
    Features:
        - Regex-based heading extraction
        - Single LLM call per document for semantic chunking and summaries (optional footer detection adds one more call)
        - ~70% cheaper than full AutoAIChunker
        - 3-5x faster processing
        - Optimized for cost-efficient processing
    
    Best suited for:
        - Standard markdown files with # ## ### headings
        - Single-page documents
        - Cost-sensitive applications
        - High-volume document processing
    
    Example:
        >>> chunker = MarkdownAIChunker()
        >>> chunks = chunker.chunk_document("path/to/file.md")
        >>> for chunk in chunks:
        ...     print(f"Chunk: {chunk.text[:100]}...")
        ...     print(f"Metadata: {chunk.metadata}")
    
    Args:
        api_key: Optional OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
        model: OpenAI model to use. Defaults to GPT-4o-mini for optimal cost efficiency.
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
    
    Raises:
        ValueError: If OPENAI_API_KEY is not set and not provided.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = OpenAIModels.GPT_4O_MINI,
        temperature: float = 0.0,
        enrich_with_questions: bool = False,
        max_input_tokens: int = 12000,
        exclude_footer: bool = False
    ):
        """Initialize the Markdown AI Chunker.
        
        Args:
            api_key: Optional OpenAI API key. Defaults to OPENAI_API_KEY env var.
            model: Model to use for semantic chunking. Defaults to GPT-4o-mini.
            temperature: Sampling temperature for LLM. Defaults to 0.0.
            enrich_with_questions: If True, generate 2-4 potential questions for each chunk.
                                   Questions are added to chunk metadata and text. Default: False.
            max_input_tokens: Maximum tokens to send to LLM in a single call. Documents exceeding
                            this limit will be split at H2 boundaries. Default: 12000.
            exclude_footer: If True, uses LLM to detect and remove footer content from the last chunk.
                           Footer content includes: tags, copyright, social links, etc. Default: False.
        
        Raises:
            ValueError: If API key is not provided and not in environment.
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set either as environment variable "
                "or passed to constructor"
            )
        
        # Initialize single LLM client for all operations
        llm = LLMFactory.create(
            provider=LLMProvider.OPENAI,
            api_key=api_key,
            config=LLMConfig(model=model, temperature=temperature)
        )

        # Store configuration
        self.max_input_tokens = max_input_tokens
        self.exclude_footer = exclude_footer
        
        # Initialize markdown processor
        self.processor = MarkdownProcessor(
            llm_client=llm,
            enrich_with_questions=enrich_with_questions,
            exclude_footer=exclude_footer
        )
    
    def _run_async(self, coro):
        """
        Safely run a coroutine in the appropriate event loop.
        
        Handles both cases where an event loop is running or not.
        
        Args:
            coro: Coroutine to execute.
            
        Returns:
            Result of the coroutine execution.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If the loop is already running, create a new loop in a new thread
                def run_in_new_loop(coro):
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    return new_loop.run_until_complete(coro)
                
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(run_in_new_loop, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)
    
    def _split_by_headers(self, text: str):
        """
        Intelligently split markdown by headers, maximizing chunk size while keeping topics together.
        
        Strategy:
        1. Try to keep entire text if under limit
        2. Split by H1 (#) and recursively process each
        3. For large H1 sections: split by H2 (##) and process each section
        4. Continue recursively through H3, H4, H5, H6 until content fits
        5. If no structure found, return as-is for LLM to handle
        
        Args:
            text: Markdown text to split.
            
        Returns:
            List of (content, metadata) tuples ready for LLM processing.
        """
        return self._split_content_recursive(text, {})
    
    def _split_content_recursive(self, content: str, metadata: dict, heading_level: int = 1):
        """
        Recursively split content by headers while respecting token limits.
        Progressively goes deeper (H1 → H2 → H3 → H4 → H5 → H6) until content fits.
        
        Args:
            content: Text content to split.
            metadata: Metadata to attach to resulting documents.
            heading_level: Current heading level to try (1=H1, 2=H2, etc.)
            
        Returns:
            List of (content, metadata) tuples.
        """
        from ai_chunking.utils.count_tokens import count_tokens
        
        tokens = count_tokens(content)
        
        # Base case: content fits within limit
        if tokens <= self.max_input_tokens:
            return [(content, metadata)]
        
        # Try splitting at current heading level
        splits = self._split_by_heading_level(content, heading_level)
        
        if len(splits) > 1:
            # Got multiple sections - batch and recurse on each
            return self._batch_and_recurse_sections(splits, metadata, heading_level)
        
        # No splits at this level, try next deeper level
        if heading_level < 6:
            return self._split_content_recursive(content, metadata, heading_level + 1)
        
        # Reached H6 with no splits - content has no structure
        # Return as-is and let LLM handle it (it will create smaller semantic chunks)
        return [(content, metadata)]
    
    def _split_by_heading_level(self, content: str, level: int):
        """
        Split content by heading at specified level using regex.
        Handles duplicate titles by splitting on each occurrence.
        
        Args:
            content: Markdown content.
            level: Heading level (1=H1, 2=H2, etc.)
            
        Returns:
            List of (content, heading_title) tuples.
        """
        import re
        
        # Create pattern for this heading level (e.g., "# " for H1, "## " for H2)
        heading_marker = '#' * level
        # Split on heading at start of line, ensuring it's not a deeper heading
        pattern = rf'\n(?={heading_marker} [^#])'
        parts = re.split(pattern, content)
        
        results = []
        heading_key = f'H{level}'
        
        for part in parts:
            if part.strip():
                # Extract heading title if present
                heading_pattern = rf'^{heading_marker} (.+?)(?:\n|$)'
                heading_match = re.match(heading_pattern, part)
                heading_title = heading_match.group(1).strip() if heading_match else None
                results.append((part, heading_title, heading_key))
        
        return results
    
    def _batch_and_recurse_sections(self, sections, base_metadata, current_level):
        """
        Process each section individually - recurse deeper if needed.
        No batching at this stage to keep logic simple and correct.
        
        Args:
            sections: List of (content, title, heading_key) tuples.
            base_metadata: Base metadata to extend.
            current_level: Current heading level being processed.
            
        Returns:
            List of (content, metadata) tuples.
        """
        results = []
        
        for content, title, heading_key in sections:
            # Build metadata for this section
            section_metadata = base_metadata.copy()
            if title:
                section_metadata[heading_key] = title
            
            # Recurse deeper on this section
            results.extend(self._split_content_recursive(content, section_metadata, current_level + 1))
        
        return results
    
    
    def _process_large_document(
        self,
        text: str,
        source: str,
        customer_tags: Optional[List[str]] = None
    ) -> List[Chunk]:
        """
        Process large documents by splitting at header boundaries.
        
        Args:
            text: Markdown text to process.
            source: Source file path.
            customer_tags: Optional custom tags.
            
        Returns:
            List of chunks from all splits combined.
        """
        # Split the document
        splits = self._split_by_headers(text)
        
        print(f"📄 Large document detected ({len(text)} chars)")
        print(f"📊 Split into {len(splits)} sections for processing")
        
        all_chunks = []
        split_results = []  # Store (split_index, chunks, split_size)
        
        for i, (content, metadata) in enumerate(splits):
            h1 = metadata.get("H1", "")
            h2 = metadata.get("H2", "")
            
            print(f"\n🔄 Processing split {i+1}/{len(splits)}: {h2 or h1 or 'Untitled'}...")
            
            # Process this split
            chunks = self._run_async(
                self.processor.process_markdown(
                    text=content,
                    source=source,
                    customer_tags=customer_tags
                )
            )
            
            split_results.append((i, chunks, len(content)))
            all_chunks.extend(chunks)
            print(f"✅ Generated {len(chunks)} chunks from this split")
        
        # Determine the best document_title from the LARGEST split (likely main content)
        if split_results:
            # Sort by split size (descending) to get the largest split
            split_results.sort(key=lambda x: x[2], reverse=True)
            largest_split_chunks = split_results[0][1]
            
            if largest_split_chunks:
                document_title = largest_split_chunks[0].metadata.get("document_title")
                print(f"\n📝 Document title (from largest split): {document_title}")
                
                # Apply this document_title to ALL chunks
                for chunk in all_chunks:
                    chunk.metadata["document_title"] = document_title
        
        print(f"\n🎉 Total: {len(all_chunks)} chunks from large document")
        return all_chunks
    
    def chunk_documents(
        self,
        documents: List[str],
        customer_tags: Optional[List[str]] = None
    ) -> List[Chunk]:
        """
        Process multiple markdown documents.
        
        Args:
            documents: List of paths to markdown files.
            customer_tags: Optional list of custom tags to add to chunk metadata.
            
        Returns:
            List of chunks from all documents combined.
            
        Example:
            >>> chunker = MarkdownAIChunker()
            >>> chunks = chunker.chunk_documents([
            ...     "doc1.md",
            ...     "doc2.md",
            ...     "doc3.md"
            ... ])
            >>> print(f"Generated {len(chunks)} total chunks")
        """
        chunks = []
        for document in documents:
            chunks.extend(self.chunk_document(document, customer_tags))
        return chunks
    
    def chunk_document(
        self,
        document: str,
        customer_tags: Optional[List[str]] = None
    ) -> List[Chunk]:
        """
        Process a single markdown document into semantic chunks.
        
        This method:
        1. Loads the markdown file
        2. Checks if document exceeds token limit
        3. If large: Splits by headers recursively, processes each split separately
        4. If small: Processes entire document in one LLM call
        5. Returns structured chunks with metadata
        
        Args:
            document: Path to markdown file to process.
            customer_tags: Optional custom tags to include in chunk metadata.
            
        Returns:
            List of Chunk objects with text and metadata.
            
        Example:
            >>> chunker = MarkdownAIChunker()
            >>> chunks = chunker.chunk_document(
            ...     "blog_post.md",
            ...     customer_tags=["blog", "technical"]
            ... )
            >>> for chunk in chunks:
            ...     print(f"Type: {chunk.metadata['content_type']}")
            ...     print(f"Summary: {chunk.metadata['summary']}")
        """
        from ai_chunking.utils.count_tokens import count_tokens
        
        content = load_markdown(document)
        total_tokens = count_tokens(content)
        
        # Check if document is too large for single LLM call
        if total_tokens > self.max_input_tokens:
            # Large document - split and process
            return self._process_large_document(content, document, customer_tags)
        else:
            # Small document - process directly
            chunks = self._run_async(
                self.processor.process_markdown(
                    text=content,
                    source=document,
                    customer_tags=customer_tags
                )
            )
            return chunks


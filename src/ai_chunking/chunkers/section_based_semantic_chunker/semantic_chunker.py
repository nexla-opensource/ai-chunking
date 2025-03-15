from typing import List, Optional
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_chunking.utils.count_tokens import count_tokens

MAX_CHUNK_SIZE = 1024


class ConstrainedSemanticChunker(SemanticChunker):
    """
    A semantic chunker that ensures all chunks fall within a specified size range.  
    If chunks are too large, they are further split using recursive semantic chunking.
    If chunks are too small, they are merged with adjacent chunks when possible.
    
    If max_chunk_size is None, only semantic chunking is performed without size constraints.
    """
    
    
    def __init__(self, min_chunk_size: int, max_chunk_size: Optional[int] = None, **kwargs):
        """
        Initialize the constrained semantic chunker.
        
        Args:
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in tokens, or None for no maximum
            **kwargs: Additional arguments to pass to SemanticChunker
        """
        super().__init__(**kwargs)
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks that fall within the specified size range.
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks
        """
        # Get initial chunks using the parent class method
        initial_chunks = super().split_text(text)
        
        # If max_chunk_size is None, return the initial semantic chunks without further processing
        if self.max_chunk_size is None:
            return initial_chunks
            
        # Process chunks to ensure they fall within size constraints
        return self._process_chunks(initial_chunks)
    
    def _process_chunks(self, chunks: List[str]) -> List[str]:
        """
        Process chunks to ensure they fall within size constraints.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Processed list of text chunks
        """
        processed_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            chunk_tokens = count_tokens(chunk)
            
            # If chunk is too large, split it further using RecursiveCharacterTextSplitter
            if chunk_tokens > self.max_chunk_size:
                # Calculate overlap as 10% of max_chunk_size
                overlap = int(self.max_chunk_size * 0.1)
                
                # Use RecursiveCharacterTextSplitter as specified
                text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    model_name="gpt-4",
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=overlap
                )
                
                sub_chunks = text_splitter.split_text(chunk)
                processed_chunks.extend(sub_chunks)
            
            # If chunk is too small, try to merge with the current accumulated chunk
            elif chunk_tokens < self.min_chunk_size:
                if current_chunk:
                    combined = current_chunk + " " + chunk
                    combined_tokens = count_tokens(combined)
                    
                    # If combined chunk is within limits, update current_chunk
                    if combined_tokens <= self.max_chunk_size:
                        current_chunk = combined
                    else:
                        # If combined chunk would be too large, add current_chunk to results
                        # and start a new current_chunk with this small chunk
                        processed_chunks.append(current_chunk)
                        current_chunk = chunk
                else:
                    # If there's no current chunk, start with this one
                    current_chunk = chunk
            
            # If chunk is within range, add current accumulated chunk (if any) to results
            # and start a new current_chunk with this chunk
            else:
                if current_chunk:
                    processed_chunks.append(current_chunk)
                    current_chunk = ""
                processed_chunks.append(chunk)
        
        # Add any remaining current_chunk
        if current_chunk:
            processed_chunks.append(current_chunk)
            
        return processed_chunks



def create_semantic_splitter(min_chunk_size: int, max_tokens: Optional[int] = MAX_CHUNK_SIZE):
    """
    Create semantic text splitter with optimized settings.
    
    Args:
        min_chunk_size: Minimum chunk size in characters
        max_tokens: Target maximum tokens per chunk. If None, only semantic chunking is performed
                   without size constraints.
        
    Returns:
        Configured SemanticChunker instance
    """
    # Determine breakpoint threshold based on max_tokens if provided
    if max_tokens is None:
        # Default threshold for unconstrained semantic chunking
        threshold_amount = 90.0
    elif max_tokens <= 500:
        threshold_amount = 80.0  # More aggressive splitting for small max_tokens
    elif max_tokens <= 1000:
        threshold_amount = 85.0  # Moderate splitting
    else:
        threshold_amount = 90.0  # Less aggressive splitting for larger max_tokens
    
    # Use the ConstrainedSemanticChunker with or without size constraints
    return ConstrainedSemanticChunker(
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_tokens,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold_amount
    )

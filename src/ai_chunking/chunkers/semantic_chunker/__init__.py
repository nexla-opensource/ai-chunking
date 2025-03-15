from pathlib import Path
from typing import List, Dict, Any
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from ai_chunking.chunkers.base_chunker import BaseChunker
from ai_chunking.models.chunk import Chunk
from ai_chunking.utils.count_tokens import count_tokens
from ai_chunking.utils.markdown_utils import load_markdown


class SemanticTextChunker(BaseChunker):
    """
    Chunks documents based on semantic similarity using Langchain's SemanticChunker.
    This chunker uses embeddings to determine semantic breakpoints in the text.
    """
    MIN_CHUNK_SIZE = 128
    MAX_CHUNK_SIZE  = 1024
    SIMILARITY_THRESHOLD = 0.9

    def __init__(self, 
                 chunk_size: int = MAX_CHUNK_SIZE,
                 min_chunk_size: int = MIN_CHUNK_SIZE,
                 similarity_threshold: float = SIMILARITY_THRESHOLD):
        """
        Initialize the semantic chunker.
        
        Args:
            chunk_size: Maximum chunk size in tokens
            min_chunk_size: Minimum chunk size in tokens
            similarity_threshold: Threshold for semantic similarity when determining breakpoints
        """
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        
        # Initialize the semantic chunker with OpenAI embeddings
        self.chunker = SemanticChunker(
            embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
            min_chunk_size=min_chunk_size,
            # max_chunk_size=chunk_size,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=similarity_threshold * 100  # Convert to percentile
        )
    

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Process a text and return chunks.
        """
        chunks = self.chunker.split_text(text)
        return chunks
    

    def chunk_documents(self, file_paths: List[str]) -> List[Chunk]:
        """
        Process multiple documents and return chunks.
        
        Args:
            file_paths: List of paths to the documents to process
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        for file_path in file_paths:
            file_chunks = self.chunk_document(file_path)
            chunks.extend(file_chunks)
        return chunks
    
    def chunk_document(self, file_path: str) -> List[Chunk]:
        """
        Process a single document and return chunks.
        
        Args:
            file_path: Path to the document to process
            
        Returns:
            List of chunks with metadata
        """
        # Convert string path to Path object
        path = Path(file_path)
        
        # Load and preprocess the document
        content = load_markdown(path)
        
        # Split text into semantic chunks
        chunks = self.chunk_text(content)
        
        # Create chunk objects with metadata
        chunk_objects = []
        for chunk_text in chunks:
            chunk_obj = Chunk(text=chunk_text, metadata={
                "filename": path.name,
                "source_path": str(path),
                "tokens_count": count_tokens(chunk_text),
            })
            chunk_objects.append(chunk_obj)
        return chunk_objects

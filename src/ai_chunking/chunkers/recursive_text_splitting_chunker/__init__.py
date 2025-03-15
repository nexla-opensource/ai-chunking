from chunk import Chunk
from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_chunking.chunkers.base_chunker import BaseChunker
from ai_chunking.models.chunk import Chunk
from ai_chunking.utils.count_tokens import count_tokens
from ai_chunking.utils.markdown_utils import load_markdown


class RecursiveTextSplitter(BaseChunker):
    """Recursive text splitter."""
    MAX_CHUNK_SIZE = 1000 
    CHUNK_OVERLAP = 100

    def __init__(self, chunk_size: int = MAX_CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[Chunk]:
        """Chunk text."""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",  # Using GPT-4o tokenizer
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_text(text)
        return [Chunk(text=chunk, metadata={}) for chunk in chunks]

    def chunk_documents(self, file_paths: List[str]) -> List[Chunk]:
        """Chunk documents."""
        chunks = []
        for file_path in file_paths:
            file_chunks = self.chunk_document(file_path)
            chunks.extend(file_chunks)
        return chunks
    
    def chunk_document(self, file_path: str) -> List[Chunk]:
        return self.chunk_markdown_file(file_path)

    def chunk_markdown_file(self, file_path: Path) -> List[Dict]:
        """
        Process a markdown file: remove pagination, join content, and split into chunks
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        print(f"Processing {file_path}")
        
        # Load content and extract page information
        content = load_markdown(file_path)
        
        # Create text splitter with recursive character splitting
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",  # Using GPT-4o tokenizer
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Split the text into chunks
        chunks = text_splitter.split_text(content)
        print(f"Created {len(chunks)} chunks from {file_path}")
        
        # Create chunk objects with metadata
        chunk_objects = []
        for chunk_text in chunks:
            chunk_obj = Chunk(text=chunk_text, metadata={
                "filename": Path(file_path).name,
                "source_path": str(file_path),
                "tokens_count": count_tokens(chunk_text)
            })
            chunk_objects.append(chunk_obj)
        return chunk_objects

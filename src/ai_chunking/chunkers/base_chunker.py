from abc import ABC, abstractmethod
from typing import List

from ..models.chunk import Chunk


class BaseChunker(ABC):
    """Base class for all chunker implementations."""
    
    @abstractmethod
    def chunk_documents(self, file_paths: List[str]) -> List[Chunk]:
        """Process a document and return chunks.

        Args:
            file_path: Path to the document to process
            
        Returns:
            List of chunks
        """
        pass

    @abstractmethod
    def chunk_document(self, file_path: str) -> List[Chunk]:
        """Process a document and return chunks.

        Args:
            file_path: Path to the document to process
        """
        pass

    @abstractmethod
    def chunk_text(self, text: str) -> List[Chunk]:
        """Process a text and return chunks.

        Args:
            text: Text to process
        """
        pass
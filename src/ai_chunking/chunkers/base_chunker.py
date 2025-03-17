from abc import ABC, abstractmethod
from typing import List

from ..models.chunk import Chunk
from ..utils.logging import logger


class BaseChunker(ABC):
    """Base class for all chunker implementations."""
    
    def __init__(self):
        self.logger = logger
    
    @abstractmethod
    def chunk_documents(self, file_paths: List[str]) -> List[Chunk]:
        """Process a document and return chunks.

        Args:
            file_path: Path to the document to process
            
        Returns:
            List of chunks
        """
        self.logger.info(f"Processing {len(file_paths)} documents")
        pass

    @abstractmethod
    def chunk_document(self, file_path: str) -> List[Chunk]:
        """Process a document and return chunks.

        Args:
            file_path: Path to the document to process
        """
        self.logger.info(f"Processing document: {file_path}")
        pass

    @abstractmethod
    def chunk_text(self, text: str) -> List[Chunk]:
        """Process a text and return chunks.

        Args:
            text: Text to process
        """
        self.logger.info("Processing text input")
        pass
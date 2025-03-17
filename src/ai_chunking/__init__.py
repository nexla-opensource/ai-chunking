"""AI Chunking - A powerful Python library for semantic document chunking and enrichment using AI"""

__version__ = "0.1.4"


from .chunkers.base_chunker import BaseChunker
from .chunkers.recursive_text_splitting_chunker import RecursiveTextSplitter
from .chunkers.section_based_semantic_chunker import SectionBasedSemanticChunker
from .chunkers.semantic_chunker import SemanticTextChunker
from .chunkers.auto_ai_chunker import AutoAIChunker


__all__ = [
    "BaseChunker",
    "SemanticTextChunker",
    "RecursiveTextSplitter",
    "SectionBasedSemanticChunker",
    "AutoAIChunker",
]

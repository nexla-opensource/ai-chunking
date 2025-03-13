"""AI Chunking - A powerful Python library for semantic document chunking and enrichment using AI"""

__version__ = "0.1.0"


from .chunkers.base_chunker import BaseChunker
from .chunkers.recursive_text_splitting_chunker import RecursiveChunker
from .chunkers.section_based_semantic_chunker import SectionBasedChunker
# from .chunkers.auto_ai_chunker import AutoAIChunker


__all__ = [
    "BaseChunker",
    "SemanticChunker",
    "RecursiveChunker",
    "SectionBasedChunker",
    # "AutoAIChunker",
]

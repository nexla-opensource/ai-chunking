"""
Type definitions for PDF content chunking.
"""

from typing import List, Dict, Any, Optional, Union, TypedDict

class SectionInfo(TypedDict):
    """Information about a section in the document."""
    id: str
    text: str
    level: int

class BlockInfo(TypedDict):
    """Information about a block in the document."""
    id: str
    type: str
    content: str
    section_path: List[str]
    section_headers: Dict[str, str]
    page_id: Optional[str]

class Chunk(TypedDict):
    """A processed chunk of text with its metadata."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    block_ids: List[str]
    section_path: List[str]
    section_headers: Dict[str, str]
    page_ids: List[str] 
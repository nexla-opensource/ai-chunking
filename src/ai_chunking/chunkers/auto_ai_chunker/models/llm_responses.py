"""Pydantic models for structured LLM responses."""

from typing import List, Optional
from pydantic import BaseModel, Field

from .document import ContentType, HeadingLevel, HeadingType, Heading

class HeadingInfo(BaseModel):
    """Information about a single heading."""
    text: str = Field(..., description="The heading text")
    level: HeadingLevel = Field(default=HeadingLevel.H2, description="The heading level (h1-h6)")
    line_number: int = Field(..., description="Line number where heading appears")
    is_bridge: bool = Field(default=False, description="Whether this is a bridge heading")
    heading_type: HeadingType = Field(default=HeadingType.SECTION, description="Type of heading")

class HeadingAnalysis(BaseModel):
    """Analysis of headings in text."""
    headings: List[HeadingInfo] = Field(default_factory=list, description="List of identified headings")

class EnrichedTextResponse(BaseModel):
    """Response model for enriched text."""
    enriched_text: str = Field(..., description="The enriched text content")
    metadata: dict = Field(default_factory=dict, description="Additional metadata about the enriched text")

class ContentAnalysis(BaseModel):
    """Analysis of content type and structure."""
    content_type: ContentType = Field(..., description="The type of content")
    summary: str = Field(..., description="A brief summary of the content")
    parent_title: Optional[str] = Field(default=None, description="Parent section title if any")
    title: Optional[str] = Field(default=None, description="Section title if any")
    sub_titles: List[str] = Field(default_factory=list, description="Any sub-section titles")

class TableAnalysis(BaseModel):
    """Analysis of table content and context."""
    caption: str = Field(..., description="Generated caption describing the table")
    context: str = Field(..., description="How the table relates to surrounding text")
    key_points: List[str] = Field(..., description="Key insights from the table")

class Section(BaseModel):
    """A section in the document."""
    title: str = Field(..., description="The section title")
    start_index: int = Field(..., description="Starting line number")
    end_index: Optional[int] = Field(None, description="Ending line number")
    content_type: ContentType = Field(..., description="Type of content in the section")
    summary: str = Field(..., description="Brief summary of the section content")
    continued: bool = Field(default=False, description="Whether this section continues from previous page")
    sub_sections: List["Section"] = Field(default_factory=list, description="List of sub-sections")
    page_number: Optional[int] = Field(None, description="Page number where this section appears")

class SemanticGroupingResponse(BaseModel):
    """Response model for semantic grouping of document content."""
    sections: List[Section] = Field(..., description="List of sections in the document")

# Required for recursive Pydantic models
Section.model_rebuild() 
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Types of content that can be found in documents."""
    METRICS = "Metrics"  # Content focused on quantitative data, performance metrics, numerical results
    NARRATIVE = "Narrative"  # Explanatory, descriptive, or analytical text content
    INSTRUCTIONS = "Instructions"  # Steps, procedures, guidelines, or recommendations
    COMPARISONS = "Comparisons"  # Content comparing multiple datasets, entities, or periods
    LEGAL = "Legal Disclaimers"  # Regulatory notices, compliance statements, warnings
    CONCEPT = "Concept Definition"  # Content defining key terms, concepts, frameworks


class HeadingLevel(str, Enum):
    """Document heading levels."""
    H1 = "h1"
    H2 = "h2"
    H3 = "h3"
    H4 = "h4"
    H5 = "h5"
    H6 = "h6"


class TableData(BaseModel):
    """Table data found in documents."""
    page_number: int
    rows: List[Dict[str, str]]
    table_id: Optional[str] = None
    caption: Optional[str] = None


class HeadingType(str, Enum):
    """Types of document headings."""
    GLOBAL = "global_heading"
    PAGE = "page_heading"
    SECTION = "section_heading"
    FOOTER = "footer_heading"
    BRIDGE = "bridge_heading"


class Heading(BaseModel):
    """Document heading with metadata."""
    text: str
    level: HeadingLevel
    line_number: int
    is_bridge: bool = False
    parent_heading: Optional[str] = None
    heading_type: HeadingType = Field(default=HeadingType.SECTION, description="Type of heading")
    page_number: Optional[int] = None
    
    def __hash__(self):
        return hash((self.text, self.level, self.line_number, self.page_number))
    
    def __eq__(self, other):
        if not isinstance(other, Heading):
            return False
        return (self.text == other.text and 
                self.level == other.level and 
                self.line_number == other.line_number and 
                self.page_number == other.page_number)


class Page(BaseModel):
    """Document page with text and metadata.
    
    Raises:
        ValueError: If page_number is not positive or text is empty
    """
    page_number: int = Field(gt=0, description="Page number must be positive")
    text: str = Field(min_length=1, description="Page text content must not be empty")
    source: Optional[str] = Field(default=None, description="Source of the page content")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Additional validation
        if not self.text.strip():
            raise ValueError("Page text cannot be only whitespace")
    tables: List[TableData] = Field(default_factory=list)
    headings: List[Heading] = Field(default_factory=list)


class Chunk(BaseModel):
    """Raw document chunk with metadata before processing."""
    parent_title: Optional[str]
    title: Optional[str]
    content_type: ContentType = Field(description="Type of content in the chunk")
    start_index: int = Field(ge=0, description="Start index must be non-negative")
    end_index: int
    summary: str = ""
    sub_titles: List[Optional[str]] = Field(default_factory=list)
    gap_index_range: List[Tuple[int, int]] = Field(default_factory=list)
    page_number: Optional[int] = Field(default=None, description="Page number where this chunk appears")
    
    def __init__(self, **data):
        if "content_type" in data and isinstance(data["content_type"], str):
            # Map common variations to our standard content types
            content_type_map = {
                "metrics": "Metrics",
                "narrative": "Narrative",
                "instructions": "Instructions",
                "comparisons": "Comparisons",
                "legal": "Legal Disclaimers",
                "concept": "Concept Definition",
                # Add title case versions too for robustness
                "Metrics": "Metrics",
                "Narrative": "Narrative",
                "Instructions": "Instructions",
                "Comparisons": "Comparisons",
                "Legal Disclaimers": "Legal Disclaimers",
                "Concept Definition": "Concept Definition"
            }
            data["content_type"] = content_type_map.get(data["content_type"], data["content_type"])
        super().__init__(**data)
        
        if self.end_index < self.start_index:
            raise ValueError(f"end_index ({self.end_index}) must be >= start_index ({self.start_index})")


class ProcessedChunk(BaseModel):
    """Processed document chunk with metadata."""
    text: str = Field(description="The chunk's text content")
    page_number: int = Field(gt=0, description="Page number must be positive")
    content_type: ContentType = Field(description="Type of content in the chunk")
    tokens: int = Field(ge=0, description="Token count must be non-negative")
    summary: str = Field(default="", description="Summary of the chunk content")
    parent_heading: Optional[str] = Field(default=None, description="Parent heading text")
    headings: List[Heading] = Field(default_factory=list, description="List of headings in this chunk")
    tables: List[TableData] = Field(default_factory=list, description="List of tables in this chunk")
    metadata: Dict[str, str] = Field(default_factory=dict, description="String-only metadata dictionary")
    
    @property
    def id(self) -> str:
        """Generate a unique identifier for the chunk."""
        return f"page_{self.page_number}_{self.content_type}_{self.tokens}"
    
    def __init__(self, **data):
        # Ensure metadata values are strings
        if "metadata" in data:
            data["metadata"] = {
                str(k): str(v) 
                for k, v in data["metadata"].items()
            }
        
        # Ensure headings are unique
        if "headings" in data:
            data["headings"] = list(set(data["headings"]))
            
        # Sort headings by line number for consistency
        if "headings" in data:
            data["headings"].sort(key=lambda h: h.line_number)
            
        super().__init__(**data)
    
    class Config:
        validate_assignment = True
        json_encoders = {
            Heading: lambda h: h.dict(),
            TableData: lambda t: t.dict()
        }


class EnrichedPageData(BaseModel):
    """Page data enriched with tables and headings."""
    page_number: int
    source: str
    enriched_text: str
    headings: List[Heading]
    tables: List[TableData] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)
    enriched_text_lines_map: Dict[int, str] = Field(default_factory=dict, description="Map of line numbers to enriched text lines") 
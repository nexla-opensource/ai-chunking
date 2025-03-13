from litellm import BaseModel
from typing import Dict, Any


class Chunk(BaseModel):
    """Represents a document chunk with text and metadata."""
    text: str
    metadata: Dict[str, Any]

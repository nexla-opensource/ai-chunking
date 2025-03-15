from typing import List, Dict, Optional
from datetime import datetime
import json
from pydantic import BaseModel, Field

class Summary(BaseModel):
    text: str
    metadata: Dict

    def to_dict(self):
        return {
            "text": self.text,
            "metadata": self.metadata
        }

class Chunk(BaseModel):
    chunk_id: int
    text: str
    tokens_count: int
    page_numbers: List[int]
    summary: Summary
    global_context: str
    section_summary: Summary
    document_summary: Summary
    previous_chunk_summary: Optional[str] = None
    next_chunk_summary: Optional[str] = None
    questions_this_excerpt_can_answer: List[str]
    
    def to_dict(self):
        return {
            "id": self.chunk_id,
            "text": self.text,
            "tokens_count": self.tokens_count,
            "page_numbers": self.page_numbers,
            "global_context": self.global_context,
            "summary": self.summary.text,
            "metadata": self.summary.metadata,
            "previous_summary": self.previous_chunk_summary,
            "next_summary": self.next_chunk_summary,
            "questions_this_excerpt_can_answer": self.questions_this_excerpt_can_answer
        }

class Section(BaseModel):
    section_id: int
    title: str
    tokens_count: int
    text: str
    summary: Summary
    chunks: List[Chunk] = Field(default_factory=list)
    
    def to_dict(self):
        return {
            "id": self.section_id,
            "title": self.title,
            "tokens_count": self.tokens_count,
            "text": self.text,
            "summary": self.summary.text,
            "metadata": self.summary.metadata,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "chunks_count": len(self.chunks)
        }

class Document(BaseModel):
    name: str
    text: str
    tokens_count: int
    total_pages: Optional[int] = None
    page_map: Optional[Dict[int, str]] = None
    summary: Optional[Summary] = None
    sections: List[Section] = Field(default_factory=list)
    creation_date: datetime = Field(default_factory=datetime.now)
    
    def to_dict(self):
        if not self.summary:
            raise ValueError("Document summary not set")
            
        return {
            "name": self.name,
            "total_pages": self.total_pages,
            "tokens_count": self.tokens_count,
            "text": self.text,
            "creation_date": self.creation_date.isoformat(),
            "document_summary": self.summary.text,
            "document_metadata": self.summary.metadata,
            "sections": [section.to_dict() for section in self.sections],
            "sections_count": len(self.sections)
        }
    
    def save_json(self, output_path):
        """Save document to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

# Extracted verbatim from chunking_code.py — see PROVENANCE.md
from pydantic import BaseModel
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ai_chunking.chunkers.document_chunker")


# ==== Models ====
class DocMeta(BaseModel):
    file_title: str
    freshness_date: str
    document_summary: str
    extras: Dict[str, Any] = {}   # Additional fields the LLM emits under customInstructions

class DocumentPlan(BaseModel):
    doc_type: str = ""                      # free-form label e.g. "lease", "financial_report", "technical_spec", "narrative"
    has_headings: bool = False              # document appears to have a heading hierarchy
    heading_depth_estimate: int = 0         # 0-5; 0 when has_headings is False
    table_density: str = "none"             # "none" | "low" | "medium" | "high"
    figure_count_estimate: int = 0          # rough count of charts/figures/diagrams in the sample
    scanned_pages_suspected: bool = False   # sample text is sparse/garbled in a way that suggests OCR need
    strategy: str = "heading_driven"        # "heading_driven" | "density" | "hybrid" — OBSERVATIONAL only; not read for routing. See _handle_pdf for why.
    special_handling: List[str] = []        # e.g. ["vision_ocr_needed", "tabular_extraction_heavy"]
    notes: str = ""                         # free-text planner observations

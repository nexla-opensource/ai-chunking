# Extracted verbatim from chunking_code.py — see PROVENANCE.md
import os
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ai_chunking.chunkers.document_chunker")


# ==== Config ====
@dataclass
class Config:
    GEMINI_MODEL: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL") or "gemini-2.5-flash")
    GEMINI_LITE_MODEL: str = field(default_factory=lambda: os.getenv("GEMINI_LITE_MODEL") or "gemini-2.5-flash-lite")
    GEMINI_METADATA_MODEL: str = field(default_factory=lambda: os.getenv("GEMINI_METADATA_MODEL") or "gemini-3-flash-preview")

    MAX_TOKEN_LIMIT: int = 900_000
    MAX_CHARS_PER_BLOCK: int = 2_200_000
    PAGES_PER_CHUNK: int = 3
    SEMAPHORE_CHUNK_PROCESSING: int = 3  # Reduced for Ray environment to avoid rate limits
    ROWS_PER_CSV_BLOCK: int = 40  # Number of rows per CSV chunk block

    # Standardized timeout configuration
    GEMINI_TIMEOUT_STANDARD: int = 100   # Docmeta, chunking
    GEMINI_TIMEOUT_VISION: int = 120    # Vision processing
    GEMINI_TIMEOUT_LARGE: int = 180     # Large document processing

    # Concurrency control configuration
    MAX_CONCURRENT_SHEETS: int = 2       # Excel sheet-level concurrency
    MAX_CONCURRENT_BLOCKS_PER_SHEET: int = 2  # Block-level per sheet

    # Vision-call optimization
    VISION_SKIP_TEXT_THRESHOLD: int = 200  # Chars; if pdfplumber extracts at least this,
                                           # skip vision even when page was flagged visual.
    VISION_BATCH_SIZE: int = 4             # Pages per multi-image vision call.

    USE_HEADING_DRIVEN_CHUNKING: bool = field(default_factory=lambda: os.getenv("CHUNKER_USE_HEADING_DRIVEN", "1").lower() in ("1", "true", "yes"))
    HEADING_MIN_SECTION_LINES: int = 3        # Sections shorter than this get merged with next.
    HEADING_MAX_SECTION_LINES: int = 120      # Sections longer than this get split.
    CHUNK_METADATA_BATCH_SIZE: int = 6        # Chunks per metadata-enrichment LLM call.
    HEADING_MIN_HEADINGS_FOR_HEADING_PATH: int = 2  # If a block yields fewer detected

    CHUNK_MAX_CHARS: int = 1800      # Section bigger than this triggers size-aware carving.
    CHUNK_MIN_CHARS: int = 400       # Don't emit fragments smaller than this.
                                     # Applied as both a size-carving floor AND
                                     # a post-section-merge guard (see
                                     # _merge_small_sections_by_chars).

    VERIFY_CHUNKS: bool = field(default_factory=lambda: os.getenv("CHUNKER_VERIFY_CHUNKS", "1").lower() in ("1", "true", "yes"))
    VERIFY_SAMPLE_RATE: float = 0.35           # Fraction of chunks to verify
    VERIFY_MIN_SAMPLE: int = 6                 # Always verify at least this many
    VERIFY_BATCH_SIZE: int = 6                 # Chunks per verification LLM call
    VERIFY_MAX_RETRIES: int = 1                # Retry rounds per failing chunk
    VERIFY_DRIFT_THRESHOLD: float = 0.20       # If drift > this fraction of sample,
    VERIFY_DRIFT_FLOOR: float = 0.05            # If drift < this, skip retry entirely.
    VERIFY_MAX_CHUNKS_RETRIED: int = 30         # Hard cap on chunks retried per doc.
    VERIFY_MAX_COST_USD: float = 0.10           # Hard cap on verify+retry cost per doc.
    VERIFY_MAX_WALLCLOCK_SEC: int = 180         # Wall-clock cap on the verify loop.
    VERIFY_REQUIRE_MONOTONIC: bool = True       # Stop retrying if a round didn't reduce drift.

    STRUCTURED_TABLE_EXTRACTION: bool = field(default_factory=lambda: os.getenv("CHUNKER_STRUCTURED_TABLES", "1").lower() in ("1", "true", "yes"))
    STRUCTURED_TABLE_BATCH_SIZE: int = 4
    STRUCTURED_TABLE_CONTENT_TYPES: Tuple[str, ...] = ("metrics", "comparisons")
    STRUCTURED_TABLE_VALIDATE: bool = True
    STRUCTURED_TABLE_MAX_UNGROUNDED_FRACTION: float = 0.30  # Drop tables where >30% of cells don't appear in source
    STRUCTURED_TABLE_BRIDGE: bool = True
    STRUCTURED_TABLE_BRIDGE_HEADER_MATCH_THRESHOLD: float = 0.75  # Header similarity required to merge

@dataclass
class PricingConfig:
    """Configuration for Gemini API pricing - can be updated as pricing changes"""

    # Gemini 2.5 Flash pricing per 1M tokens (as of current rates)
    INPUT_PRICE_PER_1M_TOKENS: float = 0.30  # $0.30 per 1M tokens (text/image/video)
    INPUT_PRICE_AUDIO_PER_1M_TOKENS: float = 1.00  # $1.00 per 1M tokens (audio)
    OUTPUT_PRICE_PER_1M_TOKENS: float = 2.50  # $2.50 per 1M tokens (including thinking tokens)

    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str = "gemini-2.5-flash", input_type: str = "text") -> float:
        # Select input pricing based on input type
        if input_type == "audio":
            input_rate = self.INPUT_PRICE_AUDIO_PER_1M_TOKENS
        else:  # text, image, video all use same rate
            input_rate = self.INPUT_PRICE_PER_1M_TOKENS

        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * input_rate
        output_cost = (output_tokens / 1_000_000) * self.OUTPUT_PRICE_PER_1M_TOKENS

        total_cost = input_cost + output_cost

        logger.debug(f"Cost calculation: {input_tokens} input tokens @ ${input_rate}/1M = ${input_cost:.6f}")
        logger.debug(f"Cost calculation: {output_tokens} output tokens @ ${self.OUTPUT_PRICE_PER_1M_TOKENS}/1M = ${output_cost:.6f}")
        logger.debug(f"Total cost: ${total_cost:.6f}")

        return total_cost

    def get_pricing_info(self) -> dict:
        """Return current pricing information for logging/debugging"""
        return {
            "input_price_per_1m_tokens_text": self.INPUT_PRICE_PER_1M_TOKENS,
            "input_price_per_1m_tokens_audio": self.INPUT_PRICE_AUDIO_PER_1M_TOKENS,
            "output_price_per_1m_tokens": self.OUTPUT_PRICE_PER_1M_TOKENS,
            "currency": "USD"
        }

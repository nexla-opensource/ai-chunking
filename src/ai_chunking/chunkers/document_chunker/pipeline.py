# Extracted verbatim from chunking_code.py — see PROVENANCE.md
import asyncio
import base64
import json
import os
import random
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import pdfplumber
import logging
from typing import Any, Dict, List, Optional, Tuple
from .chunker_utils import Utils
from .config import Config, PricingConfig
from .llm import LLMService
from .merger import Merger
from .models import DocMeta, DocumentPlan
from .pdf import PDFProfiler, PageProcessor
from .prompts import PromptFactory

logger = logging.getLogger("ai_chunking.chunkers.document_chunker")


# ==== Document Pipeline ====
class DocumentPipeline:
    def __init__(self, cfg: Config, api_key: str, custom_instructions: Optional[str] = None):
        self.cfg = cfg
        self.pricing = PricingConfig()  # Initialize pricing configuration
        self.llm = LLMService(cfg, api_key=api_key)
        self.profiler = PDFProfiler()
        self.pages = PageProcessor()
        self.merger = Merger()
        self.custom_instructions = (custom_instructions or "").strip() or None

        # Centralized token tracking for document-level metrics
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_llm_calls: int = 0

    def _reset_token_counters(self) -> None:
        """Reset token counters before processing a new file"""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_llm_calls = 0
        self._total_pages = None
        self._pages_with_chunks = None
        self._pages_with_zero_chunks = None

    async def _track_and_call_gemini(self, prompt: str, call_type: str, timeout: int = 60, image_data: Optional[str] = None, model: Optional[str] = None) -> tuple[str, int, int]:
        input_tokens = self.llm.count_tokens(prompt) or 0
        self._total_input_tokens += input_tokens

        response = ""
        try:
            if image_data:
                response = await self.llm.agen_vision(prompt, image_data, timeout, model=model)
            else:
                response = await self.llm.agen_text(prompt, timeout_s=timeout, model=model)
        except Exception as e:
            logger.error(f"Gemini call for '{call_type}' failed: {e}")
            raise e  # Re-raise to ensure error handling is triggered

        output_tokens = self.llm.count_tokens(response) or 0
        self._total_output_tokens += output_tokens
        self._total_llm_calls += 1

        logger.debug(f"Gemini call '{call_type}' completed. Tokens (In/Out): {input_tokens}/{output_tokens}. Total calls: {self._total_llm_calls}")

        return response, input_tokens, output_tokens

    async def _extract_docmeta_centralized(self, document_text: str, file_title: str, file_type: str = "pdf") -> tuple[DocMeta, int, int]:
        prompt = PromptFactory.docmeta(document_text, file_title, file_type,
                                       custom_instructions=self.custom_instructions)

        try:
            raw_response, input_tokens, output_tokens = await self._track_and_call_gemini(
                prompt, f"docmeta_{file_type}", timeout=self.cfg.GEMINI_TIMEOUT_STANDARD
            )

            cleaned = Utils.clean_json_fence(raw_response)
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError as je:
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start >= 0 and end > start:
                    salvaged = cleaned[start:end + 1]
                    try:
                        data = json.loads(salvaged)
                        logger.warning(f"[DOCMETA] Salvaged JSON from non-strict response for {file_type}")
                    except json.JSONDecodeError:
                        # Last resort: try_repair_json if available
                        data = Utils.try_repair_json(cleaned, context_label=f"docmeta_{file_type}")
                        if not isinstance(data, dict):
                            raise je
                else:
                    raise

            if not isinstance(data, dict):
                raise ValueError("Response is not a JSON object.")

            # document_summary is no longer requested from the LLM (removed
            # from prompt), but still defaulted to "" in DocMeta for back-compat.
            baseline = {"file_title", "freshness_date", "document_summary"}
            for k in baseline:
                if k not in data:
                    data[k] = ""

            _SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")

            def _valid_extra_key(k: str) -> bool:
                if not isinstance(k, str) or not k:
                    return False
                if len(k) > 80:
                    return False
                return bool(_SNAKE_CASE.match(k))

            raw_extras = {k: v for k, v in data.items() if k not in baseline and k != "extras"}
            shape_ok = {k: v for k, v in raw_extras.items() if _valid_extra_key(k)}
            shape_dropped = [k for k in raw_extras if k not in shape_ok]

            if self.custom_instructions:
                declared_fields = set(
                    re.findall(r'["\']([a-z][a-z0-9_]*)["\']\s*:', self.custom_instructions)
                )
                extras = {k: v for k, v in shape_ok.items() if k in declared_fields}
                provenance_dropped = [k for k in shape_ok if k not in extras]
                if declared_fields:
                    logger.debug(f"[DOCMETA] Declared fields from instructions: {sorted(declared_fields)}")
            else:
                extras = shape_ok
                provenance_dropped = []

            if shape_dropped:
                logger.warning(f"[DOCMETA] Dropped {len(shape_dropped)} malformed keys: "
                               f"{[k[:40] + '…' if len(k) > 40 else k for k in shape_dropped]}")
            if provenance_dropped:
                logger.warning(f"[DOCMETA] Dropped {len(provenance_dropped)} keys not declared in instructions: "
                               f"{provenance_dropped}")

            meta = DocMeta(
                file_title=str(data.get("file_title") or ""),
                freshness_date=str(data.get("freshness_date") or ""),
                document_summary=str(data.get("document_summary") or ""),
                extras=extras,
            )
            logger.info(f"Docmeta validated successfully for {file_type} (baseline + {len(extras)} extra fields): {sorted(extras.keys())}")
            return meta, input_tokens, output_tokens

        except Exception as e:
            logger.error(f"DocMeta extraction failed for {file_type}: {e}")

            # For PDF files, try regex-based filename date extraction as fallback
            freshness_date = ""
            if file_type == "pdf":
                logger.info("Attempting regex-based date extraction from filename as fallback")
                freshness_date = self._extract_date_from_filename_regex(file_title)
                if freshness_date:
                    logger.info(f"Successfully extracted date from filename: {freshness_date}")
                else:
                    logger.warning("No date found in filename using regex patterns")

            default_meta = DocMeta(
                file_title=file_title,
                freshness_date=freshness_date,
                document_summary="",
            )
            return default_meta, 0, 0

    async def _plan_document_async(self, sample_text: str) -> DocumentPlan:
        if not sample_text or not sample_text.strip():
            logger.info("[PLANNER] Empty sample text; returning default plan")
            return DocumentPlan()

        prompt = PromptFactory.document_plan(sample_text)
        try:
            raw_response, _, _ = await self._track_and_call_gemini(
                prompt, "document_plan", timeout=self.cfg.GEMINI_TIMEOUT_STANDARD,
                model=self.cfg.GEMINI_LITE_MODEL,
            )
            cleaned = Utils.clean_json_fence(raw_response)
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            data = json.loads(cleaned)
            if not isinstance(data, dict):
                raise ValueError("Planner response is not a JSON object")
            plan = DocumentPlan(**{k: v for k, v in data.items() if k in DocumentPlan.model_fields})
            return plan
        except Exception as e:
            logger.warning(f"[PLANNER] Plan generation failed (non-fatal): {e}")
            return DocumentPlan()

    def plan_document(self, sample_text: str) -> DocumentPlan:
        return asyncio.run(self._plan_document_async(sample_text))

    def _extract_date_from_filename_regex(self, filename: str) -> str:
        """Extract date from filename using regex patterns as fallback"""
        if not filename:
            return ""

        # Remove file extension for easier matching
        name_without_ext = Path(filename).stem
        logger.debug(f"Attempting regex date extraction from: {name_without_ext}")

        # Common date patterns in filenames (ordered by specificity)
        patterns = [
            # YYYY-MM-DD format
            r'(\d{4}-\d{2}-\d{2})',
            # YYYY_MM_DD format
            r'(\d{4}_\d{2}_\d{2})',
            # YYYYMMDD format
            r'(\d{8})',
            # YYYY-MM format
            r'(\d{4}-\d{2})',
            # YYYY_MM format
            r'(\d{4}_\d{2})',
            # Month Year (e.g., "June_2023", "Jun_2023", "June-2023")
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[_\s-]+\d{4})',
            # Year Month (e.g., "2023_June", "2023-Jun")
            r'(\d{4}[_\s-]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)',
            # Q1/Q2/Q3/Q4 YYYY format
            r'(Q[1-4][_\s-]*\d{4})',
            # YYYY Q1/Q2/Q3/Q4 format
            r'(\d{4}[_\s-]*Q[1-4])',
            # Year only (4 digits) - least specific, so last
            r'(\d{4})'
        ]

        for i, pattern in enumerate(patterns):
            match = re.search(pattern, name_without_ext, re.IGNORECASE)
            if match:
                date_str = match.group(1).strip()
                logger.debug(f"Pattern {i+1} matched: {date_str}")

                # Handle special cases before normalization
                if re.match(r'Q[1-4]', date_str, re.IGNORECASE):
                    # Handle Q1/Q2/Q3/Q4 formats
                    date_str = self._convert_quarter_to_date(date_str)
                elif re.match(r'\d{4}[_\s-]*Q[1-4]', date_str, re.IGNORECASE):
                    # Handle YYYY Q1/Q2/Q3/Q4 formats
                    date_str = self._convert_quarter_to_date(date_str)
                elif re.match(r'\d{8}', date_str):
                    # Handle YYYYMMDD format
                    if len(date_str) == 8:
                        date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                elif re.match(r'\d{4}[_\s-]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', date_str, re.IGNORECASE):
                    # Handle YYYY Month format - reorder to Month YYYY
                    parts = re.split(r'[_\s-]+', date_str)
                    if len(parts) >= 2:
                        date_str = f"{parts[1]} {parts[0]}"
                elif '_' in date_str:
                    # Replace underscores with spaces for better normalization
                    date_str = date_str.replace('_', ' ')
                elif '-' in date_str and not re.match(r'\d{4}-\d{2}', date_str):
                    # Replace hyphens with spaces for month-year patterns
                    date_str = date_str.replace('-', ' ')

                # Normalize the extracted date using existing utility
                normalized = Utils.normalize_freshness_date(date_str)
                if normalized:
                    logger.debug(f"Successfully normalized '{date_str}' to '{normalized}'")
                    return normalized
                else:
                    logger.debug(f"Failed to normalize: {date_str}")

        logger.debug("No date patterns matched in filename")
        return ""

    def _extract_date_from_content(self, document_text: str) -> str:
        if not document_text:
            return ""

        snippet = document_text[:5000]

        # Patterns ordered by specificity
        content_date_patterns = [
            # "December 30, 2009" or "Dec 30, 2009"
            r'([A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4})',
            # "30 December 2009" or "30 Dec 2009"
            r'(\d{1,2}\s+[A-Z][a-z]+\.?\s+\d{4})',
            # "2009-12-30"
            r'(\d{4}-\d{2}-\d{2})',
            # "12/30/2009" or "12-30-2009"
            r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})',
        ]

        for pattern in content_date_patterns:
            match = re.search(pattern, snippet)
            if match:
                raw = match.group(1).strip().rstrip(',')
                normalized = Utils.normalize_freshness_date(raw)
                if normalized:
                    logger.debug(f"Content date extraction: '{raw}' -> '{normalized}'")
                    return normalized

        return ""

    def _convert_quarter_to_date(self, quarter_str: str) -> str:
        """Convert quarter notation (Q1 2024, 2024 Q3) to end-of-quarter date"""
        # Extract year and quarter
        year_match = re.search(r'(\d{4})', quarter_str)
        quarter_match = re.search(r'Q([1-4])', quarter_str, re.IGNORECASE)

        if not year_match or not quarter_match:
            return quarter_str  # Return original if parsing fails

        year = int(year_match.group(1))
        quarter = int(quarter_match.group(1))

        # Map quarters to end-of-quarter dates
        quarter_end_dates = {
            1: f"{year}-03-31",  # Q1 ends March 31
            2: f"{year}-06-30",  # Q2 ends June 30
            3: f"{year}-09-30",  # Q3 ends September 30
            4: f"{year}-12-31",  # Q4 ends December 31
        }

        return quarter_end_dates.get(quarter, quarter_str)

    async def _extract_headers_centralized(self, table_text: str, file_type: str = "table") -> List[str]:
        """Extract clean headers from full table using centralized token tracking"""
        prompt = PromptFactory.header_extraction(table_text, file_type)

        try:
            raw_response, _, _ = await self._track_and_call_gemini(
                prompt, f"header_extraction_{file_type}", timeout=self.cfg.GEMINI_TIMEOUT_LARGE,
                model=self.cfg.GEMINI_LITE_MODEL,
            )

            # Debug: Log the raw response
            logger.debug(f"Header extraction raw response ({len(raw_response)} chars): '{raw_response[:500]}...'")

            if not raw_response.strip():
                logger.warning("Header extraction returned empty response")
                return []

            cleaned = Utils.clean_json_fence(raw_response.strip())
            logger.debug(f"Cleaned response for JSON parsing: '{cleaned[:200]}...'")

            # Try to extract JSON array if it's embedded in text
            if not cleaned.startswith('['):
                # Look for JSON array in the response
                import re
                json_match = re.search(r'\[.*?\]', cleaned, re.DOTALL)
                if json_match:
                    cleaned = json_match.group(0)
                    logger.debug(f"Extracted JSON from response: '{cleaned}'")
                else:
                    logger.warning("No JSON array found in response")
                    return []

            headers = json.loads(cleaned)
            if isinstance(headers, list):
                logger.info(f"Successfully extracted {len(headers)} headers from {file_type}: {headers}")
                return headers
            else:
                logger.warning(f"Header extraction returned non-list: {type(headers)}, value: {headers}")
                return []
        except json.JSONDecodeError as je:
            logger.error(f"Header extraction JSON decode error: {je}")
            logger.error(f"Raw response causing error: '{raw_response}'")
            return []
        except Exception as e:
            logger.error(f"Header extraction failed: {e}")
            logger.error(f"Raw response: '{raw_response}'")
            return []

    async def _extract_freshness_centralized(self, file_name: str) -> tuple[str, int]:
        """Extract freshness date from filename using centralized token tracking"""
        logger.info(f"Extracting freshness date from filename: '{file_name}'")
        prompt = PromptFactory.filename_date_extraction(file_name)
        logger.debug(f"Freshness extraction prompt length: {len(prompt)} chars")

        try:
            raw_response, _, _ = await self._track_and_call_gemini(
                prompt, "freshness_extraction", timeout=self.cfg.GEMINI_TIMEOUT_STANDARD,
                model=self.cfg.GEMINI_LITE_MODEL,
            )

            logger.debug(f"Freshness raw response: '{raw_response}'")
            logger.debug(f"Freshness raw response length: {len(raw_response)} chars")

            if not raw_response or not raw_response.strip():
                logger.warning("Freshness extraction returned empty response")
                return "Unknown", -1

            date_str = raw_response.strip().strip('"')
            logger.debug(f"Cleaned date string: '{date_str}'")

            if not date_str or date_str.lower() in ["unknown", "none", "null", ""]:
                logger.warning(f"Gemini returned no valid date: '{date_str}'")
                return "Unknown", -1

            logger.debug(f"Attempting to normalize date string: '{date_str}'")
            normalized = Utils.normalize_freshness_date(date_str)
            logger.debug(f"Normalized result: '{normalized}'")

            if not normalized:
                logger.warning(f"Date normalization failed for: '{date_str}'")
                unix_ts = -1
            else:
                logger.debug(f"Converting normalized date to unix: '{normalized}'")
                unix_ts = Utils.date_to_unix(normalized)
                logger.debug(f"Unix timestamp result: {unix_ts}")

            logger.info(f"Extracted freshness date: '{normalized}' (unix: {unix_ts})")
            return normalized if normalized else "Unknown", unix_ts
        except Exception as e:
            logger.error(f"Freshness date extraction failed for '{file_name}': {e}")
            logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
            return "Unknown", -1

    async def _process_tabular_block_centralized(self, block_index: int, start_row: int, end_row: int, table_text: str, file_type: str = "CSV", sheet_name: str = None) -> List[dict]:
        """Process a single tabular block with Gemini using centralized token tracking"""
        prompt = PromptFactory.tabular_chunking(table_text, start_row, end_row, file_type, sheet_name,
                                                custom_instructions=self.custom_instructions)

        try:
            raw_response, _, _ = await self._track_and_call_gemini(
                prompt, f"tabular_chunk_{file_type}_{block_index}", timeout=self.cfg.GEMINI_TIMEOUT_LARGE
            )

            cleaned = Utils.clean_json_fence(raw_response)
            logger.debug(f"Block {block_index} raw response length: {len(raw_response)} chars")
            logger.debug(f"Block {block_index} cleaned response: {cleaned[:300]}...")

            try:
                parsed_response = json.loads(cleaned)
            except json.JSONDecodeError as je:
                logger.warning(f"[TABULAR] Block {block_index} JSON decode error: {je.msg} at line {je.lineno}")
                try:
                    parsed_response = Utils.try_repair_json(cleaned, context_label=f"tabular_block_{block_index}")
                    logger.info(f"[TABULAR] Block {block_index} JSON repair succeeded")
                except ValueError:
                    raise  # Let the outer except handler catch it
            logger.debug(f"Block {block_index} parsed response type: {type(parsed_response)}")

            # Handle both single chunk (dict) and multiple chunks (list) responses
            chunks = []
            if isinstance(parsed_response, list):
                # Gemini returned multiple chunks - process all of them
                if len(parsed_response) > 0:
                    chunks = parsed_response
                    logger.info(f"Gemini returned {len(parsed_response)} chunks for block {block_index}")
                else:
                    # Empty list - create a basic chunk
                    chunks = [{"chunk_text": "No content generated", "summary": "Empty response"}]
                    logger.warning(f"Gemini returned empty chunk list for block {block_index}")
            elif isinstance(parsed_response, dict):
                # Single chunk object - convert to list
                chunks = [parsed_response]
            else:
                # Unexpected response type
                raise ValueError(f"Unexpected response type: {type(parsed_response)}, expected dict or list")

            # Process each chunk in the list
            processed_chunks = []
            for chunk_idx, chunk in enumerate(chunks):
                # Note: We'll renumber chunks globally later, for now use a placeholder
                chunk["chunk_number"] = f"block_{block_index}_chunk_{chunk_idx + 1}"

                # Set chunk_text from summary if not present
                if "chunk_text" not in chunk or not chunk["chunk_text"]:
                    chunk["chunk_text"] = chunk.get("summary", "")

                # Extract fields BEFORE moving to debug_info for prepending (Excel/CSV only)
                universal = chunk.get("universal", "").strip()
                topic = chunk.get("topic", "").strip()
                section = chunk.get("section", "").strip()
                sub_topics = chunk.get("sub_topics", "").strip()

                # Build prefix with pipe separation
                prefix_parts = []
                if universal:
                    prefix_parts.append(universal)
                if topic:
                    prefix_parts.append(topic)
                if section:
                    prefix_parts.append(section)
                if sub_topics:
                    prefix_parts.append(sub_topics)

                # Prepend to chunk_text if we have prefix parts
                if prefix_parts and chunk.get("chunk_text"):
                    prefix = " | ".join(prefix_parts)
                    original_chunk_text = chunk["chunk_text"]
                    chunk["chunk_text"] = f"{prefix} | {original_chunk_text}"
                    logger.debug(f"{file_type} Block {block_index} Chunk {chunk_idx + 1}: Prepended '{prefix}' to chunk_text")
                elif prefix_parts:
                    # If no original chunk_text but we have prefix parts, use just the prefix
                    prefix = " | ".join(prefix_parts)
                    chunk["chunk_text"] = prefix
                    logger.debug(f"{file_type} Block {block_index} Chunk {chunk_idx + 1}: Set chunk_text to prefix only: '{prefix}'")

                # Normalize analytical fields (keep as top-level keys, don't pop)
                chunk["universal"] = universal
                chunk["topic"] = topic
                chunk["section"] = section
                chunk["sub_topics"] = sub_topics
                if not self.custom_instructions:
                    chunk["customer_specific_tags"] = ""
                else:
                    cst = chunk.get("customer_specific_tags", "")
                    if isinstance(cst, list):
                        chunk["customer_specific_tags"] = json.dumps(cst)
                    elif isinstance(cst, str):
                        chunk["customer_specific_tags"] = cst.strip()
                    else:
                        chunk["customer_specific_tags"] = ""
                chunk.pop("content", None)  # content is redundant with chunk_text
                chunk.pop("sub-topics", None)  # normalize hyphenated key

                # Create debug_info with only essential processing information
                debug_info = {
                    "chunker_result": "Success",
                    "file_type": file_type  # Add file_type to debug_info
                }
                chunk["debug_info"] = debug_info

                processed_chunks.append(chunk)

            logger.info(f"Successfully processed {file_type} block {block_index} (rows {start_row}-{end_row}) -> {len(processed_chunks)} chunks")
            return processed_chunks

        except Exception as e:
            logger.error(f"Tabular block processing failed for block {block_index}: {e}")
            error_chunk = self._create_chunk_error(
                chunk_number=f"block_{block_index}_chunk_1",
                file_type=file_type,
                stage="block processing",
                error_message=str(e),
                block_index=block_index,
                start_row=start_row,
                end_row=end_row,
                sheet_name=sheet_name
            )
            return [error_chunk]  # Return as a list since method now returns List[dict]

    def _renumber_chunks_sequentially(self, chunks: List[dict]) -> List[dict]:
        """Renumber chunks sequentially starting from 1"""
        for i, chunk in enumerate(chunks, 1):
            chunk["chunk_number"] = str(i)
        logger.info(f"Renumbered {len(chunks)} chunks sequentially")
        return chunks

    def _inject_final_token_counts(self, chunk_df: pd.DataFrame, file_type: str = "unknown") -> pd.DataFrame:
        """Inject final token counts, model info, and cost into debug_info for all chunks"""

        # Calculate total cost for the file
        total_cost = self.pricing.calculate_cost(
            input_tokens=self._total_input_tokens,
            output_tokens=self._total_output_tokens,
            model=self.cfg.GEMINI_MODEL,
            input_type="image" if file_type == "pdf" else "text"  # PDF may have vision calls
        )

        def update_debug_info(row):
            debug_info = row.get("debug_info", {})
            if not isinstance(debug_info, dict):
                debug_info = {}

            # Add model information
            debug_info["model"] = self.cfg.GEMINI_MODEL

            # Add file_type information
            debug_info["file_type"] = file_type

            # Add centralized token tracking info under 'llm_stats' key with cost included
            debug_info["llm_stats"] = {}
            file_level_stats = {
                "input_tokens_per_file": self._total_input_tokens,
                "output_tokens_per_file": self._total_output_tokens,
                "total_tokens_per_file": self._total_input_tokens + self._total_output_tokens,
                "total_llm_calls_per_file": self._total_llm_calls,
            }
            # Page processing stats (PDF only — None for other file types)
            if getattr(self, '_total_pages', None) is not None:
                file_level_stats["total_pages"] = self._total_pages
                file_level_stats["pages_with_chunks"] = self._pages_with_chunks
                file_level_stats["pages_with_zero_chunks"] = self._pages_with_zero_chunks
            debug_info["llm_stats"]["file_level_stats"] = file_level_stats

            return debug_info

        chunk_df = chunk_df.copy()
        chunk_df["debug_info"] = chunk_df.apply(update_debug_info, axis=1)

        logger.info(f"Injected final metrics into debug_info.llm_stats for {len(chunk_df)} chunks:")
        logger.info(f"  Model: {self.cfg.GEMINI_MODEL}")
        logger.info(f"  File Type: {file_type}")
        logger.info(f"  Tokens: Input={self._total_input_tokens}, Output={self._total_output_tokens}, Total={self._total_input_tokens + self._total_output_tokens}")
        logger.info(f"  Total LLM Cost: ${total_cost:.6f} USD (Calls={self._total_llm_calls})")
        logger.info("  Cleaned debug_info: removed topic, content, and analytical fields")

        return chunk_df

    @staticmethod
    def _compute_dynamic_timeout(text: str) -> int:
        est_tokens = Utils.token_estimate(text)
        # Scale: base 120s + 15s per 1000 tokens, capped at 420s
        timeout_s = int(min(420, max(120, 120 + (est_tokens / 1000) * 15)))
        logger.info(f"[CHUNK-TIMEOUT] Block ~{est_tokens} est. tokens → dynamic timeout = {timeout_s}s")
        return timeout_s

    _PAGE_MARKER_RE = re.compile(r"^######\[Page\s+(\d+)\]######\s*$")

    @staticmethod
    def _number_lines(text: str) -> tuple[List[str], List[int], str]:
        raw_lines = text.split("\n")
        line_pages: List[int] = []
        current_page = 0
        for ln in raw_lines:
            m = DocumentPipeline._PAGE_MARKER_RE.match(ln)
            if m:
                try:
                    current_page = int(m.group(1))
                except ValueError:
                    pass
            line_pages.append(current_page)
        numbered = "\n".join(f"{i:>5} | {ln}" for i, ln in enumerate(raw_lines))
        return raw_lines, line_pages, numbered

    async def _detect_headings_async(self, numbered_text: str) -> List[Dict[str, Any]]:
        if not numbered_text.strip():
            return []
        prompt = PromptFactory.heading_detection(numbered_text)
        try:
            raw, _, _ = await self._track_and_call_gemini(
                prompt, "heading_detection",
                timeout=self.cfg.GEMINI_TIMEOUT_STANDARD,
                model=self.cfg.GEMINI_LITE_MODEL,
            )
        except Exception as e:
            logger.warning(f"[HEADINGS] detection call failed: {e}")
            return []

        cleaned = Utils.clean_json_fence(raw or "")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

        parsed = None
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            start, end = cleaned.find("["), cleaned.rfind("]")
            if 0 <= start < end:
                try:
                    parsed = json.loads(cleaned[start:end + 1])
                except json.JSONDecodeError:
                    parsed = None

        if not isinstance(parsed, list):
            logger.warning(f"[HEADINGS] non-list response; preview: {(raw or '')[:200]}")
            return []

        cleaned_headings: List[Dict[str, Any]] = []
        allowed_types = {"global", "page", "section", "footer", "bridge"}
        allowed_content_types = {"metrics", "narrative", "instructions", "comparisons", "legal", "concept", "other"}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                ln = int(item.get("line_num"))
            except (TypeError, ValueError):
                continue
            if ln < 0:
                continue
            ht = str(item.get("heading_text") or "").strip()
            if not ht:
                continue
            lvl = item.get("level", 1)
            try:
                lvl = max(1, min(3, int(lvl)))
            except (TypeError, ValueError):
                lvl = 1
            htype = str(item.get("heading_type", "section")).strip().lower()
            if htype not in allowed_types:
                htype = "section"
            ctype = str(item.get("content_type", "other")).strip().lower()
            if ctype not in allowed_content_types:
                ctype = "other"
            cleaned_headings.append({
                "line_num": ln,
                "heading_text": ht,
                "level": lvl,
                "heading_type": htype,
                "content_type": ctype,
            })

        cleaned_headings.sort(key=lambda h: h["line_num"])
        # De-duplicate consecutive headings on the same line (LLM occasionally repeats)
        deduped: List[Dict[str, Any]] = []
        seen_ln = -1
        for h in cleaned_headings:
            if h["line_num"] == seen_ln:
                continue
            deduped.append(h)
            seen_ln = h["line_num"]
        return deduped

    @staticmethod
    def _resolve_globals(headings: List[Dict[str, Any]]) -> List[str]:
        seen_norm: set = set()
        globals_list: List[str] = []
        for h in headings:
            if h.get("heading_type") != "global":
                continue
            text = h["heading_text"].strip()
            if not text:
                continue
            norm = re.sub(r"\s+", " ", text.lower())
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            globals_list.append(text)
        return globals_list

    async def _resolve_universal_headings_async(
        self,
        all_headings: List[Dict[str, Any]],
        source_info: str = "",
    ) -> List[str]:
        globals_list = [h for h in all_headings if h.get("heading_type") == "global"]
        footers_list = [h for h in all_headings if h.get("heading_type") == "footer"]
        if not globals_list and not footers_list:
            return []

        try:
            prompt = PromptFactory.universal_headings_dedupe(
                [{"title": h.get("heading_text"), "line_num": h.get("line_num")} for h in globals_list],
                [{"title": h.get("heading_text"), "line_num": h.get("line_num")} for h in footers_list],
                source_info=source_info,
            )
            raw, _, _ = await self._track_and_call_gemini(
                prompt, "universal_headings_dedupe",
                timeout=self.cfg.GEMINI_TIMEOUT_STANDARD,
                model=self.cfg.GEMINI_LITE_MODEL,
            )
            cleaned = Utils.clean_json_fence(raw or "")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                start, end = cleaned.find("{"), cleaned.rfind("}")
                data = json.loads(cleaned[start:end + 1]) if 0 <= start < end else None

            if isinstance(data, dict):
                items = data.get("universal_headings") or []
                out: List[str] = []
                for it in items:
                    if isinstance(it, dict) and it.get("title"):
                        t = str(it["title"]).strip()
                        if t and t not in out:
                            out.append(t)
                if out:
                    logger.info(f"[UNIVERSAL] LLM-resolved {len(out)} canonical universal heading(s): {out}")
                    return out
        except Exception as e:
            logger.warning(f"[UNIVERSAL] LLM resolution failed: {e}; using deterministic fallback")

        return self._resolve_globals(all_headings)

    async def _extract_structured_tables_async(
        self,
        chunks: List[Dict[str, Any]],
    ) -> None:
        if not self.cfg.STRUCTURED_TABLE_EXTRACTION:
            return

        candidate_idxs = [
            i for i, c in enumerate(chunks)
            if c.get("content_type") in self.cfg.STRUCTURED_TABLE_CONTENT_TYPES
        ]
        if not candidate_idxs:
            logger.info("[TABLES] no metrics/comparisons chunks; structured extraction skipped")
            return

        logger.info(f"[TABLES] LLM extraction for {len(candidate_idxs)} candidate chunks "
                    f"(content_type ∈ {list(self.cfg.STRUCTURED_TABLE_CONTENT_TYPES)})")

        batch_size = max(1, self.cfg.STRUCTURED_TABLE_BATCH_SIZE)
        batches: List[List[int]] = [
            candidate_idxs[i:i + batch_size]
            for i in range(0, len(candidate_idxs), batch_size)
        ]

        async def _run(batch_idxs: List[int]) -> None:
            batch = [chunks[i] for i in batch_idxs]
            prompt = PromptFactory.table_extraction_batch(batch)
            try:
                raw, _, _ = await self._track_and_call_gemini(
                    prompt, f"table_extraction_batch_{len(batch)}",
                    timeout=self.cfg.GEMINI_TIMEOUT_LARGE,
                    model=self.cfg.GEMINI_METADATA_MODEL,
                )
            except Exception as e:
                logger.warning(f"[TABLES] batch call failed: {e}; chunks will lack structured_tables")
                return

            cleaned = Utils.clean_json_fence(raw or "")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            parsed = None
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                a, b = cleaned.find("["), cleaned.rfind("]")
                if 0 <= a < b:
                    try:
                        parsed = json.loads(cleaned[a:b + 1])
                    except json.JSONDecodeError:
                        parsed = None
            if not isinstance(parsed, list):
                logger.warning(f"[TABLES] non-list response; skipping batch")
                return

            for pos, global_idx in enumerate(batch_idxs):
                if pos >= len(parsed) or not isinstance(parsed[pos], dict):
                    continue
                tables = parsed[pos].get("tables", [])
                if not isinstance(tables, list):
                    continue
                # Normalize: only keep tables with valid headers + rows
                cleaned_tables: List[Dict[str, Any]] = []
                for t in tables:
                    if not isinstance(t, dict):
                        continue
                    headers = t.get("headers") or []
                    rows = t.get("rows") or []
                    if not isinstance(headers, list) or not isinstance(rows, list):
                        continue
                    if not headers:
                        continue
                    cleaned_tables.append({
                        "title": str(t.get("title", "") or ""),
                        "headers": [str(h) for h in headers],
                        "rows": [[str(cell) for cell in (r or [])] for r in rows if isinstance(r, list)],
                        "notes": str(t.get("notes", "") or ""),
                    })
                # Store as JSON string for stable pandas serialization
                chunks[global_idx]["structured_tables"] = json.dumps(cleaned_tables, default=str)

        await asyncio.gather(*[_run(b) for b in batches])

        dropped_tables = 0
        flagged_tables = 0
        if self.cfg.STRUCTURED_TABLE_VALIDATE:
            for i in candidate_idxs:
                st = chunks[i].get("structured_tables")
                if not st or st == "[]":
                    continue
                try:
                    tables = json.loads(st)
                except Exception:
                    continue
                if not isinstance(tables, list) or not tables:
                    continue
                source_text = (chunks[i].get("chunk_text") or "")
                # Normalize source: collapse whitespace, lowercase for matching
                source_norm = re.sub(r"\s+", " ", source_text.lower())
                kept_tables: List[Dict[str, Any]] = []
                for t in tables:
                    headers = t.get("headers", []) or []
                    rows = t.get("rows", []) or []
                    # Collect non-trivial cell values (skip empty and very short)
                    cells_to_check: List[str] = []
                    for h in headers:
                        if isinstance(h, str) and len(h.strip()) >= 3:
                            cells_to_check.append(h)
                    for r in rows:
                        if not isinstance(r, list):
                            continue
                        for c in r:
                            if isinstance(c, str) and len(c.strip()) >= 3:
                                cells_to_check.append(c)
                    if not cells_to_check:
                        kept_tables.append(t)
                        continue
                    ungrounded = 0
                    for c in cells_to_check:
                        c_norm = re.sub(r"\s+", " ", c.lower()).strip()
                        grounded = False
                        if c_norm in source_norm:
                            grounded = True
                        else:
                            # Try substring checks with cell fragments
                            toks = re.findall(r"[\w]+(?:[.,/\-][\w]+)*", c_norm)
                            long_toks = [t for t in toks if len(t) >= 4]
                            if long_toks and all(t in source_norm for t in long_toks):
                                grounded = True
                        if not grounded:
                            ungrounded += 1
                    ungrounded_frac = ungrounded / len(cells_to_check)
                    if ungrounded_frac > self.cfg.STRUCTURED_TABLE_MAX_UNGROUNDED_FRACTION:
                        # Drop this table entirely — too much hallucination risk
                        dropped_tables += 1
                        logger.warning(
                            f"[TABLES] dropping hallucinated table on chunk {chunks[i].get('chunk_number')}: "
                            f"{ungrounded}/{len(cells_to_check)} cells ({ungrounded_frac:.0%}) "
                            f"not grounded in source"
                        )
                        continue
                    elif ungrounded_frac > 0.1:
                        # Keep but flag in notes
                        flagged_tables += 1
                        prior_notes = t.get("notes", "") or ""
                        t = {**t, "notes": (
                            f"{prior_notes} [validation: {ungrounded}/{len(cells_to_check)} "
                            f"cells ({ungrounded_frac:.0%}) not directly grounded in source — "
                            f"verify before use]"
                        ).strip()}
                    kept_tables.append(t)
                chunks[i]["structured_tables"] = json.dumps(kept_tables, default=str)

        merged_bridges = 0
        if self.cfg.STRUCTURED_TABLE_BRIDGE:
            merged_bridges = self._bridge_cross_chunk_tables(chunks, candidate_idxs)

        with_tables = sum(1 for i in candidate_idxs
                          if chunks[i].get("structured_tables") and chunks[i]["structured_tables"] != "[]")
        logger.info(f"[TABLES] extracted tables on {with_tables}/{len(candidate_idxs)} candidate chunks "
                    f"(validation: {dropped_tables} tables dropped as hallucinated, "
                    f"{flagged_tables} flagged; {merged_bridges} cross-chunk table bridges stitched)")

    def _bridge_cross_chunk_tables(
        self,
        chunks: List[Dict[str, Any]],
        candidate_idxs: List[int],
    ) -> int:

        def _normalize_headers(hs: List[str]) -> List[str]:
            return [re.sub(r"\s+", " ", str(h or "").lower()).strip() for h in hs]

        def _header_similarity(a: List[str], b: List[str]) -> float:
            na = _normalize_headers(a)
            nb = _normalize_headers(b)
            if not na:
                return 0.0
            if not nb or all(not h for h in nb):
                return 1.0 if len(nb) == len(na) else 0.0
            set_a, set_b = set(na), set(nb)
            set_a.discard("")
            set_b.discard("")
            if not set_a or not set_b:
                return 0.0
            return len(set_a & set_b) / len(set_a | set_b)

        threshold = self.cfg.STRUCTURED_TABLE_BRIDGE_HEADER_MATCH_THRESHOLD
        merged_count = 0

        for pos in range(len(candidate_idxs) - 1):
            i_cur = candidate_idxs[pos]
            i_next = candidate_idxs[pos + 1]
            cur_st = chunks[i_cur].get("structured_tables")
            nxt_st = chunks[i_next].get("structured_tables")
            if not cur_st or cur_st == "[]" or not nxt_st or nxt_st == "[]":
                continue
            try:
                cur_tables = json.loads(cur_st)
                nxt_tables = json.loads(nxt_st)
            except Exception:
                continue
            if not isinstance(cur_tables, list) or not isinstance(nxt_tables, list):
                continue

            for ci, ct in enumerate(cur_tables):
                best_nj = -1
                best_sim = 0.0
                for nj, nt in enumerate(nxt_tables):
                    sim = _header_similarity(ct.get("headers", []), nt.get("headers", []))
                    if sim > best_sim:
                        best_sim = sim
                        best_nj = nj
                if best_sim >= threshold and best_nj >= 0:
                    nt = nxt_tables[best_nj]
                    ct_header_norm = tuple(_normalize_headers(ct.get("headers", [])))
                    appended = 0
                    for row in nt.get("rows", []) or []:
                        if not isinstance(row, list):
                            continue
                        row_norm = tuple(_normalize_headers([str(c) for c in row]))
                        if row_norm == ct_header_norm:
                            continue
                        ct.setdefault("rows", []).append([str(c) for c in row])
                        appended += 1
                    if appended == 0:
                        continue
                    # Annotate provenance on both sides.
                    prior_notes = ct.get("notes", "") or ""
                    ct["notes"] = (prior_notes + f" [bridged: +{appended} rows from chunk "
                                   f"{chunks[i_next].get('chunk_number', '?')}]").strip()
                    nxt_tables[best_nj] = {
                        "title": nt.get("title", ""),
                        "headers": nt.get("headers", []),
                        "rows": [],
                        "notes": (nt.get("notes", "") + f" [continuation_of: chunk "
                                  f"{chunks[i_cur].get('chunk_number', '?')}]").strip(),
                        "continuation_of": str(chunks[i_cur].get("chunk_number", "")),
                    }
                    merged_count += 1
                    # Persist both sides back
                    chunks[i_cur]["structured_tables"] = json.dumps(cur_tables, default=str)
                    chunks[i_next]["structured_tables"] = json.dumps(nxt_tables, default=str)
                    break  # one bridge per pair is enough
        return merged_count

    def _attach_csv_excel_phase4_fields(
        self,
        chunk_df: "pd.DataFrame",
        source_df: Optional["pd.DataFrame"],
        file_type: str,
    ) -> "pd.DataFrame":
        if chunk_df.empty:
            return chunk_df

        # content_type default
        if "content_type" not in chunk_df.columns:
            chunk_df["content_type"] = "metrics"
        else:
            chunk_df["content_type"] = chunk_df["content_type"].fillna("").astype(str).replace("", "metrics")

        # potential_questions default
        if "potential_questions" not in chunk_df.columns:
            chunk_df["potential_questions"] = [[] for _ in range(len(chunk_df))]

        # structured_tables: deterministic emission from source rows
        def _slice_to_table(start_row, end_row) -> str:
            if source_df is None:
                return "[]"
            try:
                sr = int(start_row)
                er = int(end_row)
            except (TypeError, ValueError):
                return "[]"
            if sr < 0 or er < sr or sr >= len(source_df):
                return "[]"
            er = min(er, len(source_df) - 1)
            slice_df = source_df.iloc[sr:er + 1]
            headers = [str(h) for h in source_df.columns.tolist()]
            rows = [[str(cell) for cell in row] for row in slice_df.values.tolist()]
            table = {
                "title": f"{file_type.upper()} rows {sr}-{er}",
                "headers": headers,
                "rows": rows,
                "notes": f"deterministic extraction from source {file_type} (no LLM)",
            }
            return json.dumps([table], default=str)

        if "structured_tables" not in chunk_df.columns:
            if source_df is not None and "start_row" in chunk_df.columns and "end_row" in chunk_df.columns:
                chunk_df["structured_tables"] = chunk_df.apply(
                    lambda r: _slice_to_table(r.get("start_row"), r.get("end_row")),
                    axis=1,
                )
            else:
                chunk_df["structured_tables"] = "[]"

        return chunk_df

    async def _verify_chunks_async(
        self,
        chunks: List[Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        if not chunks:
            return {}
        batch_size = max(1, self.cfg.VERIFY_BATCH_SIZE)
        batches = [
            (start, chunks[start:start + batch_size])
            for start in range(0, len(chunks), batch_size)
        ]

        async def _run(start_idx: int, batch: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
            prompt = PromptFactory.chunk_verification_batch(batch)
            try:
                raw, _, _ = await self._track_and_call_gemini(
                    prompt, f"chunk_verification_batch_{len(batch)}",
                    timeout=self.cfg.GEMINI_TIMEOUT_LARGE,
                    model=self.cfg.GEMINI_LITE_MODEL,
                )
            except Exception as e:
                logger.warning(f"[VERIFY] batch call failed: {e}; skipping")
                return {}

            cleaned = Utils.clean_json_fence(raw or "")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            parsed = None
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                a, b = cleaned.find("["), cleaned.rfind("]")
                if 0 <= a < b:
                    try:
                        parsed = json.loads(cleaned[a:b + 1])
                    except json.JSONDecodeError:
                        parsed = None
            if not isinstance(parsed, list):
                logger.warning(f"[VERIFY] non-list response for batch@{start_idx}")
                return {}

            out: Dict[int, Dict[str, Any]] = {}
            for i, item in enumerate(parsed):
                if not isinstance(item, dict):
                    continue
                idx = start_idx + i
                if idx >= len(chunks):
                    break
                grounded_raw = item.get("grounded", True)
                grounded = bool(grounded_raw) if not isinstance(grounded_raw, str) \
                    else grounded_raw.strip().lower() in ("true", "yes", "1")
                issues = item.get("issues") or []
                if isinstance(issues, str):
                    issues = [issues]
                severity = str(item.get("severity", "ok")).strip().lower()
                if severity not in ("ok", "minor", "major"):
                    severity = "ok" if grounded else "minor"
                out[idx] = {
                    "grounded": grounded,
                    "issues": list(issues),
                    "severity": severity,
                }
            return out

        verdicts: Dict[int, Dict[str, Any]] = {}
        results = await asyncio.gather(*[_run(s, b) for s, b in batches])
        for r in results:
            verdicts.update(r)
        return verdicts

    async def _retry_metadata_for_failed_async(
        self,
        all_chunks: List[Dict[str, Any]],
        failed_indices: List[int],
        issues_by_chunk: Dict[int, List[str]],
        doc_summary: Optional[str],
    ) -> None:
        if not failed_indices:
            return
        failed_chunks = [all_chunks[i] for i in failed_indices]

        batch_size = max(1, self.cfg.CHUNK_METADATA_BATCH_SIZE)
        batches: List[Tuple[List[int], List[Dict[str, Any]]]] = []
        for i in range(0, len(failed_chunks), batch_size):
            bi = failed_indices[i:i + batch_size]
            bc = failed_chunks[i:i + batch_size]
            batches.append((bi, bc))

        declared_fields: frozenset = frozenset(
            re.findall(r'["\']([a-z][a-z0-9_]*)["\']\s*:', self.custom_instructions or "")
        )

        async def _run(batch_indices: List[int], batch: List[Dict[str, Any]]):
            issues_by_pos = {pos: issues_by_chunk.get(bi, [])
                             for pos, bi in enumerate(batch_indices)}
            prompt = PromptFactory.chunk_metadata_batch_retry(
                batch, issues_by_index=issues_by_pos,
                doc_summary=doc_summary, custom_instructions=self.custom_instructions,
            )
            try:
                raw, _, _ = await self._track_and_call_gemini(
                    prompt, f"chunk_metadata_retry_{len(batch)}",
                    timeout=self.cfg.GEMINI_TIMEOUT_LARGE,
                    model=self.cfg.GEMINI_METADATA_MODEL,
                )
            except Exception as e:
                logger.warning(f"[VERIFY-RETRY] batch call failed: {e}; leaving chunks as-is")
                return

            cleaned = Utils.clean_json_fence(raw or "")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            parsed = None
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                a, b = cleaned.find("["), cleaned.rfind("]")
                if 0 <= a < b:
                    try:
                        parsed = json.loads(cleaned[a:b + 1])
                    except json.JSONDecodeError:
                        parsed = None
            if not isinstance(parsed, list):
                logger.warning(f"[VERIFY-RETRY] non-list response; skipping batch")
                return

            for pos, global_idx in enumerate(batch_indices):
                if pos >= len(parsed) or not isinstance(parsed[pos], dict):
                    continue
                merged = self._merge_chunk_metadata(
                    all_chunks[global_idx], parsed[pos], declared_fields,
                )
                # Update in place (preserve chunk_text, page_number, etc.)
                all_chunks[global_idx] = merged

        await asyncio.gather(*[_run(bi, bc) for bi, bc in batches])

    async def _verify_and_retry_chunks_async(
        self,
        chunks: List[Dict[str, Any]],
        doc_summary: Optional[str],
    ) -> Dict[str, Any]:
        if not self.cfg.VERIFY_CHUNKS or not chunks:
            return {"sampled": 0, "flagged": 0, "retried": 0, "final_drift": 0}

        import random
        import time as _time
        _loop_start = _time.time()
        _loop_start_cost = (self._total_input_tokens + self._total_output_tokens)  # token snapshot

        def _cost_so_far_usd() -> float:
            # Rough: use PricingConfig to estimate cost spent in this verify loop only.
            input_delta = self._total_input_tokens - _loop_start_cost
            return max(0.0, input_delta / 1_000_000) * max(
                self.pricing.INPUT_PRICE_PER_1M_TOKENS,
                0.01,
            )

        def _wallclock_elapsed() -> float:
            return _time.time() - _loop_start

        # --- Sample selection (deterministic seed) ---
        n = len(chunks)
        sample_n = max(self.cfg.VERIFY_MIN_SAMPLE, int(n * self.cfg.VERIFY_SAMPLE_RATE))
        sample_n = min(sample_n, n)
        rng = random.Random(sum(len(c.get("chunk_text") or "") for c in chunks))
        sampled_indices = sorted(rng.sample(range(n), sample_n))
        logger.info(f"[VERIFY] sampling {sample_n}/{n} chunks for groundedness verification")

        sample_chunks = [chunks[i] for i in sampled_indices]
        verdicts = await self._verify_chunks_async(sample_chunks)
        global_verdicts: Dict[int, Dict[str, Any]] = {
            sampled_indices[local_i]: v for local_i, v in verdicts.items()
        }

        major_failed = [i for i, v in global_verdicts.items() if v.get("severity") == "major"]
        minor_drift = [i for i, v in global_verdicts.items() if v.get("severity") == "minor"]
        drift_fraction = len(major_failed) / max(1, sample_n)
        ok_count = sum(1 for v in global_verdicts.values() if v.get("severity") == "ok")
        logger.info(
            f"[VERIFY] verdicts — ok={ok_count}, minor={len(minor_drift)}, major={len(major_failed)}. "
            f"drift={drift_fraction:.0%} (threshold={self.cfg.VERIFY_DRIFT_THRESHOLD:.0%}, "
            f"floor={self.cfg.VERIFY_DRIFT_FLOOR:.0%})"
        )

        retried_count = 0
        stop_reason: Optional[str] = None

        # --- Rail 1: floor guard — don't retry a near-perfect output ---
        if drift_fraction < self.cfg.VERIFY_DRIFT_FLOOR:
            stop_reason = f"drift {drift_fraction:.0%} below floor {self.cfg.VERIFY_DRIFT_FLOOR:.0%}"
            logger.info(f"[VERIFY] {stop_reason}; skipping retry")

        # --- Rail 2: normal threshold check ---
        elif drift_fraction < self.cfg.VERIFY_DRIFT_THRESHOLD or not major_failed:
            stop_reason = "drift below retry threshold"

        else:
            # --- Rail 3: systemic-failure detection ---
            failed_content_types = [
                chunks[i].get("content_type") or "?" for i in major_failed
            ]
            if len(set(failed_content_types)) == 1 and len(failed_content_types) >= 3:
                stop_reason = (
                    f"systemic failure — all {len(major_failed)} major failures are "
                    f"content_type={failed_content_types[0]!r}; retrying won't help, "
                    f"flag for prompt-level review"
                )
                logger.warning(f"[VERIFY] {stop_reason}")

            else:
                # --- Retry loop, with rails on every round ---
                logger.info(
                    f"[VERIFY] drift exceeds threshold; starting retry loop "
                    f"(rails: max_retried={self.cfg.VERIFY_MAX_CHUNKS_RETRIED}, "
                    f"max_cost=${self.cfg.VERIFY_MAX_COST_USD:.2f}, "
                    f"wallclock={self.cfg.VERIFY_MAX_WALLCLOCK_SEC}s)"
                )
                issues_by_chunk = {i: global_verdicts[i].get("issues", []) for i in major_failed}
                remaining = list(major_failed)
                prev_bad_count = len(remaining)

                for attempt in range(self.cfg.VERIFY_MAX_RETRIES):
                    # Rail 3: retried-count cap
                    if retried_count + len(remaining) > self.cfg.VERIFY_MAX_CHUNKS_RETRIED:
                        stop_reason = (
                            f"retried-chunks cap hit ({retried_count + len(remaining)} > "
                            f"{self.cfg.VERIFY_MAX_CHUNKS_RETRIED})"
                        )
                        logger.warning(f"[VERIFY] {stop_reason}")
                        break

                    # Rail 4: cost cap
                    if _cost_so_far_usd() > self.cfg.VERIFY_MAX_COST_USD:
                        stop_reason = (
                            f"cost cap hit (${_cost_so_far_usd():.3f} > "
                            f"${self.cfg.VERIFY_MAX_COST_USD:.2f})"
                        )
                        logger.warning(f"[VERIFY] {stop_reason}")
                        break

                    # Rail 5: wall-clock cap
                    if _wallclock_elapsed() > self.cfg.VERIFY_MAX_WALLCLOCK_SEC:
                        stop_reason = (
                            f"wallclock cap hit ({_wallclock_elapsed():.0f}s > "
                            f"{self.cfg.VERIFY_MAX_WALLCLOCK_SEC}s)"
                        )
                        logger.warning(f"[VERIFY] {stop_reason}")
                        break

                    await self._retry_metadata_for_failed_async(
                        chunks, remaining, issues_by_chunk, doc_summary,
                    )
                    retried_count += len(remaining)

                    re_sample = [chunks[i] for i in remaining]
                    re_verdicts = await self._verify_chunks_async(re_sample)
                    still_bad = [
                        remaining[li] for li, v in re_verdicts.items()
                        if v.get("severity") == "major"
                    ]
                    logger.info(
                        f"[VERIFY] retry round {attempt + 1}: "
                        f"{len(still_bad)}/{len(remaining)} still major-failing"
                    )

                    # Rail 6: monotonic-improvement check
                    if self.cfg.VERIFY_REQUIRE_MONOTONIC and len(still_bad) >= prev_bad_count:
                        stop_reason = (
                            f"monotonic-improvement rail: retry round {attempt + 1} did not "
                            f"reduce failure count ({prev_bad_count} → {len(still_bad)}); "
                            f"further retries unlikely to help"
                        )
                        logger.warning(f"[VERIFY] {stop_reason}")
                        break

                    prev_bad_count = len(still_bad)
                    if not still_bad:
                        stop_reason = "all retried chunks now ok"
                        break
                    issues_by_chunk = {
                        remaining[li]: re_verdicts[li].get("issues", [])
                        for li in re_verdicts if re_verdicts[li].get("severity") == "major"
                    }
                    remaining = still_bad

                if stop_reason is None:
                    stop_reason = f"exhausted {self.cfg.VERIFY_MAX_RETRIES} retry round(s)"

        return {
            "sampled": sample_n,
            "ok": ok_count,
            "minor": len(minor_drift),
            "major": len(major_failed),
            "initial_drift": drift_fraction,
            "retried": retried_count,
            "stop_reason": stop_reason or "no retry triggered",
            "wallclock_s": _wallclock_elapsed(),
        }

    async def _classify_bridge_async(
        self,
        last_heading: Dict[str, Any],
        last_heading_text: Dict[int, str],
        next_prelude_text: Dict[int, str],
        current_block_headings: List[Dict[str, Any]],
        universal_headings: List[str],
    ) -> Optional[Dict[str, Any]]:
        if not next_prelude_text:
            return None
        try:
            prompt = PromptFactory.bridge_classification(
                current_page_last_heading=last_heading.get("heading_text", ""),
                current_page_last_heading_text=last_heading_text,
                next_page_first_heading_text=next_prelude_text,
                current_page_headings=current_block_headings,
                universal_headings=universal_headings,
            )
            raw, _, _ = await self._track_and_call_gemini(
                prompt, "bridge_classification",
                timeout=self.cfg.GEMINI_TIMEOUT_STANDARD,
                model=self.cfg.GEMINI_LITE_MODEL,
            )
            cleaned = Utils.clean_json_fence(raw or "")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                start, end = cleaned.find("{"), cleaned.rfind("}")
                if 0 <= start < end:
                    data = json.loads(cleaned[start:end + 1])
                else:
                    return None
            if not isinstance(data, dict):
                return None
            bridging = str(data.get("bridging", "false")).strip().lower()
            if bridging not in ("true", "false", "relevant"):
                return None
            if bridging == "false":
                return None
            return {
                "bridging": bridging,
                "start_index": data.get("start_index"),
                "rationale": str(data.get("rationale", "")),
                "most_relevant_headings": data.get("most_relevant_headings", []) or [],
            }
        except Exception as e:
            logger.warning(f"[BRIDGE] classification failed: {e}")
            return None

    def _compute_sections(
        self,
        headings: List[Dict[str, Any]],
        lines: List[str],
        line_pages: List[int],
    ) -> List[Dict[str, Any]]:
        n = len(lines)
        boundary_types = {"section", "page", "bridge"}
        boundary_headings = [h for h in headings if h.get("heading_type") in boundary_types]

        if not boundary_headings:
            return [{
                "start": 0,
                "end": n - 1,
                "heading_text": "",
                "level": 1,
                "heading_type": "section",
                "page_number": line_pages[0] if line_pages else 0,
            }]

        # Prepend a synthetic preamble section if there's content before the first boundary
        sections: List[Dict[str, Any]] = []
        first_hn = min(n - 1, max(0, boundary_headings[0]["line_num"]))
        if first_hn > 0:
            sections.append({
                "start": 0,
                "end": first_hn - 1,
                "heading_text": "",
                "level": 1,
                "heading_type": "section",
                "page_number": line_pages[0] if line_pages else 0,
            })

        for i, h in enumerate(boundary_headings):
            start = max(0, min(n - 1, h["line_num"]))
            if i + 1 < len(boundary_headings):
                end = max(start, min(n - 1, boundary_headings[i + 1]["line_num"] - 1))
            else:
                end = n - 1
            sections.append({
                "start": start,
                "end": end,
                "heading_text": h["heading_text"],
                "level": h["level"],
                "heading_type": h.get("heading_type", "section"),
                "content_type": h.get("content_type", "other"),
                "page_number": line_pages[start] if start < len(line_pages) else 0,
            })

        # Merge sections shorter than the minimum with the *next* section
        merged: List[Dict[str, Any]] = []
        pending: Optional[Dict[str, Any]] = None
        for s in sections:
            section_len = s["end"] - s["start"] + 1
            if pending is not None:
                # Extend pending backwards-compatibly
                s = {**s, "start": pending["start"]}
                if not s["heading_text"] and pending["heading_text"]:
                    s["heading_text"] = pending["heading_text"]
                pending = None
            if section_len < self.cfg.HEADING_MIN_SECTION_LINES and s is not sections[-1]:
                pending = s
                continue
            merged.append(s)
        if pending is not None:
            if merged:
                merged[-1]["end"] = pending["end"]
            else:
                merged.append(pending)

        # Split sections longer than the maximum
        final: List[Dict[str, Any]] = []
        for s in merged:
            section_len = s["end"] - s["start"] + 1
            if section_len <= self.cfg.HEADING_MAX_SECTION_LINES:
                final.append(s)
                continue
            # Equal-ish split
            n_parts = max(2, (section_len + self.cfg.HEADING_MAX_SECTION_LINES - 1) // self.cfg.HEADING_MAX_SECTION_LINES)
            part_len = section_len // n_parts
            base_heading = s["heading_text"] or "(unheaded section)"
            for k in range(n_parts):
                sub_start = s["start"] + k * part_len
                sub_end = s["end"] if k == n_parts - 1 else sub_start + part_len - 1
                final.append({
                    "start": sub_start,
                    "end": sub_end,
                    "heading_text": f"{base_heading} (part {k + 1}/{n_parts})" if n_parts > 1 else base_heading,
                    "level": s["level"],
                    "heading_type": s.get("heading_type", "section"),
                    "content_type": s.get("content_type", "other"),
                    "page_number": line_pages[sub_start] if sub_start < len(line_pages) else s["page_number"],
                })

        return final

    # C0 control chars (U+0000..U+001F) minus \t \n \r. PDF extraction
    # occasionally bleeds NULs and other control bytes which later break
    # strict JSON parsers and embedding APIs. Stripped at assembly time.
    _C0_STRIP_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

    def _assemble_chunk_text(self, lines: List[str], start: int, end: int,
                             gap_index_range: Optional[List[tuple]] = None) -> str:
        excluded: set = set()
        if gap_index_range:
            for lo, hi in gap_index_range:
                for i in range(lo, hi + 1):
                    excluded.add(i)
        out: List[str] = []
        for i in range(start, min(end + 1, len(lines))):
            if i in excluded:
                continue
            ln = lines[i]
            if self._PAGE_MARKER_RE.match(ln):
                continue
            out.append(ln)
        text = "\n".join(out).strip()
        return self._C0_STRIP_RE.sub("", text)

    @staticmethod
    def _must_be_separate_chunk(content_type: str, section_len: int) -> bool:
        strict = {"metrics", "instructions", "concept"}
        lenient = {"narrative", "comparisons", "legal"}
        if content_type in strict:
            return section_len > 3
        if content_type in lenient:
            return section_len > 10
        return section_len > 10

    def _merge_single_line_chunks(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(sections) < 2:
            return sections
        out = list(sections)
        # First-chunk merge
        first = out[0]
        if first["end"] - first["start"] < 1:
            nxt = out[1]
            out[1] = {**nxt, "start": first["start"],
                      "heading_text": nxt["heading_text"] or first["heading_text"]}
            out.pop(0)
        if len(out) < 2:
            return out
        # Last-chunk merge
        last = out[-1]
        if last["end"] - last["start"] < 1:
            prev = out[-2]
            out[-2] = {**prev, "end": last["end"]}
            out.pop()
        return out

    # Stub threshold: a section with fewer than this many assembled chars
    # is almost certainly just a stray heading line with no body (e.g.
    # "Download Sample Notebook For Tutorial" alone). These MUST be merged.
    # Legitimate short sections (200-400 chars) are kept as standalone chunks
    # — the char-based merge used to collapse those transitively, eating
    # 40+ real section boundaries on structured tutorial docs.
    _STUB_MIN_CHARS: int = 100

    def _merge_small_sections_by_chars(
        self,
        sections: List[Dict[str, Any]],
        lines: List[str],
    ) -> List[Dict[str, Any]]:
        """Merge true stub sections (<_STUB_MIN_CHARS of assembled content)
        into their NEXT neighbor. Falls back to PREVIOUS neighbor for a stub
        at the last position. Single-pass, non-transitive — does NOT collapse
        a chain of small-but-legit sections into one giant chunk.
        """
        if len(sections) < 2:
            return sections
        stub_floor = self._STUB_MIN_CHARS
        out: List[Dict[str, Any]] = []
        absorb_next_with_start: Optional[int] = None

        for idx, s in enumerate(sections):
            # If previous iteration marked a stub to absorb into THIS section,
            # extend this section's start back to include the stub's range.
            if absorb_next_with_start is not None:
                s = {**s, "start": absorb_next_with_start}
                absorb_next_with_start = None
            span_chars = self._section_text_length(
                lines, s["start"], s["end"], s.get("gap_index_range")
            )
            is_last = (idx == len(sections) - 1)
            if span_chars < stub_floor and not is_last:
                # Signal next section to absorb this stub's line range.
                absorb_next_with_start = s["start"]
                continue
            out.append(s)
        # If the final section was itself a stub (would set absorb but has
        # no next), merge it into the previous.
        if absorb_next_with_start is not None:
            if out:
                out[-1]["end"] = sections[-1]["end"]
            else:
                out.append(sections[-1])
        return out

    _BRIDGE_HINT_RE = re.compile(
        r"\(cont(?:inued|\.|d)?\)|—\s*cont\b|\bcont\.\b",
        re.IGNORECASE,
    )

    def _resolve_cross_block_bridges(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def _root_key(section: str) -> str:
            # Heuristic: take up to the first 4 words, lowercase, strip punct.
            s = re.sub(r"[\(\)\[\]\.,;:]", " ", section or "").strip()
            toks = [t for t in s.split() if not self._BRIDGE_HINT_RE.fullmatch(t)]
            return " ".join(toks[:4]).lower()

        stitched = 0
        for i, c in enumerate(chunks):
            htype = c.get("heading_type")
            section = c.get("section") or ""
            is_bridge = htype == "bridge" or bool(self._BRIDGE_HINT_RE.search(section))
            if not is_bridge:
                continue
            bridge_key = _root_key(section)
            if not bridge_key:
                continue
            # Look back for a non-bridge chunk with matching root
            for j in range(i - 1, -1, -1):
                prev = chunks[j]
                if prev.get("heading_type") == "bridge":
                    continue
                prev_key = _root_key(prev.get("section") or "")
                if prev_key and (prev_key.startswith(bridge_key) or bridge_key.startswith(prev_key)):
                    parent = prev.get("parent_title") or prev.get("section") or ""
                    if parent and not c.get("parent_title"):
                        c["parent_title"] = parent
                        # Also prepend to the section label for retrieval context
                        if parent and parent not in section:
                            c["section"] = f"{parent} / {section}" if section else parent
                        stitched += 1
                    break
        if stitched:
            logger.info(f"[BRIDGE] Stitched {stitched} cross-block continuation chunks to their parents")
        return chunks

    def _carve_parent_child(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(sections) < 2:
            return sections
        out: List[Dict[str, Any]] = []
        i = 0
        while i < len(sections):
            cur = sections[i]
            parent_end = cur.get("end")
            parent_level = cur.get("level", 1)
            j = i + 1
            children_ranges: List[tuple] = []
            absorbed_children: List[Dict[str, Any]] = []
            parent_title = cur.get("heading_text", "")
            while j < len(sections):
                child = sections[j]
                if child.get("level", 1) <= parent_level:
                    break
                if child["end"] > parent_end:
                    break
                child_len = child["end"] - child["start"] + 1
                child_ct = child.get("content_type", "other")
                # Thesis rule: only force-carve when content type demands it.
                if self._must_be_separate_chunk(child_ct, child_len):
                    sections[j] = {**child, "parent_title": parent_title}
                    children_ranges.append((child["start"], child["end"]))
                    absorbed_children.append(sections[j])
                else:
                    pass
                j += 1
            if children_ranges:
                parent = dict(cur)
                parent["gap_index_range"] = children_ranges
                out.append(parent)
                out.extend(sections[i + 1:j])
            else:
                out.append(cur)
            i = j if children_ranges or j > i + 1 else i + 1
        return out

    # ----- Size-aware carving (deterministic) -----

    _SPLIT_PATTERNS: List[Tuple[str, "re.Pattern"]] = [
        # Nested numbered sections: "5.1 ", "5.1.2 ", "10.4.3  Title"
        ("nested_numeric",      re.compile(r"^\s{0,8}\d+(?:\.\d+){1,3}\s")),
        # Roman-numeral parenthesized clauses: "(i) ", "(ii) ", "(iv) "
        ("roman_clause",        re.compile(r"^\s{0,8}\([ivxlcdm]{1,6}\)\s")),
        # Letter-parenthesized clauses: "(a) ", "(b) "
        ("letter_clause",       re.compile(r"^\s{0,8}\([a-z]\)\s")),
        # Numeric-parenthesized clauses: "(1) ", "(2) "
        ("numeric_paren",       re.compile(r"^\s{0,8}\(\d{1,3}\)\s")),
        # Simple numbered list items: "1. ", "2. "
        ("numbered_list",       re.compile(r"^\s{0,8}\d+\.\s")),
        # Bullet list markers
        ("bullet",              re.compile(r"^\s{0,8}[•\-*]\s")),
    ]

    def _find_split_points(
        self,
        lines: List[str],
        start: int,
        end: int,
        gap_index_range: Optional[List[Tuple[int, int]]] = None,
    ) -> List[int]:
        excluded: set = set()
        if gap_index_range:
            for lo, hi in gap_index_range:
                for i in range(lo, hi + 1):
                    excluded.add(i)

        for _name, pat in self._SPLIT_PATTERNS:
            candidates: List[int] = []
            for i in range(start + 1, min(end + 1, len(lines))):  # skip the section's first line
                if i in excluded:
                    continue
                ln = lines[i]
                if self._PAGE_MARKER_RE.match(ln):
                    continue
                if pat.match(ln):
                    candidates.append(i)
            if len(candidates) >= 2:
                return candidates

        para_candidates: List[int] = []
        for i in range(start + 1, min(end + 1, len(lines)) - 1):
            if i in excluded:
                continue
            if not lines[i].strip():
                # Split point = next non-empty, non-excluded, non-marker line
                j = i + 1
                while j <= end and j < len(lines):
                    if j in excluded or self._PAGE_MARKER_RE.match(lines[j]) or not lines[j].strip():
                        j += 1
                        continue
                    para_candidates.append(j)
                    break
        if len(para_candidates) >= 2:
            return para_candidates

        # Last resort: sentence-boundary split. When the structural ladder
        # and paragraph breaks both return empty on an oversized section
        # (common in prose-heavy tutorials without bullets / numbered items),
        # fall through to splitting at sentence terminators. We only split
        # on lines that end with ". " / ".\n" / "!\n" / "?\n" so we don't
        # split mid-abbreviation. Emitted split points are the line AFTER
        # the terminator.
        sentence_end_re = re.compile(r'[.!?](?:\s|$)')
        sentence_candidates: List[int] = []
        for i in range(start + 1, min(end + 1, len(lines)) - 1):
            if i in excluded:
                continue
            ln = lines[i]
            if self._PAGE_MARKER_RE.match(ln):
                continue
            if not ln.strip():
                continue
            if sentence_end_re.search(ln.strip()[-3:]):
                # Candidate split is the NEXT non-excluded line
                j = i + 1
                while j <= end and j < len(lines):
                    if j in excluded or self._PAGE_MARKER_RE.match(lines[j]) or not lines[j].strip():
                        j += 1
                        continue
                    sentence_candidates.append(j)
                    break
        if len(sentence_candidates) >= 2:
            # Thin out sentence candidates to avoid hyper-fragmentation:
            # keep roughly every other one so sub-ranges are ~500-1000 chars.
            stride = max(1, len(sentence_candidates) // max(2, (end - start) // 800))
            thinned = sentence_candidates[::stride]
            if len(thinned) >= 2:
                return thinned
            return sentence_candidates

        return []

    def _section_text_length(
        self,
        lines: List[str],
        start: int,
        end: int,
        gap_index_range: Optional[List[Tuple[int, int]]] = None,
    ) -> int:
        excluded: set = set()
        if gap_index_range:
            for lo, hi in gap_index_range:
                for i in range(lo, hi + 1):
                    excluded.add(i)
        total = 0
        for i in range(start, min(end + 1, len(lines))):
            if i in excluded:
                continue
            ln = lines[i]
            if self._PAGE_MARKER_RE.match(ln):
                continue
            total += len(ln) + 1  # +1 for the newline we'd join with
        return total

    @staticmethod
    def _parent_section_id(s: Dict[str, Any]) -> str:
        base = f"{s.get('heading_text', '')}__L{s.get('start', 0)}-L{s.get('end', 0)}"
        return re.sub(r"\s+", "_", base.strip())[:128]

    def _fixed_interval_splits(
        self,
        lines: List[str],
        start: int,
        end: int,
        gap_index_range: Optional[List[Tuple[int, int]]] = None,
        target_chars: int = 900,
    ) -> List[int]:
        """Hard fallback: emit split points at line boundaries that accumulate
        to ~target_chars each. Used only when structural/paragraph/sentence
        fallbacks all returned empty (pathological prose like code dumps,
        transcripts, or long run-on strings with no terminators).
        """
        excluded: set = set()
        if gap_index_range:
            for lo, hi in gap_index_range:
                for i in range(lo, hi + 1):
                    excluded.add(i)
        candidates: List[int] = []
        accum = 0
        for i in range(start + 1, min(end + 1, len(lines))):
            if i in excluded:
                continue
            ln = lines[i]
            if self._PAGE_MARKER_RE.match(ln):
                continue
            accum += len(ln) + 1
            if accum >= target_chars:
                candidates.append(i)
                accum = 0
        # Only return if we produced enough to actually split.
        return candidates if len(candidates) >= 1 else []

    def _carve_by_size(
        self,
        sections: List[Dict[str, Any]],
        lines: List[str],
        line_pages: List[int],
    ) -> List[Dict[str, Any]]:
        max_chars = self.cfg.CHUNK_MAX_CHARS
        min_chars = self.cfg.CHUNK_MIN_CHARS
        out: List[Dict[str, Any]] = []
        for s in sections:
            gap = s.get("gap_index_range")
            section_len = self._section_text_length(lines, s["start"], s["end"], gap)
            if section_len <= max_chars:
                out.append(s)
                continue

            splits = self._find_split_points(lines, s["start"], s["end"], gap)
            if not splits:
                # Last-resort fixed-interval fallback. Oversized prose that
                # has no structural markers AND no sentence terminators
                # (rare — code snippets, transcript blocks) still needs to
                # be split. Target ~max_chars/2 per sub-range at line
                # boundaries.
                splits = self._fixed_interval_splits(lines, s["start"], s["end"], gap, max_chars // 2)
                if not splits:
                    out.append(s)
                    continue

            # Build sub-ranges: [start..splits[0]-1], [splits[0]..splits[1]-1], ..., [splits[-1]..end]
            boundaries = [s["start"]] + splits + [s["end"] + 1]
            raw_sub_ranges: List[Tuple[int, int]] = []
            for i in range(len(boundaries) - 1):
                sub_start = boundaries[i]
                sub_end = boundaries[i + 1] - 1
                if sub_end < sub_start:
                    continue
                raw_sub_ranges.append((sub_start, sub_end))

            # Merge tiny sub-ranges with a neighbor. Prefer merging forward
            # for the FIRST sub-range (no previous to merge into), backward
            # for any subsequent small sub-range.
            merged: List[Tuple[int, int]] = []
            pending_tiny_first: Optional[Tuple[int, int]] = None
            for idx, (a, b) in enumerate(raw_sub_ranges):
                sub_len = self._section_text_length(lines, a, b, gap)
                if sub_len < min_chars:
                    if merged:
                        # Backward-merge into previous.
                        prev_a, _prev_b = merged[-1]
                        merged[-1] = (prev_a, b)
                    elif pending_tiny_first is None:
                        # First sub-range is too small — hold it to forward-merge
                        # into whatever comes next.
                        pending_tiny_first = (a, b)
                    else:
                        # Two consecutive tiny leading sub-ranges — extend the pending.
                        pending_tiny_first = (pending_tiny_first[0], b)
                else:
                    if pending_tiny_first is not None:
                        # Forward-merge the pending tiny leading sub-range.
                        merged.append((pending_tiny_first[0], b))
                        pending_tiny_first = None
                    else:
                        merged.append((a, b))
            if pending_tiny_first is not None:
                # All sub-ranges were tiny (rare) — emit as single range.
                if merged:
                    merged[-1] = (pending_tiny_first[0], merged[-1][1])
                else:
                    merged.append(pending_tiny_first)

            if len(merged) <= 1:
                out.append(s)
                continue

            # Preserve existing parent_section_id if this section already
            # came from a prior size-carve pass (iterative carving). Only
            # generate a fresh psid when none exists. This keeps all
            # descendants of the ORIGINAL section grouped under one psid
            # regardless of how many passes it took to reach final size.
            psid = s.get("parent_section_id") or self._parent_section_id(s)
            for k, (a, b) in enumerate(merged):
                page_num = line_pages[a] if 0 <= a < len(line_pages) else s.get("page_number", 0)
                out.append({
                    **s,
                    "start": a,
                    "end": b,
                    "gap_index_range": [(lo, hi) for (lo, hi) in (gap or [])
                                        if a <= lo <= b or a <= hi <= b],
                    "parent_section_id": psid,
                    "split_index": k + 1,
                    "split_total": len(merged),
                    "page_number": page_num,
                })
        return out

    async def _enrich_chunks_metadata_async(
        self,
        chunks: List[Dict[str, Any]],
        doc_summary: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        declared_fields: frozenset = frozenset(
            re.findall(r'["\']([a-z][a-z0-9_]*)["\']\s*:', self.custom_instructions or "")
        )

        batch_size = max(1, self.cfg.CHUNK_METADATA_BATCH_SIZE)
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        logger.info(f"[META] Enriching {len(chunks)} chunks across {len(batches)} batches of {batch_size}; "
                    f"declared per-chunk fields from prose: {sorted(declared_fields) or '(none)'}")

        async def _run(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            prompt = PromptFactory.chunk_metadata_batch(
                batch, doc_summary=doc_summary, custom_instructions=self.custom_instructions
            )
            try:
                raw, _, _ = await self._track_and_call_gemini(
                    prompt, f"chunk_metadata_batch_{len(batch)}",
                    timeout=self.cfg.GEMINI_TIMEOUT_LARGE,
                    model=self.cfg.GEMINI_METADATA_MODEL,
                )
            except Exception as e:
                logger.warning(f"[META] batch call failed: {e}; using defaults")
                return [self._chunk_default_metadata(c) for c in batch]

            cleaned = Utils.clean_json_fence(raw or "")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            parsed = None
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                start, end = cleaned.find("["), cleaned.rfind("]")
                if 0 <= start < end:
                    try:
                        parsed = json.loads(cleaned[start:end + 1])
                    except json.JSONDecodeError:
                        parsed = None

            if not isinstance(parsed, list):
                logger.warning(f"[META] non-list metadata response; preview: {(raw or '')[:200]}")
                return [self._chunk_default_metadata(c) for c in batch]

            out: List[Dict[str, Any]] = []
            for i, c in enumerate(batch):
                meta = parsed[i] if i < len(parsed) and isinstance(parsed[i], dict) else {}
                out.append(self._merge_chunk_metadata(c, meta, declared_fields))
            return out

        results = await asyncio.gather(*[_run(b) for b in batches])
        flat: List[Dict[str, Any]] = []
        for r in results:
            flat.extend(r)
        return flat

    _CHUNK_BASELINE_META_KEYS = {
        "chunk_index", "topic", "summary", "sub_topics", "section_reference",
        "content_type", "potential_questions", "customer_specific_tags",
        "universal",
    }

    @staticmethod
    def _chunk_default_metadata(c: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **c,
            "topic": c.get("heading_text") or "",
            "summary": "",
            "sub_topics": "",
            "section_reference": "",
            "content_type": "other",
            "potential_questions": [],
            "customer_specific_tags": [],
            "universal": "",
            "chunker_result": "Success",
        }

    def _merge_chunk_metadata(self, c: Dict[str, Any], meta: Dict[str, Any],
                              declared_fields: frozenset) -> Dict[str, Any]:
        def _str(v): return str(v) if v is not None else ""
        def _arr(v):
            if isinstance(v, list):
                return v
            if isinstance(v, str) and v.strip():
                return [v]
            return []

        # Validate content_type against closed enum; fall back to "other".
        ct = _str(meta.get("content_type", "other")).strip().lower()
        if ct not in {"metrics", "narrative", "instructions", "comparisons", "legal", "concept", "other"}:
            ct = "other"

        extras: Dict[str, Any] = {}
        if declared_fields:
            _snake = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
            for k, v in meta.items():
                if k in self._CHUNK_BASELINE_META_KEYS:
                    continue
                if not isinstance(k, str) or not _snake.match(k):
                    continue
                if k not in declared_fields:
                    continue
                extras[k] = v

        # customer_specific_tags are only meaningful against a customer
        # taxonomy. Without customInstructions declaring one, the LLM
        # free-associates (Title Case vs snake_case, runaway cardinality).
        # Gate: emit tags only when instructions were provided.
        tags = _arr(meta.get("customer_specific_tags")) if self.custom_instructions else []

        return {
            **c,
            "topic": _str(meta.get("topic") or c.get("heading_text") or ""),
            "summary": _str(meta.get("summary")),
            "sub_topics": _str(meta.get("sub_topics")),
            "section_reference": _str(meta.get("section_reference")),
            "content_type": ct,
            "potential_questions": _arr(meta.get("potential_questions")),
            "customer_specific_tags": tags,
            "universal": _str(meta.get("universal")),
            "chunker_result": "Success",
            **extras,
        }

    async def _detect_block_headings_async(self, text: str) -> Dict[str, Any]:
        lines, line_pages, numbered = self._number_lines(text)
        logger.info(f"[HEADING-CHUNK] block: {len(lines)} lines, numbered_prompt={len(numbered)} chars")

        headings = await self._detect_headings_async(numbered)
        htype_counts: Dict[str, int] = {}
        for h in headings:
            htype_counts[h.get("heading_type", "section")] = htype_counts.get(h.get("heading_type", "section"), 0) + 1
        logger.info(f"[HEADING-CHUNK] detected {len(headings)} headings: {htype_counts}")

        return {
            "lines": lines,
            "line_pages": line_pages,
            "numbered": numbered,
            "headings": headings,
            "text": text,
        }

    async def _heading_driven_chunk_block_async(
        self,
        text: str,
        doc_summary: Optional[str],
        pre_detected: Optional[Dict[str, Any]] = None,
        universal_headings: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if pre_detected is None:
            pre_detected = await self._detect_block_headings_async(text)

        lines = pre_detected["lines"]
        line_pages = pre_detected["line_pages"]
        headings = pre_detected["headings"]

        # Auto-fallback: heading-poor blocks use density path for THIS block.
        boundary_headings = [h for h in headings if h.get("heading_type") in ("section", "page", "bridge")]
        if len(boundary_headings) < self.cfg.HEADING_MIN_HEADINGS_FOR_HEADING_PATH:
            logger.info(
                f"[HEADING-CHUNK] only {len(boundary_headings)} boundary headings "
                f"(< {self.cfg.HEADING_MIN_HEADINGS_FOR_HEADING_PATH}); "
                f"falling back to density-driven chunking for this block"
            )
            return await self._chunk_block_centralized(text, doc_summary, "pdf")

        if universal_headings is None:
            universal_headings = self._resolve_globals(headings)
        if universal_headings:
            logger.info(f"[HEADING-CHUNK] using universal headings: {universal_headings}")
        global_universal = " | ".join(universal_headings)

        sections = self._compute_sections(headings, lines, line_pages)
        sections = self._merge_single_line_chunks(sections)
        sections = self._merge_small_sections_by_chars(sections, lines)
        sections = self._carve_parent_child(sections)
        sections_before_size = len(sections)
        # Iterate size-carving up to a bounded number of passes to catch
        # oversized outliers that survived a single pass (one structural
        # split can still leave sub-ranges above max_chars when markers
        # are sparse). Fixed-interval fallback inside _carve_by_size
        # guarantees eventual convergence.
        max_carve_passes = 4
        max_chars = self.cfg.CHUNK_MAX_CHARS
        for pass_idx in range(max_carve_passes):
            sections = self._carve_by_size(sections, lines, line_pages)
            still_oversized = sum(
                1 for s in sections
                if self._section_text_length(
                    lines, s["start"], s["end"], s.get("gap_index_range")
                ) > max_chars
            )
            if still_oversized == 0:
                break
            logger.info(f"[HEADING-CHUNK] carve pass {pass_idx + 1}: {still_oversized} sections still >{max_chars} chars; re-carving")

        # Reflow split_index / split_total across all sections sharing the
        # same parent_section_id so numbering stays consistent after
        # iterative carving. Without this, a section that got re-carved in
        # pass 2 would have mixed numbering (e.g. 1/2, 2/2, 2/3, 3/3).
        from collections import OrderedDict
        groups: "OrderedDict[str, List[int]]" = OrderedDict()
        for idx, s in enumerate(sections):
            psid = s.get("parent_section_id")
            if psid:
                groups.setdefault(psid, []).append(idx)
        for psid, idxs in groups.items():
            total = len(idxs)
            if total <= 1:
                continue
            for order, idx in enumerate(idxs, start=1):
                sections[idx]["split_index"] = order
                sections[idx]["split_total"] = total

        split_count = sum(1 for s in sections if s.get("split_total", 1) > 1)
        logger.info(f"[HEADING-CHUNK] final sections after merge/carve: "
                    f"{len(sections)} ({sections_before_size} pre-size-split; "
                    f"{split_count} are size-split siblings)")

        proto_chunks: List[Dict[str, Any]] = []
        for i, s in enumerate(sections):
            chunk_text = self._assemble_chunk_text(
                lines, s["start"], s["end"],
                gap_index_range=s.get("gap_index_range"),
            )
            if not chunk_text.strip():
                continue
            base_universal = global_universal
            # Parent-child: prepend parent title to children's "section" for context
            section_label = s["heading_text"]
            parent_title = s.get("parent_title")
            if parent_title and parent_title != section_label:
                section_label = f"{parent_title} / {section_label}" if section_label else parent_title
            split_idx = s.get("split_index")
            split_tot = s.get("split_total")
            if split_idx and split_tot and split_tot > 1:
                section_label = f"{section_label} (split {split_idx}/{split_tot})"
            proto_chunks.append({
                "chunk_number": str(i + 1),
                "page_number": str(s["page_number"]),
                "section": section_label,
                "heading_text": s["heading_text"],
                "heading_level": s["level"],
                "heading_type": s.get("heading_type", "section"),
                "parent_title": parent_title or "",
                "parent_section_id": s.get("parent_section_id", ""),
                "split_index": int(split_idx) if split_idx else 0,
                "split_total": int(split_tot) if split_tot else 0,
                "universal_from_globals": base_universal,
                "chunk_text": chunk_text,
                "_line_span": (s["start"], s["end"]),
            })

        if not proto_chunks:
            logger.warning("[HEADING-CHUNK] no chunks produced (empty block?)")
            return []

        enriched = await self._enrich_chunks_metadata_async(proto_chunks, doc_summary)

        # Finalize each chunk: merge globals into universal (deduped); drop internal keys.
        for c in enriched:
            llm_universal = c.get("universal") or ""
            base_universal = c.pop("universal_from_globals", "")
            # Dedup by normalized token; preserve order (globals first, then LLM).
            seen_norm: set = set()
            merged_parts: List[str] = []
            for piece in (base_universal, llm_universal):
                for tok in re.split(r"\s*\|\s*", piece or ""):
                    t = tok.strip()
                    if not t:
                        continue
                    key = re.sub(r"\s+", " ", t.lower())
                    if key in seen_norm:
                        continue
                    seen_norm.add(key)
                    merged_parts.append(t)
            c["universal"] = " | ".join(merged_parts)
            c.pop("_line_span", None)
            c.pop("heading_text", None)
            c.pop("heading_level", None)
        return enriched

    async def _chunk_block_centralized(self, text: str, doc_summary: Optional[str], file_type: str = "pdf", _depth: int = 0) -> List[dict]:
        est_tokens = Utils.token_estimate(text)
        dynamic_timeout = self._compute_dynamic_timeout(text)
        prompt = PromptFactory.chunking(text, doc_summary, file_type, custom_instructions=self.custom_instructions)

        logger.info(f"[CHUNKING] Starting {file_type} block chunking - {len(text)} chars, ~{est_tokens} est. tokens, timeout={dynamic_timeout}s, depth={_depth}")
        logger.debug(f"[CHUNKING] Doc summary provided: {bool(doc_summary)} ({'yes' if doc_summary else 'no'})")
        logger.debug(f"[CHUNKING] Prompt length: {len(prompt)} chars")

        try:
            logger.info(f"[CHUNKING] Calling Gemini for {file_type} chunking (timeout={dynamic_timeout}s)...")
            raw_response, input_tokens, output_tokens = await self._track_and_call_gemini(
                prompt, f"chunk_block_{file_type}", timeout=dynamic_timeout
            )

            logger.info(f"[CHUNKING] Gemini response received - {len(raw_response)} chars, tokens in/out: {input_tokens}/{output_tokens}")
            logger.debug(f"[CHUNKING] Raw response preview (first 500 chars): {raw_response[:500]}")

            if not raw_response or not raw_response.strip():
                logger.error("[CHUNKING] Gemini returned empty response for chunking")
                raise ValueError("Empty response from Gemini")

            # Clean the response and log the cleaning process
            logger.debug("[CHUNKING] Cleaning JSON fences from response...")
            cleaned = Utils.clean_json_fence(raw_response)
            logger.debug(f"[CHUNKING] Cleaned response length: {len(cleaned)} chars")
            logger.debug(f"[CHUNKING] Cleaned response preview (first 500 chars): {cleaned[:500]}")

            # Check if cleaned response looks like valid JSON
            if not cleaned.strip().startswith(('[', '{')):
                logger.warning(f"[CHUNKING] Cleaned response doesn't start with [ or {{. Starts with: '{cleaned[:50]}'")
                logger.warning(f"[CHUNKING] Full cleaned response: {cleaned}")
                raise ValueError(f"Response doesn't appear to be valid JSON. Starts with: '{cleaned[:50]}'")

            # Try to parse JSON with detailed error logging
            logger.debug("[CHUNKING] Attempting to parse JSON...")
            try:
                chunks = json.loads(cleaned)
                logger.info(f"[CHUNKING] JSON parsing successful - parsed type: {type(chunks).__name__}")
            except json.JSONDecodeError as je:
                logger.error(f"[CHUNKING] JSON decode error at line {je.lineno}, column {je.colno}: {je.msg}")
                logger.error(f"[CHUNKING] Error position in response: char {je.pos}")

                # Log context around the error position
                if je.pos and je.pos < len(cleaned):
                    start = max(0, je.pos - 100)
                    end = min(len(cleaned), je.pos + 100)
                    context = cleaned[start:end]
                    logger.error(f"[CHUNKING] Context around error position: ...{context}...")

                # --- JSON REPAIR FALLBACK ---
                try:
                    chunks = Utils.try_repair_json(cleaned, context_label=f"chunk_block_{file_type}")
                    logger.info(f"[CHUNKING] JSON repair succeeded - parsed type: {type(chunks).__name__}")
                except ValueError:
                    # --- PARTIAL SALVAGE FALLBACK ---
                    salvaged = Utils.salvage_partial_json_array(cleaned, context_label=f"chunk_block_{file_type}")
                    if salvaged:
                        logger.warning(
                            f"[CHUNKING] Full repair failed but salvaged {len(salvaged)} chunks "
                            f"from truncated response (original error: {je.msg} at line {je.lineno}, col {je.colno})"
                        )
                        chunks = salvaged
                    else:
                        logger.error(f"[CHUNKING] Full response that failed to parse ({len(cleaned)} chars): {cleaned[:500]}...")
                        raise ValueError(f"Failed to parse JSON: {je.msg} at line {je.lineno}, col {je.colno}. Repair and salvage both failed.")

            # Validate the parsed structure
            if not isinstance(chunks, (list, dict)):
                logger.error(f"[CHUNKING] Unexpected response type: {type(chunks)}. Expected list or dict.")
                logger.error(f"[CHUNKING] Response content: {chunks}")
                raise ValueError(f"Expected list or dict, got {type(chunks)}")

            if isinstance(chunks, dict):
                logger.debug("[CHUNKING] Converting single chunk dict to list")
                chunks = [chunks]  # Single chunk case

            logger.info(f"[CHUNKING] Parsed {len(chunks)} chunks from response")

            # Ensure each chunk has required fields with detailed logging
            valid_chunks = []
            for i, chunk in enumerate(chunks):
                logger.debug(f"[CHUNKING] Validating chunk {i+1}/{len(chunks)}")

                if not isinstance(chunk, dict):
                    logger.warning(f"[CHUNKING] Chunk {i+1} is not a dict: {type(chunk)}, skipping")
                    continue

                # Log chunk keys for debugging
                logger.debug(f"[CHUNKING] Chunk {i+1} keys: {list(chunk.keys())}")

                # Ensure chunk_number is set
                if "chunk_number" not in chunk:
                    chunk["chunk_number"] = str(i + 1)
                    logger.debug(f"[CHUNKING] Set chunk_number for chunk {i+1}")

                # Set chunk_text from summary if not present
                if "chunk_text" not in chunk or not chunk["chunk_text"]:
                    chunk["chunk_text"] = chunk.get("summary", "")
                    logger.debug(f"[CHUNKING] Set chunk_text from summary for chunk {i+1}")

                # Extract universal and topic to prepend to chunk_text
                universal = chunk.get("universal", "").strip()
                topic = chunk.get("topic", "").strip()

                # Build prefix with pipe separation
                prefix_parts = []
                if universal:
                    prefix_parts.append(universal)
                if topic:
                    prefix_parts.append(topic)

                # Prepend to chunk_text if we have prefix parts
                if prefix_parts and chunk.get("chunk_text"):
                    prefix = " | ".join(prefix_parts)
                    original_chunk_text = chunk["chunk_text"]
                    chunk["chunk_text"] = f"{prefix} | {original_chunk_text}"
                    logger.debug(f"[CHUNKING] Chunk {i+1}: Prepended '{prefix}' to chunk_text")
                elif prefix_parts:
                    # If no original chunk_text but we have prefix parts, use just the prefix
                    prefix = " | ".join(prefix_parts)
                    chunk["chunk_text"] = prefix
                    logger.debug(f"[CHUNKING] Chunk {i+1}: Set chunk_text to prefix only: '{prefix}'")

                # Normalize and preserve analytical fields as top-level keys
                chunk["universal"] = chunk.get("universal", "").strip() if isinstance(chunk.get("universal"), str) else ""
                chunk["topic"] = chunk.get("topic", "").strip() if isinstance(chunk.get("topic"), str) else ""
                chunk["section"] = chunk.get("section", "").strip() if isinstance(chunk.get("section"), str) else ""
                # Handle "sub-topics" (hyphen) vs "sub_topics" (underscore) key mismatch
                sub_t = chunk.get("sub_topics", chunk.get("sub-topics", ""))
                chunk["sub_topics"] = sub_t.strip() if isinstance(sub_t, str) else ""
                chunk.pop("sub-topics", None)
                # customer_specific_tags: suppress entirely when no
                # customInstructions taxonomy provided (see _merge_chunk_metadata).
                if not self.custom_instructions:
                    chunk["customer_specific_tags"] = ""
                else:
                    cst = chunk.get("customer_specific_tags", "")
                    if isinstance(cst, list):
                        chunk["customer_specific_tags"] = json.dumps(cst)
                    elif isinstance(cst, str):
                        chunk["customer_specific_tags"] = cst.strip()
                    else:
                        chunk["customer_specific_tags"] = ""

                # Preserve section_reference from Gemini response
                chunk["section_reference"] = chunk.get("section_reference", "").strip() if isinstance(chunk.get("section_reference"), str) else ""

                # Set chunker_result
                chunk["chunker_result"] = "Success"
                valid_chunks.append(chunk)
                logger.debug(f"[CHUNKING] Chunk {i+1} validated successfully")

            logger.info(f"[CHUNKING] Successfully processed {file_type} block into {len(valid_chunks)} valid chunks (depth={_depth})")
            return valid_chunks

        except (RuntimeError, Exception) as e:
            is_timeout = "timed out" in str(e).lower() or "timeout" in str(e).lower()
            min_split_chars = 4000  # Don't try splitting blocks smaller than this
            max_recursion_depth = 3  # Safety limit on recursive splitting

            # --- Split-on-timeout fallback ---
            if is_timeout and len(text) > min_split_chars and _depth < max_recursion_depth:
                logger.warning(
                    f"[CHUNKING] Timeout at depth={_depth} for {file_type} block ({len(text)} chars, ~{est_tokens} est. tokens). "
                    f"Splitting block and retrying..."
                )
                sub_blocks = self._split_text_block(text, min_chars=min_split_chars)
                if len(sub_blocks) >= 2:
                    logger.info(f"[CHUNKING] Split into {len(sub_blocks)} sub-blocks, retrying each recursively (depth={_depth+1})")
                    all_chunks = []
                    for sb_idx, sb in enumerate(sub_blocks):
                        sb_est = Utils.token_estimate(sb)
                        logger.info(f"[CHUNKING] Sub-block {sb_idx+1}/{len(sub_blocks)}: {len(sb)} chars, ~{sb_est} est. tokens")
                        sub_result = await self._chunk_block_centralized(sb, doc_summary, file_type, _depth=_depth + 1)
                        all_chunks.extend(sub_result)
                    logger.info(f"[CHUNKING] Split-retry produced {len(all_chunks)} total chunks from {len(sub_blocks)} sub-blocks")
                    return all_chunks
                else:
                    logger.warning(f"[CHUNKING] Could not split block further (split returned {len(sub_blocks)} pieces), falling through to error")

            # --- Normal error handling ---
            error_msg = f"Chunk block processing failed for {file_type}: {str(e)}"
            logger.error(f"[CHUNKING] {error_msg}")
            logger.error(f"[CHUNKING] Exception type: {type(e).__name__}, timeout={is_timeout}, depth={_depth}, text_len={len(text)}")

            # Log the full traceback for debugging
            import traceback
            logger.error(f"[CHUNKING] Full traceback: {traceback.format_exc()}")

            return [self._create_chunk_error(
                chunk_number="1",
                file_type=file_type,
                stage="chunk processing",
                error_message=str(e)
            )]

    def _create_chunk_error(self,
                           chunk_number: str,
                           file_type: str,
                           stage: str,
                           error_message: str,
                           block_index: int = None,
                           start_row: int = None,
                           end_row: int = None,
                           sheet_name: str = None) -> dict:
        """Create standardized error chunk dictionary with consistent structure"""
        from datetime import datetime

        # Build context description
        context_parts = []
        if block_index is not None:
            context_parts.append(f"block {block_index}")
        if start_row is not None and end_row is not None:
            context_parts.append(f"rows {start_row}-{end_row}")
        if sheet_name:
            context_parts.append(f"sheet '{sheet_name}'")

        context = f" ({', '.join(context_parts)})" if context_parts else ""

        error_chunk = {
            "chunk_number": chunk_number,
            "summary": f"Processing failed: {error_message}",
            "chunk_text": f"Failed to process {file_type} {stage}{context}",
            "chunker_result": "Fail",
            "page_number": "",
            "debug_info": {
                "topic": f"Processing Error - {stage}{context}",
                "content": f"Failed to process {file_type} {stage}{context}: {error_message}",
                "chunker_result": "Fail",
                "file_type": file_type,
                "error_message": error_message,
                "processing_stage": stage,
                "timestamp": datetime.now().isoformat()
            }
        }

        # Add sheet_name for Excel if provided
        if sheet_name and file_type.lower() == "excel":
            error_chunk["sheet_name"] = sheet_name

        return error_chunk

    def _resolved_source_info(self, nexla_meta: dict, file_path: str) -> str:
        try:
            tags = (nexla_meta or {}).get("tags", {})
            if isinstance(tags, dict):
                display_path = tags.get("display_path", "")
                if display_path:
                    resolved = Path(display_path).name
                    logger.info(f"Resolved source_info from display_path: {resolved}")
                    return resolved
        except Exception:
            pass
        resolved = Path(file_path).name
        logger.info(f"Resolved source_info from file_path: {resolved}")
        return resolved

    def _create_error_output(self, file_path: str, error_message: str, file_type: str = "unknown", stage: str = "file_reading") -> Dict[str, pd.DataFrame]:
        """Create standardized error output for failed file processing"""
        file_title = Path(file_path).name
        output_filename = Path(file_path).stem + ".json"

        error_chunk = {
            "chunk_number": "0",
            "summary": f"Processing failed during {stage}: {error_message}",
            "chunk_text": f"Failed to process {file_type} file during {stage}",
            "chunker_result": "Fail",
            "source_info": file_path,
            "freshness_date": "Unknown",
            "freshness_date_unix": -1,
            "topic": "",
            "section": "",
            "universal": "",
            "sub_topics": "",
            "customer_specific_tags": "",
            "section_reference": "",
            "debug_info": {
                "chunker_result": "Fail",
                "file_type": file_type,
                "error_message": error_message,
                "processing_stage": stage,
                "file_name": file_title,
                "timestamp": datetime.now().isoformat()
            }
        }

        # Add file-type specific fields
        if file_type == "excel":
            error_chunk["sheet_name"] = "ERROR"
        elif file_type == "pdf":
            error_chunk["page_number"] = ""

        err_df = pd.DataFrame([error_chunk])
        return {output_filename: err_df}

    def _split_csv_into_blocks(self, df: pd.DataFrame, rows_per_block: Optional[int] = None) -> List[tuple]:
        """Split CSV DataFrame into row blocks for scalable processing"""
        if rows_per_block is None:
            rows_per_block = self.cfg.ROWS_PER_CSV_BLOCK

        blocks = []
        for i in range(0, len(df), rows_per_block):
            chunk_df = df.iloc[i:i + rows_per_block]
            text_block = chunk_df.to_string(index=False)
            start_row = i
            end_row = min(i + rows_per_block - 1, len(df) - 1)
            blocks.append((start_row, end_row, text_block))
        return blocks

    def _clean_chunk_text(self, text: str) -> str:
        """Remove page/sheet markers and clean chunk text"""
        if not text:
            return text

        # Remove page markers
        text = re.sub(r"page_marker:\s*######\[Page\s+\d+\]######\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"######\[Page\s+\d+\]######\s*", "", text, flags=re.IGNORECASE)

        # Remove sheet markers
        text = re.sub(r"sheet_marker:\s*######\[Sheet:\s*.+?\]######\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"######\[Sheet:\s*.+?\]######\s*", "", text, flags=re.IGNORECASE)

        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n', '\n', text)  # Multiple newlines to single
        text = text.strip()

        return text

    def _standardize_chunk_order(self, chunk: dict, file_type: str) -> dict:
        """Standardize the order of keys in chunk output"""
        ordered_chunk = {}

        # Core fields first
        if "chunk_number" in chunk:
            ordered_chunk["chunk_number"] = chunk["chunk_number"]
        if "summary" in chunk:
            ordered_chunk["summary"] = chunk["summary"]
        if "chunk_text" in chunk:
            ordered_chunk["chunk_text"] = self._clean_chunk_text(chunk["chunk_text"])

        # Location fields (file-type specific)
        if file_type == "excel" and "sheet_name" in chunk:
            ordered_chunk["sheet_name"] = chunk["sheet_name"]
        elif file_type == "pdf" and "page_number" in chunk:
            ordered_chunk["page_number"] = chunk["page_number"]

        # Metadata fields
        if "source_info" in chunk:
            ordered_chunk["source_info"] = chunk["source_info"]
        if "freshness_date" in chunk:
            ordered_chunk["freshness_date"] = chunk["freshness_date"]
        if "freshness_date_unix" in chunk:
            ordered_chunk["freshness_date_unix"] = chunk["freshness_date_unix"]

        for col in ["topic", "section", "universal", "sub_topics",
                    "customer_specific_tags", "section_reference"]:
            ordered_chunk[col] = chunk.get(col, "")

        # Debug info last
        if "debug_info" in chunk:
            ordered_chunk["debug_info"] = chunk["debug_info"]

        # Add any remaining fields not handled above
        for key, value in chunk.items():
            if key not in ordered_chunk:
                ordered_chunk[key] = value

        return ordered_chunk

    def _fit_docmeta_budget(self, text: str, file_title: str, llm: LLMService, max_tokens: Optional[int] = None, headroom: int = 2_000) -> str:
        """Trim document text to fit within token budget for docmeta extraction"""
        if max_tokens is None:
            max_tokens = self.cfg.MAX_TOKEN_LIMIT

        # Split by markers to drop tail pages if needed
        parts = re.split(r"(######\[Page \d+\]######\s*)", text)
        if len(parts) <= 1:  # no markers
            return text

        # Reassemble as pairs: [marker, content]
        pairs = ["".join(parts[i:i+2]) for i in range(1, len(parts), 2)]
        kept = pairs[:]
        while kept:
            prompt = PromptFactory.docmeta("".join(kept), file_title)
            if llm.count_tokens(prompt) <= max_tokens - headroom:
                break
            kept.pop()  # drop last page
        return "".join(kept) if kept else pairs[0]  # at least first page

    def _assemble_doc_text(self, file_results: List[dict]) -> str:
        blocks = []
        for p in file_results:
            dbg = p["debug_info"]
            if dbg.get("is_visual"):
                extracted_text = (dbg.get("extracted_text") or "").strip()
                enriched_text = (dbg.get("enriched_text") or "").strip()

                # Combine both extracted text and enriched visual descriptions
                if extracted_text and enriched_text:
                    text = f"{extracted_text}\n\n[VISUAL ANALYSIS]\n{enriched_text}"
                elif enriched_text:
                    text = enriched_text
                else:
                    text = extracted_text

                logger.debug(f"Page {p['page_number']} - Assembled text: {len(text)} chars (extracted: {len(extracted_text)}, enriched: {len(enriched_text)})")
            else:
                text = dbg.get("extracted_text") or ""

            if len(text.strip()) < 10:
                logger.warning(f"[PDF] Page {p['page_number']} has minimal content ({len(text.strip())} chars) - may be blank/image-only")

            blocks.append(f"######[Page {p['page_number']}]######\n{text}")
        return "\n\n".join(blocks)

    @staticmethod
    def _split_text_block(text: str, min_chars: int = 2000) -> List[str]:
        if len(text) <= min_chars:
            logger.debug(f"[SPLIT] Text too small to split ({len(text)} chars <= {min_chars}), returning as-is")
            return [text]

        # Strategy 1: Split on page markers
        page_pattern = r"(?=######\[Page \d+\]######)"
        segments = re.split(page_pattern, text)
        segments = [s.strip() for s in segments if s.strip()]
        if len(segments) >= 2:
            # Rebalance into two halves (split at the midpoint segment)
            mid = len(segments) // 2
            first_half = "\n\n".join(segments[:mid])
            second_half = "\n\n".join(segments[mid:])
            logger.info(f"[SPLIT] Split on page markers → 2 sub-blocks ({len(first_half)} + {len(second_half)} chars)")
            return [first_half, second_half]

        # Strategy 2: Split on double-newlines (paragraph boundaries)
        paragraphs = text.split("\n\n")
        if len(paragraphs) >= 2:
            mid = len(paragraphs) // 2
            first_half = "\n\n".join(paragraphs[:mid])
            second_half = "\n\n".join(paragraphs[mid:])
            if len(first_half) >= min_chars // 2 and len(second_half) >= min_chars // 2:
                logger.info(f"[SPLIT] Split on paragraphs → 2 sub-blocks ({len(first_half)} + {len(second_half)} chars)")
                return [first_half, second_half]

        # Strategy 3: Split at midpoint on nearest sentence boundary
        mid_pos = len(text) // 2
        # Search for '. ' near midpoint (within +/- 20% of text length)
        search_start = max(0, mid_pos - len(text) // 5)
        search_end = min(len(text), mid_pos + len(text) // 5)
        search_region = text[search_start:search_end]
        sentence_end = search_region.rfind(". ")
        if sentence_end != -1:
            split_pos = search_start + sentence_end + 2  # after the '. '
            first_half = text[:split_pos].strip()
            second_half = text[split_pos:].strip()
            if first_half and second_half:
                logger.info(f"[SPLIT] Split on sentence boundary → 2 sub-blocks ({len(first_half)} + {len(second_half)} chars)")
                return [first_half, second_half]

        # Strategy 4: Hard midpoint split (last resort)
        first_half = text[:mid_pos].strip()
        second_half = text[mid_pos:].strip()
        logger.warning(f"[SPLIT] Hard midpoint split (last resort) → 2 sub-blocks ({len(first_half)} + {len(second_half)} chars)")
        return [first_half, second_half]

    def _compute_doc_profile(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Doc-adaptive gating profile. Consumes already-assembled chunks
        (with content_type and heading_type populated) and returns a dict
        the caller uses to skip expensive optional passes when they wouldn't
        add value.

        Computed from what we already have in hand — no extra LLM call.
        """
        n = max(1, len(chunks))
        ct_counts: Dict[str, int] = {}
        ht_counts: Dict[str, int] = {}
        for c in chunks:
            ct = str(c.get("content_type") or "other")
            ht = str(c.get("heading_type") or "section")
            ct_counts[ct] = ct_counts.get(ct, 0) + 1
            ht_counts[ht] = ht_counts.get(ht, 0) + 1
        metrics_chunks = ct_counts.get("metrics", 0) + ct_counts.get("comparisons", 0)
        metrics_ratio = metrics_chunks / n
        bridge_ratio = ht_counts.get("bridge", 0) / n

        # Gate 1: structured-table extraction is worth the LLM cost when the
        # doc has >= 3 metrics/comparisons chunks. Ratio gate dropped because
        # it mis-fires on legal-heavy docs (e.g., a VTS lease with 15 metrics
        # chunks out of 200 legal chunks — small ratio, but those 15 are rent
        # schedules worth extracting).
        run_tables = metrics_chunks >= 3

        # Gate 2: verify-retry is worth it only when the doc has risk surface —
        # lots of page-continuation bridges (context can drift), OR metric-heavy
        # content (hallucinated numbers are dangerous), OR a large chunk count
        # (probability of at least one bad batch grows with N).
        run_verify = (
            bridge_ratio >= 0.30
            or metrics_ratio >= 0.15
            or n >= 100
        )

        profile = {
            "chunks": n,
            "metrics_chunks": metrics_chunks,
            "metrics_ratio": metrics_ratio,
            "bridge_ratio": bridge_ratio,
            "content_type_counts": ct_counts,
            "heading_type_counts": ht_counts,
            "run_tables": run_tables,
            "run_verify": run_verify,
        }
        logger.info(
            f"[PROFILE] chunks={n} metrics={metrics_chunks} "
            f"(ratio={metrics_ratio:.0%}) bridges={ht_counts.get('bridge', 0)} "
            f"(ratio={bridge_ratio:.0%}) → run_tables={run_tables} run_verify={run_verify}"
        )
        return profile

    def _compute_density_target(self, page_token_counts: List[int]) -> int:
        import math, statistics
        if not page_token_counts:
            logger.warning("[DENSITY] No page token counts provided, using default target of 15000")
            return 15_000

        sorted_counts = sorted(page_token_counts)
        n = len(sorted_counts)
        median_toks = int(statistics.median(sorted_counts))
        p90_idx = min(int(n * 0.9), n - 1)
        p90_toks = sorted_counts[p90_idx]
        max_toks = sorted_counts[-1]
        mean_toks = int(statistics.mean(sorted_counts))

        # ---- Continuous log-linear formula ----
        FLOOR_P90  =    500     # below this, all docs are treated as very sparse
        CEIL_P90   = 12_000     # above this, all docs are treated as very dense
        MAX_TARGET = 25_000     # block size for sparsest docs
        MIN_TARGET =  8_000     # block size for densest docs

        p90_clamped = max(FLOOR_P90, min(CEIL_P90, p90_toks))
        # log ratio: 0.0 (sparse) → 1.0 (dense)
        log_ratio = (math.log(p90_clamped) - math.log(FLOOR_P90)) / (math.log(CEIL_P90) - math.log(FLOOR_P90))
        target = int(MAX_TARGET - log_ratio * (MAX_TARGET - MIN_TARGET))

        # Estimate resulting block count so operator can judge cost/latency
        total_toks = sum(page_token_counts)
        est_blocks = max(1, total_toks // target)

        logger.info(
            f"[DENSITY] Page stats: pages={n}, mean={mean_toks}, median={median_toks}, "
            f"p90={p90_toks}, max={max_toks}"
        )
        logger.info(
            f"[DENSITY] Dynamic target: p90={p90_toks} → ratio={log_ratio:.3f} → "
            f"target_block_tokens={target}, est_blocks={est_blocks}"
        )
        return target

    def _group_blocks(self, full_text: str) -> List[str]:
        # Split into (marker, page_text) pairs
        pages = []
        pattern = r"(######\[Page (\d+)\]######\s*)(.*?)(?=(######\[Page \d+\]######|$))"
        for m in re.finditer(pattern, full_text, re.DOTALL):
            pages.append((m.group(1), m.group(3)))  # (marker, text)

        if not pages:  # fallback: whole doc as single block
            est = Utils.token_estimate(full_text)
            logger.warning(f"[BLOCK-SPLIT] No page markers found, returning full text as 1 block ({est} est. tokens)")
            return [full_text]

        logger.info(f"[BLOCK-SPLIT] Found {len(pages)} pages to pack into blocks")

        # Estimate tokens per page (fast, local)
        page_token_counts = [Utils.token_estimate(body) for _, body in pages]

        # Compute density-driven target
        target_block_tokens = self._compute_density_target(page_token_counts)

        # Pack pages into blocks using the density-driven target
        groups: List[str] = []
        buf: List[str] = []
        buf_tokens = 0

        for idx, (marker, body) in enumerate(pages):
            page_text = marker + body
            page_toks = page_token_counts[idx]

            # Would adding this page exceed the target?
            if buf and (buf_tokens + page_toks) > target_block_tokens:
                # Flush current buffer as a block
                groups.append("\n\n".join(buf))
                logger.debug(f"[BLOCK-SPLIT] Block {len(groups)}: {len(buf)} pages, ~{buf_tokens} est. tokens")
                buf = []
                buf_tokens = 0

            # If a single page exceeds the target, try intra-page splitting
            if not buf and page_toks > target_block_tokens:
                logger.warning(f"[BLOCK-SPLIT] Page {idx+1} is oversized ({page_toks} est. tokens > target {target_block_tokens}), splitting within page")
                sub_blocks = self._split_text_block(page_text)
                for sb in sub_blocks:
                    groups.append(sb)
                    sb_toks = Utils.token_estimate(sb)
                    logger.debug(f"[BLOCK-SPLIT] Block {len(groups)} (sub-page split): ~{sb_toks} est. tokens")
                continue

            buf.append(page_text)
            buf_tokens += page_toks

        # Flush remaining buffer
        if buf:
            groups.append("\n\n".join(buf))
            logger.debug(f"[BLOCK-SPLIT] Block {len(groups)}: {len(buf)} pages, ~{buf_tokens} est. tokens")

        total_est = sum(Utils.token_estimate(g) for g in groups)
        logger.info(f"[BLOCK-SPLIT] Produced {len(groups)} blocks from {len(pages)} pages (total ~{total_est} est. tokens, target/block={target_block_tokens})")
        return groups

    async def _process_pages_async(self, per_page_data, pdf_path, resolved_source):
        extract_sem = asyncio.Semaphore(self.cfg.SEMAPHORE_CHUNK_PROCESSING)
        batch_sem = asyncio.Semaphore(self.cfg.SEMAPHORE_CHUNK_PROCESSING)
        pdf_lock = asyncio.Lock()

        def _extract_text_and_png(pdf_path: str, page_idx_1: int):
            with pdfplumber.open(pdf_path) as pdf:
                page_obj = pdf.pages[page_idx_1 - 1]
                return PageProcessor.extract_text(page_obj), PageProcessor.to_png_b64(page_obj)

        def _extract_text_only(pdf_path: str, page_idx_1: int):
            with pdfplumber.open(pdf_path) as pdf:
                return PageProcessor.extract_text(pdf.pages[page_idx_1 - 1])

        # ------- Phase 1: text extraction for every page -------
        text_by_page: Dict[int, str] = {}
        png_by_page: Dict[int, Optional[str]] = {}

        async def _extract(page):
            async with extract_sem:
                page_num = page["page_number"]
                has_image = page.get("images", 0) >= 1
                need_png = page["likely_visual"] or has_image
                async with pdf_lock:
                    if need_png:
                        extracted, b64 = await asyncio.to_thread(_extract_text_and_png, pdf_path, page_num)
                    else:
                        extracted = await asyncio.to_thread(_extract_text_only, pdf_path, page_num)
                        b64 = None
                text_by_page[page_num] = extracted or ""
                png_by_page[page_num] = b64

        logger.info(f"[PAGES] Phase 1/3: extracting text from {len(per_page_data)} pages")
        await asyncio.gather(*[_extract(p) for p in per_page_data])

        # ------- Phase 2: gate — decide which pages actually need vision -------
        vision_queue: List[dict] = []  # page dicts that need a vision call
        gate_skipped = 0
        for page in per_page_data:
            page_num = page["page_number"]
            extracted = text_by_page.get(page_num, "")
            is_visual = page["likely_visual"]
            has_image = page.get("images", 0) >= 1
            text_len = len(extracted.strip())

            if text_len >= self.cfg.VISION_SKIP_TEXT_THRESHOLD:
                # Text is substantial — skip vision even if page was flagged visual.
                gate_skipped += 1
                continue
            if is_visual or (text_len < 10 and has_image):
                vision_queue.append(page)

        logger.info(
            f"[PAGES] Phase 2/3: gating — {gate_skipped} pages have sufficient text "
            f"(>= {self.cfg.VISION_SKIP_TEXT_THRESHOLD} chars), skipping vision. "
            f"{len(vision_queue)} pages need vision."
        )

        # ------- Phase 3: batched vision calls -------
        vision_result_by_page: Dict[int, dict] = {}  # page_num -> {enriched_text, classification}

        async def _single_page_vision(page: dict) -> None:
            """Per-page fallback: one image, one Gemini call. Used when a batch fails."""
            page_num = page["page_number"]
            b64 = png_by_page.get(page_num)
            if not b64:
                vision_result_by_page[page_num] = {"enriched_text": "", "classification": "Error"}
                return
            extracted = text_by_page.get(page_num, "")
            use_ocr_prompt = len(extracted.strip()) < 10
            vision_prompt = (PromptFactory.vision_ocr_prompt() if use_ocr_prompt
                             else PromptFactory.vision_enrich_prompt(extracted))
            call_type_label = (f"vision_ocr_page_{page_num}" if use_ocr_prompt
                               else f"vision_enrichment_page_{page_num}")
            try:
                raw, _, _ = await asyncio.wait_for(
                    self._track_and_call_gemini(
                        prompt=vision_prompt,
                        call_type=call_type_label,
                        timeout=self.cfg.GEMINI_TIMEOUT_VISION,
                        image_data=b64,
                    ),
                    timeout=self.cfg.GEMINI_TIMEOUT_VISION + 30,
                )
                result = PageProcessor.post_process_vision_response(raw)
                vision_result_by_page[page_num] = {
                    "enriched_text": result.get("enriched_text", ""),
                    "classification": result.get("classification", ""),
                }
            except Exception as e:
                logger.error(f"[PAGES]   Per-page fallback page {page_num} failed: {e}")
                vision_result_by_page[page_num] = {"enriched_text": f"ERROR: {e}", "classification": "Error"}

        async def _run_batch(batch: List[dict]) -> None:
            """Try batched call; on any failure, fall back to per-page calls for this batch."""
            async with batch_sem:
                page_nums = [p["page_number"] for p in batch]
                images = [png_by_page[pn] for pn in page_nums if png_by_page.get(pn)]
                if not images:
                    return
                prompt = PromptFactory.vision_batch_prompt(len(images))
                batch_timeout = self.cfg.GEMINI_TIMEOUT_VISION + len(images) * 30

                batch_failed = False
                failure_reason = ""
                try:
                    logger.info(f"[PAGES]   Vision batch pages={page_nums} ({len(images)} images), timeout={batch_timeout}s")
                    in_tok = self.llm.count_tokens(prompt) or 0
                    self._total_input_tokens += in_tok
                    raw = await asyncio.wait_for(
                        self.llm.agen_vision_batch(prompt=prompt, base64_pngs=images, timeout_s=batch_timeout),
                        timeout=batch_timeout + 30,
                    )
                    out_tok = self.llm.count_tokens(raw) or 0
                    self._total_output_tokens += out_tok
                    self._total_llm_calls += 1

                    if isinstance(raw, str) and raw.startswith("ERROR:"):
                        batch_failed = True
                        failure_reason = raw[:200]
                    else:
                        cleaned = Utils.clean_json_fence(raw or "")
                        if cleaned.startswith("json"):
                            cleaned = cleaned[4:].strip()
                        parsed = None
                        try:
                            parsed = json.loads(cleaned)
                        except json.JSONDecodeError:
                            start, end = cleaned.find("["), cleaned.rfind("]")
                            if 0 <= start < end:
                                try:
                                    parsed = json.loads(cleaned[start:end + 1])
                                except json.JSONDecodeError:
                                    parsed = None

                        if not isinstance(parsed, list):
                            batch_failed = True
                            failure_reason = f"non-list response; raw preview: {(raw or '')[:200]}"
                        else:
                            # Map results by 1-based index within the batch
                            for i, pn in enumerate(page_nums):
                                item = parsed[i] if i < len(parsed) else {}
                                if not isinstance(item, dict):
                                    item = {}
                                vision_result_by_page[pn] = {
                                    "enriched_text": str(item.get("enriched_text", "") or ""),
                                    "classification": str(item.get("classification", "") or ""),
                                }
                            logger.info(f"[PAGES]   Vision batch pages={page_nums} done; tokens in/out={in_tok}/{out_tok}")
                except Exception as e:
                    batch_failed = True
                    failure_reason = str(e)[:200]

                if batch_failed:
                    logger.warning(
                        f"[PAGES]   Vision batch pages={page_nums} FAILED ({failure_reason}); "
                        f"falling back to per-page vision for this batch"
                    )
                    await asyncio.gather(*[_single_page_vision(p) for p in batch])

        if vision_queue:
            batch_size = max(1, self.cfg.VISION_BATCH_SIZE)
            batches = [vision_queue[i:i + batch_size] for i in range(0, len(vision_queue), batch_size)]
            logger.info(f"[PAGES] Phase 3/3: {len(batches)} vision batches of up to {batch_size} pages each (with per-page fallback on batch failure)")
            await asyncio.gather(*[_run_batch(b) for b in batches])
        else:
            logger.info(f"[PAGES] Phase 3/3: no vision calls needed")

        # ------- Assemble per-page result dicts (shape unchanged for downstream) -------
        results = []
        for page in per_page_data:
            page_num = page["page_number"]
            extracted = text_by_page.get(page_num, "")
            is_visual = page["likely_visual"]
            visual_reasons = list(page.get("triggered_features", []))

            vres = vision_result_by_page.get(page_num)
            if vres is not None:
                enriched_text = vres["enriched_text"]
                classification = vres["classification"]
                is_visual_final = True
            else:
                enriched_text = ""
                classification = ""
                is_visual_final = False
                if is_visual and len(extracted.strip()) >= self.cfg.VISION_SKIP_TEXT_THRESHOLD:
                    visual_reasons = list(visual_reasons) + ["gate_skipped_text_sufficient"]

            chunker_result = "Fail" if classification == "Error" else "Success"
            visual_metrics = {
                "num_curves": page["curves"],
                "num_rectangles": page["rects"],
                "num_lines": page["lines"],
                "num_images": page["images"],
            }
            debug_info = {
                "is_visual": is_visual_final,
                "visual_metrics": visual_metrics,
                "visual_reasons": ",".join(visual_reasons) if visual_reasons else "",
                "enriched_text": enriched_text,
                "extracted_text": extracted,
                "was_gemini_necessary": classification,
                "chunker_result": chunker_result,
            }
            results.append({"source_info": resolved_source, "page_number": page_num, "debug_info": debug_info})

        v_total = len(vision_queue)
        v_ok = sum(1 for r in results if r["debug_info"].get("chunker_result") == "Success" and r["debug_info"].get("is_visual"))
        v_err = sum(1 for r in results if r["debug_info"].get("chunker_result") == "Fail" and r["debug_info"].get("is_visual"))
        logger.info(
            f"[PAGES] vision summary: total_pdf_pages={len(per_page_data)}, "
            f"text_gated_skip={gate_skipped}, vision_called={v_total} (batches={max(1, (v_total + self.cfg.VISION_BATCH_SIZE - 1)//self.cfg.VISION_BATCH_SIZE) if v_total else 0}), "
            f"vision_ok={v_ok}, vision_err={v_err}"
        )
        return results

    def process_file(self, file_path: str, nexla_meta: dict) -> Dict[str, pd.DataFrame]:
        """Process different file types through appropriate handlers"""
        if not isinstance(file_path, str):
            logger.error(f"process_file called with non-string file_path: {type(file_path)}")
            return {}

        # Extract file metadata from nexla_meta for logging
        tags = (nexla_meta or {}).get("tags", {})
        display_path = tags.get("display_path", "")
        source_key = str((nexla_meta or {}).get("sourceKey", ""))  # full S3 object key
        file_size_bytes = tags.get("file_size", "unknown")
        source_id = (nexla_meta or {}).get("resourceId", "unknown")
        source_type = (nexla_meta or {}).get("sourceType", "unknown")
        file_id = tags.get("file_id", "unknown")

        file_ext = Path(file_path).suffix.lower()
        # Fallback: if no extension found, check nexla_meta tags.display_path
        if not file_ext:
            if display_path:
                file_ext = Path(display_path).suffix.lower()
                logger.info(f"No extension in file_path, resolved from tags.display_path: {file_ext}")
            else:
                logger.warning(f"No extension in file_path and no display_path in tags")

        file_name = Path(display_path).name if display_path else Path(file_path).name
        logger.info(f"===== PROCESS_FILE START =====")
        logger.info(f"  File name     : {file_name}")
        logger.info(f"  File path     : {file_path}")
        logger.info(f"  Source key    : {source_key or 'N/A'}")
        logger.info(f"  Display path  : {display_path or 'N/A'}")
        logger.info(f"  Extension     : {file_ext or 'NONE'}")
        logger.info(f"  File size     : {file_size_bytes} bytes")
        logger.info(f"  Source type   : {source_type}")
        logger.info(f"  Source ID     : {source_id}")
        logger.info(f"  File ID       : {file_id}")
        logger.info(f"  nexla_meta keys: {list((nexla_meta or {}).keys())}")
        logger.info(f"  tags keys      : {list(tags.keys())}")
        logger.info(f"  ---- nexla_meta full dump ----")
        for k, v in (nexla_meta or {}).items():
            logger.info(f"    nexla_meta[{k}] = {v}")
        logger.info(f"  ---- tags full dump ----")
        for k, v in tags.items():
            logger.info(f"    tags[{k}] = {v}")

        handlers = {
            ".pdf": self._handle_pdf,
            ".csv": self._process_csv,
            ".xlsx": lambda path, meta: asyncio.run(self._process_excel(path, meta)),
            ".xls": lambda path, meta: asyncio.run(self._process_excel(path, meta)),

            # Image handlers
            ".png": lambda path, meta: asyncio.run(self._handle_image(path, meta)),
            ".jpeg": lambda path, meta: asyncio.run(self._handle_image(path, meta)),
            ".jpg": lambda path, meta: asyncio.run(self._handle_image(path, meta)),
            ".webp": lambda path, meta: asyncio.run(self._handle_image(path, meta)),

        }

        handler = handlers.get(file_ext)
        if not handler:
            logger.error(f"Unsupported file format: '{file_ext}' for file: {file_name}. Supported: {list(handlers.keys())}")
            return {}

        logger.info(f"Dispatching {file_name} to {file_ext} handler")
        import time as _time
        _start = _time.time()
        result = handler(file_path, nexla_meta)
        elapsed = _time.time() - _start

        total_rows = sum(df.shape[0] for df in result.values()) if result else 0
        output_files = list(result.keys()) if result else []
        logger.info(f"===== PROCESS_FILE COMPLETE =====")
        logger.info(f"  File name     : {file_name}")
        logger.info(f"  Handler       : {file_ext}")
        logger.info(f"  Elapsed       : {elapsed:.2f}s")
        logger.info(f"  Output files  : {output_files}")
        logger.info(f"  Total rows    : {total_rows}")
        return result

    def _handle_pdf(self, pdf_path: str, nexla_meta: dict) -> Dict[str, pd.DataFrame]:
        """Handle PDF file processing using the original PDF pipeline"""
        import time as _time
        _pdf_start = _time.time()
        # Reset token counters for this file
        self._reset_token_counters()

        file_title = pdf_path.split("/")[-1]
        resolved_source = self._resolved_source_info(nexla_meta, pdf_path)
        tags = (nexla_meta or {}).get("tags", {})

        logger.info(f"[PDF] ========== Starting PDF pipeline for: {file_title} ==========")
        logger.info(f"[PDF] Resolved source: {resolved_source}")
        logger.info(f"[PDF] Config: concurrency={self.cfg.SEMAPHORE_CHUNK_PROCESSING}, model={self.cfg.GEMINI_MODEL}")

        # 1) profile
        logger.info(f"[PDF] Step 1/6: Profiling {file_title} ...")
        _t = _time.time()
        per_page, _ = self.profiler.profile(pdf_path)
        visual_count = sum(1 for p in per_page if p.get("likely_visual"))
        logger.info(f"[PDF] Step 1/6: Profiling done - {len(per_page)} pages detected, {visual_count} visual ({_time.time()-_t:.2f}s)")

        # 2) per-page processing (async)
        logger.info(f"[PDF] Step 2/6: Page processing (text extraction + vision) for {len(per_page)} pages ...")
        _t = _time.time()
        async def run_page_processing():
            return await self._process_pages_async(per_page, pdf_path, resolved_source)

        file_results = asyncio.run(run_page_processing())
        page_df = pd.DataFrame(file_results)
        visual_pages = sum(1 for p in per_page if p.get("likely_visual"))
        logger.info(f"[PDF] Step 2/6: Page processing done - {len(file_results)} pages processed, {visual_pages} visual pages ({_time.time()-_t:.2f}s)")

        # Planner call removed: its output was observational only (logged,
        # never used for routing). Per-block density fallback inside the
        # heading-driven path already handles heading-poor sections.
        plan = None  # kept as a sentinel; downstream log-guard handles None

        # --- Diagnostic guard: detect if majority of pages produced no content ---
        empty_page_count = 0
        for r in file_results:
            dbg = r.get("debug_info", {})
            ext_text = (dbg.get("extracted_text") or "").strip()
            enr_text = (dbg.get("enriched_text") or "").strip()
            if len(ext_text) < 10 and len(enr_text) < 10:
                empty_page_count += 1
        if len(file_results) > 0 and empty_page_count / len(file_results) > 0.8:
            sample_pages = file_results[:3]
            sample_diag = [
                {
                    "page": r.get("page_number"),
                    "extracted_len": len((r.get("debug_info", {}).get("extracted_text") or "").strip()),
                    "enriched_len": len((r.get("debug_info", {}).get("enriched_text") or "").strip()),
                    "is_visual": r.get("debug_info", {}).get("is_visual"),
                }
                for r in sample_pages
            ]
            logger.critical(
                f"[PDF] CONTENT WARNING: {empty_page_count}/{len(file_results)} pages "
                f"({100*empty_page_count/len(file_results):.0f}%) have no meaningful content. "
                f"First 3 pages diagnostics: {sample_diag}"
            )

        # 3) assemble text
        logger.info(f"[PDF] Step 3/6: Assembling document text from {len(file_results)} pages ...")
        _t = _time.time()
        document_text = self._assemble_doc_text(file_results)
        doc_est_tokens = Utils.token_estimate(document_text)
        logger.info(f"[PDF] Step 3/6: Document text assembled - {len(document_text)} chars, ~{doc_est_tokens} est. tokens ({_time.time()-_t:.2f}s)")

        # 4) docmeta (async so we get a true timeout, with token budget)
        logger.info(f"[PDF] Step 4/6: Extracting document metadata for {file_title} ...")
        _t = _time.time()
        doc_for_meta = self._fit_docmeta_budget(document_text, file_title, self.llm)
        logger.info(f"[PDF] Step 4/6: Docmeta budget text: {len(doc_for_meta)} chars (from {len(document_text)} chars)")

        async def _run_docmeta():
            return await self._extract_docmeta_centralized(doc_for_meta, file_title, "pdf")

        try:
            meta, _, _ = asyncio.run(_run_docmeta())
            logger.info(f"[PDF] Step 4/6: Metadata extraction complete ({_time.time()-_t:.2f}s)")
            logger.info(f"[PDF]   doc_title   : {getattr(meta, 'document_title', 'N/A')}")
            logger.info(f"[PDF]   freshness   : {getattr(meta, 'freshness_date', 'N/A')}")

            # Fallback: if Gemini returned empty/unparseable freshness_date, try filename regex
            normalized_freshness = Utils.normalize_freshness_date(meta.freshness_date or "")
            if not normalized_freshness:
                logger.info(f"[PDF] Freshness date from docmeta is empty/unparseable, trying filename fallback for: {file_title}")
                filename_date = self._extract_date_from_filename_regex(file_title)
                if filename_date:
                    logger.info(f"[PDF] Extracted freshness date from filename: {filename_date}")
                    meta = DocMeta(
                        file_title=meta.file_title,
                        freshness_date=filename_date,
                        document_summary=meta.document_summary,
                        extras=meta.extras,
                    )
                else:
                    content_date = self._extract_date_from_content(document_text)
                    if content_date:
                        logger.info(f"[PDF] Extracted freshness date from content: {content_date}")
                        meta = DocMeta(
                            file_title=meta.file_title,
                            freshness_date=content_date,
                            document_summary=meta.document_summary,
                            extras=meta.extras,
                        )
                    else:
                        logger.warning(f"[PDF] No freshness date found in filename or content: {file_title}")
        except Exception as e:
            error_msg = f"Document metadata extraction failed: {str(e)}"
            logger.error(f"[PDF] Step 4/6: FAILED - {e}")
            return self._create_error_output(pdf_path, error_msg, "pdf", "metadata_extraction")

        # 5) chunk per block (async) - density-driven blocking
        logger.info(f"[PDF] Step 5/6: Grouping pages into density-driven blocks ...")
        _t = _time.time()
        blocks = self._group_blocks(document_text)
        block_details = [(i + 1, len(b), Utils.token_estimate(b)) for i, b in enumerate(blocks)]
        logger.info(f"[PDF] Step 5/6: Produced {len(blocks)} blocks for chunking:")
        for b_num, b_chars, b_toks in block_details:
            logger.info(f"[PDF]   Block {b_num}/{len(blocks)}: {b_chars} chars, ~{b_toks} est. tokens")

        use_heading = self.cfg.USE_HEADING_DRIVEN_CHUNKING
        chunk_strategy = "heading-driven" if use_heading else "density-generative"
        logger.info(f"[PDF] Step 5/6: Starting {chunk_strategy} chunking of {len(blocks)} blocks "
                    f"(concurrency={self.cfg.SEMAPHORE_CHUNK_PROCESSING})")

        async def run_chunks_heading_driven():
            sem = asyncio.Semaphore(self.cfg.SEMAPHORE_CHUNK_PROCESSING)

            # --- Phase A: detect headings per block in parallel ---
            async def _detect(b):
                async with sem:
                    return await self._detect_block_headings_async(b)

            logger.info(f"[PDF]   Phase A: detecting headings across {len(blocks)} blocks in parallel")
            detections = await asyncio.gather(*[_detect(b) for b in blocks])

            # --- Phase B: LLM-resolve universal headings across the doc ---
            logger.info(f"[PDF]   Phase B: resolving universal headings across document")
            all_headings_flat = [h for d in detections for h in d["headings"]]
            universal_headings = await self._resolve_universal_headings_async(
                all_headings_flat, source_info=resolved_source,
            )

            # --- Phase C: bridge classification at block boundaries ---
            logger.info(f"[PDF]   Phase C: classifying bridges at {max(0, len(detections) - 1)} block boundaries")
            bridge_decisions = 0
            for b_idx in range(len(detections) - 1):
                cur = detections[b_idx]
                nxt = detections[b_idx + 1]
                if not cur["headings"]:
                    continue
                last_heading = next(
                    (h for h in reversed(cur["headings"])
                     if h.get("heading_type") in ("section", "page", "global", "bridge")),
                    None,
                )
                if last_heading is None:
                    continue
                last_ln = last_heading["line_num"]
                last_text = {
                    ln + 1: cur["lines"][ln]
                    for ln in range(last_ln, min(len(cur["lines"]), last_ln + 40))
                }
                # Next block prelude = lines 0..(first heading line_num - 1)
                nxt_headings = nxt["headings"]
                nxt_first_ln = nxt_headings[0]["line_num"] if nxt_headings else len(nxt["lines"])
                prelude = {
                    ln + 1: nxt["lines"][ln]
                    for ln in range(0, min(nxt_first_ln, 80))
                }
                decision = await self._classify_bridge_async(
                    last_heading=last_heading,
                    last_heading_text=last_text,
                    next_prelude_text=prelude,
                    current_block_headings=cur["headings"],
                    universal_headings=universal_headings,
                )
                if not decision:
                    continue
                bridge_decisions += 1
                bridging = decision["bridging"]
                if bridging == "true":
                    bridge_heading_text = last_heading["heading_text"]
                elif bridging == "relevant":
                    bridge_heading_text = ", ".join(decision.get("most_relevant_headings") or []) \
                        or last_heading["heading_text"]
                else:
                    continue
                bridge_entry = {
                    "line_num": 0,
                    "heading_text": bridge_heading_text,
                    "level": last_heading.get("level", 1),
                    "heading_type": "bridge",
                    "content_type": last_heading.get("content_type", "other"),
                }
                nxt["headings"] = [bridge_entry] + nxt["headings"]
                logger.info(f"[PDF]   Phase C: bridge inserted at block {b_idx + 2} (bridging={bridging!r}, text={bridge_heading_text!r})")
            logger.info(f"[PDF]   Phase C: {bridge_decisions} bridge decision(s) applied")

            # --- Phase D: chunk each block with its enriched headings ---
            logger.info(f"[PDF]   Phase D: chunking {len(blocks)} blocks (with universal + bridge context)")

            async def _chunk(block_idx, b, pre):
                import time as _bt
                _block_start = _bt.time()
                async with sem:
                    result = await self._heading_driven_chunk_block_async(
                        b, meta.document_summary,
                        pre_detected=pre,
                        universal_headings=universal_headings,
                    )
                    _block_elapsed = _bt.time() - _block_start
                    error_count = sum(1 for c in result if c.get("chunker_result") == "Fail")
                    logger.info(f"[PDF]   Block {block_idx + 1}/{len(blocks)} done - {len(result)} chunks ({error_count} errors) in {_block_elapsed:.2f}s")
                    return result

            out = await asyncio.gather(*[_chunk(i, b, detections[i]) for i, b in enumerate(blocks)])
            return [c for sub in out for c in sub]

        async def run_chunks_density():
            """Legacy density-driven flow."""
            sem = asyncio.Semaphore(self.cfg.SEMAPHORE_CHUNK_PROCESSING)

            async def _task(block_idx, b):
                import time as _bt
                _block_start = _bt.time()
                b_est = Utils.token_estimate(b)
                async with sem:
                    logger.info(f"[PDF]   Block {block_idx + 1}/{len(blocks)} - {len(b)} chars, ~{b_est} est. tokens - sending to Gemini (density) ...")
                    result = await self._chunk_block_centralized(b, meta.document_summary, "pdf")
                    _block_elapsed = _bt.time() - _block_start
                    error_count = sum(1 for c in result if c.get("chunker_result") == "Fail")
                    logger.info(f"[PDF]   Block {block_idx + 1}/{len(blocks)} done - {len(result)} chunks ({error_count} errors) in {_block_elapsed:.2f}s")
                    return result

            out = await asyncio.gather(*[_task(i, b) for i, b in enumerate(blocks)])
            return [c for sub in out for c in sub]

        chunked_output = asyncio.run(run_chunks_heading_driven() if use_heading else run_chunks_density())
        chunking_elapsed = _time.time() - _t
        logger.info(f"[PDF] Step 5/6: Chunking complete - {len(chunked_output)} chunks from {len(blocks)} blocks ({chunking_elapsed:.2f}s)")

        if use_heading:
            chunked_output = self._resolve_cross_block_bridges(chunked_output)

        # Doc-adaptive gating for the two optional passes. Env vars
        # (CHUNKER_VERIFY_CHUNKS / CHUNKER_STRUCTURED_TABLES) act as override
        # gates — when explicitly set, honor them; otherwise let the profile
        # decide. The profile costs nothing (pure local computation).
        doc_profile = self._compute_doc_profile(chunked_output) if use_heading and chunked_output else {
            "run_tables": False, "run_verify": False,
        }
        _verify_env = os.getenv("CHUNKER_VERIFY_CHUNKS")
        _tables_env = os.getenv("CHUNKER_STRUCTURED_TABLES")
        # If env was set, use its bool; else use profile decision.
        should_verify = (
            _verify_env.lower() in ("1", "true", "yes") if _verify_env is not None
            else doc_profile.get("run_verify", False)
        )
        should_tables = (
            _tables_env.lower() in ("1", "true", "yes") if _tables_env is not None
            else doc_profile.get("run_tables", False)
        )

        if use_heading and should_verify and chunked_output:
            _vt = _time.time()
            verify_report = asyncio.run(
                self._verify_and_retry_chunks_async(chunked_output, meta.document_summary)
            )
            logger.info(
                f"[PDF] Step 5.5/6: Verify-retry loop — "
                f"sampled={verify_report.get('sampled', 0)}, "
                f"ok={verify_report.get('ok', 0)}, "
                f"minor={verify_report.get('minor', 0)}, "
                f"major={verify_report.get('major', 0)}, "
                f"initial_drift={verify_report.get('initial_drift', 0):.0%}, "
                f"retried={verify_report.get('retried', 0)}, "
                f"stop_reason={verify_report.get('stop_reason', '?')!r} "
                f"({_time.time() - _vt:.2f}s)"
            )

        if use_heading and should_tables and chunked_output:
            _tt = _time.time()
            asyncio.run(self._extract_structured_tables_async(chunked_output))
            logger.info(f"[PDF] Step 5.7/6: Structured-table extraction done ({_time.time() - _tt:.2f}s)")
        elif use_heading and chunked_output and not should_tables:
            logger.info(f"[PDF] Step 5.7/6: Structured-table extraction skipped by profile "
                        f"(metrics_chunks={doc_profile.get('metrics_chunks', 0)})")

        # Filter out error objects and convert them to valid chunks
        valid_chunks = []
        for chunk in chunked_output:
            if isinstance(chunk, dict) and "error" in chunk:
                logger.error(f"[PDF] Found error chunk: {chunk}")
                # Convert error to valid chunk format
                processed_error_chunk = {
                    "page_number": "",
                    "chunk_number": "0",
                    "chunk_text": f"ERROR: {chunk.get('error', 'Unknown error')}",
                    "universal": "",
                    "section": "",
                    "topic": "Error Processing",
                    "sub_topics": "",
                    "customer_specific_tags": [],
                    "section_reference": "",
                    "content": f"Error occurred during processing: {chunk.get('error', 'Unknown error')}",
                    "summary": f"Processing failed: {chunk.get('error', 'Unknown error')}",
                    "chunker_result": "Fail"
                }
                valid_chunks.append(processed_error_chunk)
            else:
                valid_chunks.append(chunk)

        if not valid_chunks:
            error_msg = "No valid chunks produced during processing"
            logger.error(f"[PDF] No valid chunks produced for {file_title}")
            return self._create_error_output(pdf_path, error_msg, "pdf", "chunk_validation")

        # --- Empty-content guard: catch hollow chunks that have structure but no real text ---
        meaningful_count = sum(
            1 for c in valid_chunks
            if len((c.get("chunk_text") or "").strip()) > 20
            or len((c.get("summary") or "").strip()) > 20
        )
        if meaningful_count == 0:
            error_msg = (
                f"All {len(valid_chunks)} chunks produced are empty/hollow "
                f"(no chunk_text or summary > 20 chars). "
                f"The PDF may be scanned/image-based and OCR failed."
            )
            logger.critical(f"[PDF] EMPTY CONTENT: {error_msg}")
            return self._create_error_output(pdf_path, error_msg, "pdf", "empty_content_detection")

        chunk_df = pd.DataFrame(valid_chunks)
        error_chunks = sum(1 for c in valid_chunks if c.get("chunker_result") == "Fail")
        success_chunks = len(valid_chunks) - error_chunks
        logger.info(f"[PDF] Step 5/6: Chunk validation - {len(valid_chunks)} total, {success_chunks} success, {error_chunks} errors")

        # Coverage quality metric
        total_input_chars = sum(len(b) for b in blocks)
        total_chunk_chars = sum(len(str(c.get("chunk_text", ""))) for c in valid_chunks)
        coverage_ratio = total_chunk_chars / total_input_chars if total_input_chars > 0 else 0.0
        logger.info(f"[PDF] Coverage ratio: {coverage_ratio:.2f} (chunk_chars={total_chunk_chars}, input_chars={total_input_chars})")
        if coverage_ratio < 0.3:
            logger.warning(f"[PDF] LOW COVERAGE WARNING: ratio {coverage_ratio:.2f} < 0.30 — content may have been dropped")

        # 6) merge
        logger.info(f"[PDF] Step 6/6: Merging {len(valid_chunks)} chunks with {len(page_df)} page records ...")
        _t = _time.time()
        final_df = self.merger.merge(chunk_df, page_df, resolved_source, meta)

        # Inject any extra docmeta fields (populated from customInstructions) into every row
        for key, value in meta.extras.items():
            final_df[key] = value if value is not None else ""

        # Compute page-level processing stats
        self._total_pages = len(file_results)
        self._pages_with_chunks = final_df["page_number"].replace("", pd.NA).dropna().nunique()
        self._pages_with_zero_chunks = self._total_pages - self._pages_with_chunks

        # Inject final token counts from centralized tracking
        final_df = self._inject_final_token_counts(final_df, file_type="pdf")

        output_filename = file_title.replace(".pdf", "") + ".json"
        total_elapsed = _time.time() - _pdf_start
        merge_elapsed = _time.time() - _t
        logger.info(f"[PDF] Step 6/6: Merge done ({merge_elapsed:.2f}s)")
        logger.info(f"[PDF] ========== Pipeline complete for {file_title} ==========")
        logger.info(f"[PDF]   Total time    : {total_elapsed:.2f}s")
        logger.info(f"[PDF]   Pages (total) : {self._total_pages}")
        logger.info(f"[PDF]   Pages (chunks): {self._pages_with_chunks}")
        logger.info(f"[PDF]   Pages (empty) : {self._pages_with_zero_chunks}")
        logger.info(f"[PDF]   Blocks        : {len(blocks)}")
        logger.info(f"[PDF]   Chunks        : {len(final_df)} ({success_chunks} success, {error_chunks} errors)")
        logger.info(f"[PDF]   Chunking time : {chunking_elapsed:.2f}s")
        logger.info(f"[PDF]   LLM calls     : {self._total_llm_calls}")
        logger.info(f"[PDF]   Tokens (in)   : {self._total_input_tokens}")
        logger.info(f"[PDF]   Tokens (out)  : {self._total_output_tokens}")
        logger.info(f"[PDF]   Output        : {output_filename}")

        return {output_filename: final_df}

    async def _process_excel(self, file_path: str, nexla_meta: dict) -> Dict[str, pd.DataFrame]:
        """Handle Excel file processing with multi-sheet async support"""
        import time as _time
        _excel_start = _time.time()
        # Reset token counters for this file
        self._reset_token_counters()

        file_title = Path(file_path).name
        resolved_source = self._resolved_source_info(nexla_meta, file_path)
        tags = (nexla_meta or {}).get("tags", {})

        logger.info(f"[EXCEL] === Starting Excel pipeline for: {file_title} ===")

        # Read all sheets
        try:
            sheets_dict = pd.read_excel(file_path, sheet_name=None)  # Returns dict of all sheets
        except Exception as e:
            error_msg = f"Failed to read Excel file: {str(e)}"
            logger.error(f"Failed to read Excel file {file_path}: {e}")
            return self._create_error_output(file_path, error_msg, "excel", "file_reading")

        logger.info(f"Loaded Excel file with {len(sheets_dict)} sheets: {list(sheets_dict.keys())}")

        # Basic cleaning: drop rows that are completely null in each sheet
        cleaned_sheets_dict = {}
        for sheet_name, df in sheets_dict.items():
            before_drop = df.shape[0]
            df_clean = df.dropna(how="all")
            after_drop = df_clean.shape[0]
            if after_drop < before_drop:
                logger.info(f"Sheet '{sheet_name}': Dropped {before_drop - after_drop} completely null rows (now {after_drop} rows)")
            cleaned_sheets_dict[sheet_name] = df_clean

        # Combine all sheets for metadata extraction
        full_text_parts = []
        for sheet_name, df in cleaned_sheets_dict.items():
            sheet_text = df.to_string(index=False, max_rows=500)  # Limit per sheet for token budget
            full_text_parts.append(f"######[Sheet: {sheet_name}]######\n{sheet_text}")

        full_text = "\n\n".join(full_text_parts)
        sheets_dict = cleaned_sheets_dict  # Use cleaned sheets for downstream processing
        # Extract document metadata
        logger.info(f"Step 1: Starting document metadata extraction for {file_title}")
        try:
            meta, in_tok, out_tok = await self._extract_docmeta_centralized(full_text[:5000], file_title, "excel")
            logger.info(f"Document metadata extraction complete - tokens used: {in_tok + out_tok}")
        except Exception as e:
            error_msg = f"Document metadata extraction failed: {str(e)}"
            logger.error(f"Document metadata extraction failed: {e}")
            return self._create_error_output(file_path, error_msg, "excel", "metadata_extraction")

        # Extract freshness date from filename
        logger.info("Step 1.5: Extracting freshness date from filename")
        freshness_date, freshness_unix = await self._extract_freshness_centralized(file_title)

        # Process each sheet asynchronously with header extraction and row-block chunking
        async def process_sheet(sheet_name: str, df: pd.DataFrame):
            logger.info(f"Processing Excel sheet: {sheet_name} ({df.shape[0]} rows, {df.shape[1]} cols)")

            # Step 1: Extract headers for this sheet
            try:
                sheet_sample = df.head(50).to_string(index=False)
                extracted_headers = await self._extract_headers_centralized(sheet_sample, f"Excel sheet '{sheet_name}'")
                if extracted_headers:
                    header_text = "\t".join(extracted_headers)
                    logger.info(f"Sheet '{sheet_name}': extracted {len(extracted_headers)} headers")
                else:
                    header_text = "\t".join(df.columns.tolist())
                    logger.warning(f"Sheet '{sheet_name}': header extraction failed, using pandas column names")
            except Exception as e:
                logger.error(f"Sheet '{sheet_name}': header extraction error: {e}, using pandas column names")
                header_text = "\t".join(df.columns.tolist())

            # Step 2: Split sheet into row blocks and prepend headers
            raw_blocks = self._split_csv_into_blocks(df)  # Reuse the same splitting logic
            header_prefixed_blocks = []

            for start_row, end_row, _ in raw_blocks:
                block_df = df.iloc[start_row:end_row+1]
                data_only = block_df.to_string(index=False, header=False)
                block_with_headers = header_text + "\n" + data_only
                header_prefixed_blocks.append((start_row, end_row, block_with_headers))

            logger.info(f"Sheet '{sheet_name}': split into {len(header_prefixed_blocks)} header-prefixed blocks")

            # Step 3: Process blocks for this sheet with bounded concurrency
            async def process_sheet_blocks():
                # Limit concurrent blocks per sheet to prevent API overload
                block_semaphore = asyncio.Semaphore(self.cfg.MAX_CONCURRENT_BLOCKS_PER_SHEET)

                async def process_block_with_semaphore(idx, start_row, end_row, block_text):
                    async with block_semaphore:
                        return await self._process_tabular_block_centralized(
                            block_index=idx-1,  # Adjust for 0-based indexing
                            start_row=start_row,
                            end_row=end_row,
                            table_text=block_text,
                            file_type="Excel",
                            sheet_name=sheet_name
                        )

                tasks = [
                    process_block_with_semaphore(idx, start_row, end_row, block_text)
                    for idx, (start_row, end_row, block_text) in enumerate(header_prefixed_blocks, 1)
                ]

                return await asyncio.gather(*tasks, return_exceptions=True)

            chunk_results = await process_sheet_blocks()

            # Process results and add sheet_name
            sheet_chunks = []
            for i, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    logger.error(f"Sheet '{sheet_name}' block {i+1} failed: {result}")
                    error_chunk = {
                        "chunk_number": str(i+1),
                        "sheet_name": sheet_name,
                        "summary": f"Processing failed for sheet {sheet_name} block {i+1}: {str(result)}",
                        "chunk_text": f"Processing failed for sheet {sheet_name} block {i+1}",
                        "chunker_result": "Fail",
                        "debug_info": {
                            "topic": f"Sheet {sheet_name} Block {i+1} - Processing Failed",
                            "content": f"Failed to process data block: {str(result)}",
                            "error": str(result)
                        }
                    }
                    sheet_chunks.append(error_chunk)
                elif isinstance(result, list):
                    # Handle multiple chunks from a single block
                    for chunk in result:
                        # Add sheet_name and remove page_number
                        chunk["sheet_name"] = sheet_name
                        chunk.pop("page_number", None)

                        # Remove topic and content fields entirely (no longer needed in output)
                        chunk.pop("topic", None)
                        chunk.pop("content", None)

                        sheet_chunks.append(chunk)
                else:
                    logger.error(f"Sheet '{sheet_name}' block {i+1}: unexpected result type {type(result)}")

            logger.info(f"Sheet '{sheet_name}' processed - {len(sheet_chunks)} chunks generated")
            return sheet_chunks

        # Process sheets with controlled concurrency to prevent API rate limits
        sheet_semaphore = asyncio.Semaphore(self.cfg.MAX_CONCURRENT_SHEETS)
        logger.info(f"Step 2: Starting async processing of {len(sheets_dict)} sheets (max {self.cfg.MAX_CONCURRENT_SHEETS} concurrent)")

        async def process_sheet_with_semaphore(sheet_name: str, df: pd.DataFrame):
            async with sheet_semaphore:
                return await process_sheet(sheet_name, df)

        tasks = [process_sheet_with_semaphore(sheet_name, df) for sheet_name, df in sheets_dict.items()]
        chunk_outputs = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten all chunks from all sheets and handle sheet-level exceptions
        all_chunks = []
        for i, sheet_result in enumerate(chunk_outputs):
            if isinstance(sheet_result, Exception):
                sheet_name = list(sheets_dict.keys())[i]
                logger.error(f"Sheet '{sheet_name}' processing failed completely: {sheet_result}")
                error_chunk = {
                    "chunk_number": "1",
                    "sheet_name": sheet_name,
                    "summary": f"Complete processing failure for sheet {sheet_name}: {str(sheet_result)}",
                    "chunk_text": f"Failed to process entire sheet {sheet_name}",
                    "chunker_result": "Fail",
                    "debug_info": {
                        "topic": f"Sheet {sheet_name} - Complete Processing Failure",
                        "content": f"Failed to process entire sheet: {str(sheet_result)}",
                        "error": str(sheet_result)
                    }
                }
                all_chunks.append(error_chunk)
            elif isinstance(sheet_result, list):
                all_chunks.extend(sheet_result)
            else:
                logger.error(f"Unexpected sheet result type: {type(sheet_result)}")

        logger.info(f"Excel processing complete - total chunks generated: {len(all_chunks)}")

        # Process and validate chunks
        if not all_chunks:
            error_msg = "No chunks produced during Excel processing"
            logger.error("No chunks produced")
            return self._create_error_output(file_path, error_msg, "excel", "chunk_validation")

        # Renumber chunks sequentially before creating DataFrame
        all_chunks = self._renumber_chunks_sequentially(all_chunks)

        # Create chunk DataFrame
        chunk_df = pd.DataFrame(all_chunks)
        logger.info(f"Excel chunks created: {len(all_chunks)}")

        # Add required metadata fields including freshness dates
        chunk_df["source_info"] = resolved_source
        chunk_df["freshness_date"] = freshness_date
        chunk_df["freshness_date_unix"] = freshness_unix

        # Set chunk_text as clean readable content (no labels per new PRD)
        def build_chunk_text(row):
            # Use chunk_text if provided by LLM, otherwise fall back to summary
            return row.get('chunk_text', row.get('summary', 'Data block processed'))

        chunk_df["chunk_text"] = chunk_df.apply(build_chunk_text, axis=1)

        # Ensure all required columns exist with proper defaults
        required_cols = ["chunk_number", "sheet_name", "summary", "chunk_text"]
        for col in required_cols:
            if col not in chunk_df.columns:
                chunk_df[col] = ""

        # Add debug info for consistency
        if "debug_info" not in chunk_df.columns:
            chunk_df["debug_info"] = [{"chunker_result": "Success", "file_type": "excel", "processing_method": "header_prefixed_blocks_per_sheet"}] * len(chunk_df)

        # Update debug_info with chunker_result from individual chunks
        error_count = 0
        for idx, row in chunk_df.iterrows():
            debug_info = row.get("debug_info", {})
            if isinstance(debug_info, dict):
                debug_info["chunker_result"] = row.get("chunker_result", "Success")
                debug_info["file_type"] = "excel"
                debug_info["processing_method"] = "header_prefixed_blocks_per_sheet"
                debug_info["total_chunks"] = len(all_chunks)
                debug_info["total_sheets"] = len(sheets_dict)
                # Preserve topic and content that were moved to debug_info
                chunk_df.at[idx, "debug_info"] = debug_info
                if row.get("chunker_result") == "Fail":
                    error_count += 1

        # Create final output
        output_filename = Path(file_path).stem + ".json"
        total_elapsed = _time.time() - _excel_start
        logger.info(f"[EXCEL] === Pipeline complete for {file_title} ===")
        logger.info(f"[EXCEL]   Total time  : {total_elapsed:.2f}s")
        logger.info(f"[EXCEL]   Sheets      : {len(sheets_dict)}")
        logger.info(f"[EXCEL]   Chunks      : {len(chunk_df)}")
        logger.info(f"[EXCEL]   Errors      : {error_count}")
        logger.info(f"[EXCEL]   Output      : {output_filename}")

        # Apply standardization and cleaning to each chunk
        standardized_chunks = []
        for _, row in chunk_df.iterrows():
            standardized_chunk = self._standardize_chunk_order(row.to_dict(), "excel")
            standardized_chunks.append(standardized_chunk)

        # Create new DataFrame with standardized chunks
        chunk_df = pd.DataFrame(standardized_chunks)

        # Inject any extra docmeta fields (from customInstructions) into every row
        for key, value in meta.extras.items():
            chunk_df[key] = value if value is not None else ""

        chunk_df = self._attach_csv_excel_phase4_fields(chunk_df, None, file_type="excel")

        # Final schema: fixed base columns in order, then append extras
        base_cols = [
            "chunk_number", "sheet_name", "summary", "chunk_text", "source_info",
            "freshness_date", "freshness_date_unix",
            "topic", "section", "universal", "sub_topics", "customer_specific_tags",
            "section_reference", "content_type", "potential_questions",
            "structured_tables",
            "debug_info",
        ]
        for col in base_cols:
            if col not in chunk_df.columns:
                chunk_df[col] = ""

        # Inject final token counts into debug_info for all chunks
        chunk_df = self._inject_final_token_counts(chunk_df, file_type="excel")

        present_base = [c for c in base_cols if c in chunk_df.columns]
        extras_cols = [c for c in chunk_df.columns if c not in base_cols]
        return {output_filename: chunk_df[present_base + extras_cols]}

    def _process_csv(self, file_path: str, nexla_meta: dict) -> Dict[str, pd.DataFrame]:
        """Handle CSV file processing with header extraction and row-block chunking"""
        import time as _time
        _csv_start = _time.time()
        # Reset token counters for this file
        self._reset_token_counters()

        file_title = Path(file_path).name
        resolved_source = self._resolved_source_info(nexla_meta, file_path)
        tags = (nexla_meta or {}).get("tags", {})

        logger.info(f"[CSV] === Starting CSV pipeline for: {file_title} ===")

        # Read the CSV data
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            error_msg = f"Failed to read CSV file: {str(e)}"
            logger.error(f"Failed to read CSV file {file_path}: {e}")
            return self._create_error_output(file_path, error_msg, "csv", "file_reading")

        logger.info(f"Loaded CSV data: {df.shape[0]} rows, {df.shape[1]} columns")

        # Basic cleaning: drop rows that are completely null
        before_drop = df.shape[0]
        df = df.dropna(how="all")
        after_drop = df.shape[0]
        if after_drop < before_drop:
            rows_dropped = before_drop - after_drop
            logger.info(f"Dropped {rows_dropped} completely null rows from CSV (now {after_drop} rows)")
            print(f"Null rows cleaned: Dropped {rows_dropped} completely empty rows from CSV")
        else:
            print("Null rows cleaned: No completely empty rows found")
        # Step 1: Extract headers from full table using Gemini
        full_table_text = df.head(50).to_string(index=False)  # Use first 50 rows for header extraction
        logger.info("Step 1: Extracting headers from CSV table")

        csv_docmeta_sample = df.head(200).to_string(index=False)[:5000]

        async def extract_headers_freshness_and_docmeta_async():
            headers_task = self._extract_headers_centralized(full_table_text, "CSV")
            freshness_task = self._extract_freshness_centralized(file_title)
            docmeta_task = self._extract_docmeta_centralized(csv_docmeta_sample, file_title, "csv")
            headers, (freshness_date, freshness_unix), (meta_obj, _in_tok, _out_tok) = await asyncio.gather(
                headers_task, freshness_task, docmeta_task
            )
            return headers, freshness_date, freshness_unix, meta_obj

        try:
            extracted_headers, freshness_date, freshness_unix, meta = asyncio.run(
                extract_headers_freshness_and_docmeta_async()
            )
            if extracted_headers:
                header_text = "\t".join(extracted_headers)
                logger.info(f"Successfully extracted {len(extracted_headers)} headers")
                logger.info(f"Headers: {extracted_headers}")
            else:
                # Fallback to pandas column names
                header_text = "\t".join(df.columns.tolist())
                logger.warning("Header extraction failed, using pandas column names")
        except Exception as e:
            logger.error(f"Header/freshness/docmeta extraction failed: {e}, using fallbacks")
            header_text = "\t".join(df.columns.tolist())
            freshness_date, freshness_unix = "Unknown", -1
            meta = DocMeta(file_title=file_title, freshness_date="", document_summary="")

        # Step 2: Split CSV into row blocks and prepend headers
        raw_blocks = self._split_csv_into_blocks(df)
        logger.info(f"Step 2: Split CSV into {len(raw_blocks)} blocks of ~{self.cfg.ROWS_PER_CSV_BLOCK} rows each")

        # Create header-prefixed blocks
        header_prefixed_blocks = []
        for start_row, end_row, block_text in raw_blocks:
            # Remove default pandas headers and index, keep only data rows
            block_df = df.iloc[start_row:end_row+1]
            data_only = block_df.to_string(index=False, header=False)
            # Prepend extracted headers
            block_with_headers = header_text + "\n" + data_only
            header_prefixed_blocks.append((start_row, end_row, block_with_headers))

        # Step 3: Process blocks asynchronously with header context
        logger.info(f"Step 3: Starting async chunking of {len(header_prefixed_blocks)} header-prefixed blocks")

        async def run_csv_chunking():
            # Control concurrency to prevent API rate limits
            block_semaphore = asyncio.Semaphore(self.cfg.SEMAPHORE_CHUNK_PROCESSING)

            async def process_block_with_semaphore(idx, start_row, end_row, block_text):
                async with block_semaphore:
                    return await self._process_tabular_block_centralized(
                        block_index=idx-1,  # Adjust for 0-based indexing
                        start_row=start_row,
                        end_row=end_row,
                        table_text=block_text,
                        file_type="CSV",
                        sheet_name=None
                    )

            tasks = [
                process_block_with_semaphore(idx, start_row, end_row, block_text)
                for idx, (start_row, end_row, block_text) in enumerate(header_prefixed_blocks, 1)
            ]

            return await asyncio.gather(*tasks, return_exceptions=True)

        try:
            chunk_outputs = asyncio.run(run_csv_chunking())
            logger.info(f"Async chunking complete - processed {len(chunk_outputs)} blocks")
        except Exception as e:
            error_msg = f"Async chunking process failed: {str(e)}"
            logger.error(f"Async chunking failed: {e}")
            return self._create_error_output(file_path, error_msg, "csv", "chunking")

        # Step 4: Process results and handle errors
        valid_chunks = []
        error_count = 0

        for i, result in enumerate(chunk_outputs):
            if isinstance(result, Exception):
                logger.error(f"Block {i+1} processing failed with exception: {result}")
                error_chunk = {
                    "chunk_number": str(i+1),
                    "summary": f"Processing failed for data block {i+1}: {str(result)}",
                    "chunk_text": f"Processing failed for data block {i+1}",
                    "chunker_result": "Fail",
                    "debug_info": {
                        "topic": f"Data Block {i+1} - Processing Failed",
                        "content": f"Failed to process data block: {str(result)}",
                        "error": str(result)
                    }
                }
                valid_chunks.append(error_chunk)
                error_count += 1
            elif isinstance(result, list):
                # Handle multiple chunks from a single block
                for chunk in result:
                    # Clean up chunk fields - remove page/sheet fields for CSV
                    chunk.pop("page_number", None)
                    chunk.pop("sheet_name", None)

                    # Remove topic and content fields entirely (no longer needed in output)
                    chunk.pop("topic", None)
                    chunk.pop("content", None)

                    valid_chunks.append(chunk)
                    if chunk.get("chunker_result") == "Fail":
                        error_count += 1
            else:
                logger.error("Unexpected result type for block %d: %s", i+1, type(result))
                error_count += 1

        if not valid_chunks:
            error_msg = "No valid chunks produced during CSV processing"
            logger.error("No valid chunks produced")
            return self._create_error_output(file_path, error_msg, "csv", "chunk_validation")

        logger.info(f"CSV chunking results: {len(valid_chunks)} total chunks, {error_count} errors")

        # Renumber chunks sequentially before creating DataFrame
        valid_chunks = self._renumber_chunks_sequentially(valid_chunks)

        # Create chunk DataFrame
        chunk_df = pd.DataFrame(valid_chunks)

        # Add required metadata fields including freshness dates
        chunk_df["source_info"] = resolved_source
        chunk_df["freshness_date"] = freshness_date
        chunk_df["freshness_date_unix"] = freshness_unix

        # Set chunk_text as clean readable content (no labels per new PRD)
        def build_chunk_text(row):
            # Use chunk_text if provided by LLM, otherwise fall back to summary
            return row.get('chunk_text', row.get('summary', 'Data block processed'))

        chunk_df["chunk_text"] = chunk_df.apply(build_chunk_text, axis=1)

        # Ensure all required columns exist with proper defaults
        required_cols = ["chunk_number", "summary", "chunk_text"]
        for col in required_cols:
            if col not in chunk_df.columns:
                chunk_df[col] = ""

        # Add debug info for consistency
        if "debug_info" not in chunk_df.columns:
            chunk_df["debug_info"] = [{"chunker_result": "Success", "file_type": "csv", "processing_method": "header_prefixed_blocks"}] * len(chunk_df)

        # Update debug_info with chunker_result from individual chunks
        for idx, row in chunk_df.iterrows():
            debug_info = row.get("debug_info", {})
            if isinstance(debug_info, dict):
                debug_info["chunker_result"] = row.get("chunker_result", "Success")
                debug_info["file_type"] = "csv"
                debug_info["processing_method"] = "header_prefixed_blocks"
                debug_info["total_chunks"] = len(valid_chunks)
                debug_info["error_count"] = error_count
                # Preserve topic and content that were moved to debug_info
                chunk_df.at[idx, "debug_info"] = debug_info

        # Create final output
        output_filename = Path(file_path).stem + ".json"
        total_elapsed = _time.time() - _csv_start
        logger.info(f"[CSV] === Pipeline complete for {file_title} ===")
        logger.info(f"[CSV]   Total time  : {total_elapsed:.2f}s")
        logger.info(f"[CSV]   Rows        : {df.shape[0]}")
        logger.info(f"[CSV]   Blocks      : {len(header_prefixed_blocks)}")
        logger.info(f"[CSV]   Chunks      : {len(chunk_df)}")
        logger.info(f"[CSV]   Errors      : {error_count}")
        logger.info(f"[CSV]   Output      : {output_filename}")

        # Apply standardization and cleaning to each chunk
        standardized_chunks = []
        for _, row in chunk_df.iterrows():
            standardized_chunk = self._standardize_chunk_order(row.to_dict(), "csv")
            standardized_chunks.append(standardized_chunk)

        # Create new DataFrame with standardized chunks
        chunk_df = pd.DataFrame(standardized_chunks)

        # Inject any extra docmeta fields (from customInstructions) into every row
        for key, value in meta.extras.items():
            chunk_df[key] = value if value is not None else ""

        chunk_df = self._attach_csv_excel_phase4_fields(chunk_df, df, file_type="csv")

        base_cols = [
            "chunk_number", "summary", "chunk_text", "source_info",
            "freshness_date", "freshness_date_unix",
            "topic", "section", "universal", "sub_topics", "customer_specific_tags",
            "section_reference", "content_type", "potential_questions",
            "structured_tables",
            "debug_info",
        ]
        for col in base_cols:
            if col not in chunk_df.columns:
                chunk_df[col] = ""

        chunk_df = self._inject_final_token_counts(chunk_df, file_type="csv")

        present_base = [c for c in base_cols if c in chunk_df.columns]
        extras_cols = [c for c in chunk_df.columns if c not in base_cols]
        return {output_filename: chunk_df[present_base + extras_cols]}

    async def _handle_image(self, file_path: str, nexla_meta: dict) -> Dict[str, pd.DataFrame]:
        """Handles image files by sending them to the vision model for analysis."""
        import time as _time
        _img_start = _time.time()
        self._reset_token_counters()
        file_title = Path(file_path).name
        output_filename = Path(file_path).stem + ".json"
        resolved_source = self._resolved_source_info(nexla_meta, file_path)
        tags = (nexla_meta or {}).get("tags", {})

        # --- 1. File Size Validation ---
        MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
        file_size = Path(file_path).stat().st_size

        logger.info(f"[IMAGE] === Starting image pipeline for: {file_title} ===")
        logger.info(f"[IMAGE]   File size   : {file_size / 1e6:.2f} MB")

        if file_size >= MAX_IMAGE_SIZE_BYTES:
            logger.warning(
                f"[IMAGE] Skipping image '{file_title}' ({file_size / 1e6:.2f} MB) "
                f"- exceeds 20 MB size limit."
            )
            return {}

        logger.info(f"[IMAGE] Step 1: Validating and reading image {file_title}")

        # --- 2. Read Image and Determine MIME Type ---
        mime_types = {
            ".png": "image/png",
            ".jpeg": "image/jpeg",
            ".jpg": "image/jpeg",
            ".webp": "image/webp"
        }
        file_ext = Path(file_path).suffix.lower()
        mime_type = mime_types.get(file_ext)

        if not mime_type:
            logger.error(f"Unsupported image extension: {file_ext}")
            return {}

        # Read image bytes and base64 encode
        with open(file_path, "rb") as image_file:
            image_bytes = image_file.read()
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # --- 3. Call Gemini Vision API ---
        logger.info(f"[IMAGE] Step 2: Calling Gemini Vision API for {file_title} (mime={mime_type}, b64_len={len(b64_image)})")
        prompt = PromptFactory.image_analysis_prompt()

        try:
            raw_response, _, _ = await self._track_and_call_gemini(
                prompt,
                call_type=f"image_analysis_{file_ext[1:]}",
                timeout=self.cfg.GEMINI_TIMEOUT_VISION,
                image_data=b64_image
            )
            logger.info(f"[IMAGE] Step 2: Vision API response received - {len(raw_response)} chars")
        except Exception as e:
            error_msg = f"Gemini Vision API failed for image '{file_title}': {e}"
            logger.error(f"[IMAGE] Step 2: FAILED - {error_msg}")
            return self._create_error_output(file_path, error_msg, "image", "api_call")

        # --- 4. Parse JSON Response ---
        try:
            logger.debug(f"Image analysis raw response: {raw_response[:200]}...")
            cleaned_response = Utils.clean_json_fence(raw_response)
            logger.debug(f"Image analysis cleaned response: {cleaned_response[:200]}...")

            parsed_response = json.loads(cleaned_response)

            if not isinstance(parsed_response, dict):
                raise ValueError(f"Expected JSON object, got {type(parsed_response)}")

            summary = parsed_response.get("summary", "Image analysis completed")
            chunk_text = parsed_response.get("chunk_text", raw_response)

            logger.debug(f"Extracted summary: {summary[:100]}...")
            logger.debug(f"Extracted chunk_text: {chunk_text[:100]}...")

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse JSON response for image '{file_title}': {e}")
            logger.warning("Using raw response as fallback")
            summary = f"Image analysis of {file_title}"
            chunk_text = raw_response

        # --- 5. Format Output ---
        freshness_date, freshness_unix = await self._extract_freshness_centralized(file_title)

        chunk = {
            "chunk_number": "1",
            "page_number": None, # Not applicable for images
            "sheet_name": None, # Not applicable for images
            "summary": summary,
            "chunk_text": chunk_text,
            "source_info": resolved_source,
            "freshness_date": freshness_date,
            "freshness_date_unix": freshness_unix,
            "topic": "",
            "section": "",
            "universal": "",
            "sub_topics": "",
            "customer_specific_tags": "",
            "section_reference": "",
            "debug_info": {
                "chunker_result": "Success",
                "processing_method": "image_analysis",
                "original_mime_type": mime_type
            }
        }

        chunk_df = pd.DataFrame([chunk])
        final_df = self._inject_final_token_counts(chunk_df, file_type="image")

        total_elapsed = _time.time() - _img_start
        logger.info(f"[IMAGE] === Pipeline complete for {file_title} ===")
        logger.info(f"[IMAGE]   Total time  : {total_elapsed:.2f}s")
        logger.info(f"[IMAGE]   File size   : {file_size / 1e6:.2f} MB")
        logger.info(f"[IMAGE]   MIME type   : {mime_type}")
        logger.info(f"[IMAGE]   Output      : {output_filename}")
        return {output_filename: final_df}

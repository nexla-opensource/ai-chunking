# Extracted verbatim from chunking_code.py — see PROVENANCE.md
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Tuple
from .chunker_utils import Utils
from .models import DocMeta

logger = logging.getLogger("ai_chunking.chunkers.document_chunker")


# ==== Merger ====
class Merger:
    def merge(
        self,
        chunks_df: pd.DataFrame,
        page_df: pd.DataFrame,
        resolved_source_info: str,
        docmeta: DocMeta,
    ):
        # backfill page_number from chunk_text if missing
        if "page_number" not in chunks_df.columns:
            chunks_df["page_number"] = None
        miss = chunks_df["page_number"].isna() | (
            chunks_df["page_number"].astype(str).str.strip() == ""
        )
        chunks_df.loc[miss, "page_number"] = chunks_df.loc[miss, "chunk_text"].apply(
            Utils.infer_page_number_from_text
        )

        # backfill chunk_number
        if "chunk_number" not in chunks_df.columns:
            chunks_df["chunk_number"] = None

        def _fill_chunk_numbers(df):
            df = df.copy()

            def _assign(g):
                need = g["chunk_number"].isna() | (
                    g["chunk_number"].astype(str).str.strip() == ""
                )
                g.loc[need, "chunk_number"] = (np.arange(need.sum()) + 1).astype(str)
                return g

            return df.groupby(
                df["page_number"].replace("", "__UNMAPPED__").fillna("__UNMAPPED__"), group_keys=False
            ).apply(_assign)

        chunks_df = _fill_chunk_numbers(chunks_df)

        pn_series = chunks_df["page_number"]

        def _is_numeric_str(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return False
            s = str(x).strip()
            return s.isdigit() and int(s) >= 1

        mapped_mask = pn_series.apply(_is_numeric_str)

        # Now cast to str for the merge key
        page_df = page_df.copy()
        page_df["source_info"] = resolved_source_info

        def _normalize_pn_to_str(s):
            if s is None or (isinstance(s, float) and pd.isna(s)) or str(s).strip().lower() == "none":
                return ""
            s = str(s).strip()
            return str(int(s)) if s.isdigit() else s

        page_df["page_number"] = page_df["page_number"].apply(_normalize_pn_to_str)
        chunks_df["page_number"] = chunks_df["page_number"].apply(_normalize_pn_to_str)

        mapped = chunks_df[mapped_mask].copy()
        unmapped = chunks_df[~mapped_mask].copy()

        # merge mapped
        mapped_final = pd.merge(mapped, page_df, on="page_number", how="left")

        # attach meta for unmapped
        freshness = Utils.normalize_freshness_date(docmeta.freshness_date or "")
        f_unix = Utils.date_to_unix(freshness)
        unmapped["source_info"] = resolved_source_info

        unmapped["freshness_date"] = freshness
        unmapped["freshness_date_unix"] = f_unix
        if "debug_info" not in unmapped.columns:
            unmapped["debug_info"] = [dict() for _ in range(len(unmapped))]

        # combine
        final_df = pd.concat([mapped_final, unmapped], ignore_index=True)

        # ensure source_info never NaN
        final_df["source_info"] = final_df["source_info"].fillna(resolved_source_info)

        # patch debug_info
        def _patch(row):
            d = row.get("debug_info", {})
            d = d if isinstance(d, dict) else {}
            d["chunker_result"] = row.get("chunker_result", d.get("chunker_result", "Success"))
            # Preserve visual fields with proper defaults so they always appear in output
            defaults = {
                "is_visual": False,
                "visual_metrics": {},
                "visual_reasons": "",
                "was_gemini_necessary": "",
            }
            keep = [
                "is_visual",
                "visual_metrics",
                "visual_reasons",
                "was_gemini_necessary",
                "chunker_result",
            ]
            return {k: d.get(k, defaults.get(k)) for k in keep}

        final_df["debug_info"] = final_df.apply(_patch, axis=1)

        # set fund/date
        freshness = Utils.normalize_freshness_date(docmeta.freshness_date or "")
        f_unix = Utils.date_to_unix(freshness)

        final_df["freshness_date"] = freshness
        final_df["freshness_date_unix"] = f_unix

        # stable numeric sort by (page, chunk)
        def _to_int(x):
            try:
                return int(str(x).strip())
            except (ValueError, TypeError):
                return 10**9
        final_df["_p"] = final_df["page_number"].map(_to_int)
        final_df["_c"] = final_df["chunk_number"].map(_to_int)
        final_df = final_df.sort_values(["_p","_c"]).drop(columns=["_p","_c"])

        # Ensure generic chunk-level columns exist with defaults.
        # document_summary is NOT a chunk-level column — it's doc-level,
        # used internally as context for chunking prompts but not propagated
        # into every row (was ~800-1500 tokens × N chunks of duplicated data).
        for col in ["topic", "content", "summary", "section", "universal", "sub_topics",
                     "customer_specific_tags", "section_reference"]:
            if col not in final_df.columns:
                final_df[col] = ""

        # Merge summary logging
        mapped_rows = mapped_final.shape[0] if not mapped_final.empty else 0
        unmapped_rows = unmapped.shape[0] if not unmapped.empty else 0
        error_rows = int(((chunks_df.get("chunker_result") == "Fail") if "chunker_result" in chunks_df else pd.Series(dtype=bool)).sum())
        logger.info(f"merge summary: mapped={mapped_rows} unmapped={unmapped_rows} errors={error_rows}")

        def _build_contextualized(row) -> str:
            parts: List[str] = []
            universal = str(row.get("universal") or "").strip()
            section = str(row.get("section") or "").strip()
            topic = str(row.get("topic") or "").strip()
            section_reference = str(row.get("section_reference") or "").strip()
            if universal:
                parts.append(f"[Document context: {universal}]")
            if section and section != topic:
                parts.append(f"[Section: {section}]")
            if topic:
                parts.append(f"[Topic: {topic}]")
            if section_reference:
                parts.append(f"[Reference: {section_reference}]")
            try:
                split_idx = int(row.get("split_index") or 0)
                split_tot = int(row.get("split_total") or 0)
            except (TypeError, ValueError):
                split_idx, split_tot = 0, 0
            if split_idx and split_tot and split_tot > 1:
                parts.append(f"[Part {split_idx} of {split_tot} in parent section]")
            prefix = " ".join(parts)
            body = str(row.get("chunk_text") or "")
            return f"{prefix}\n\n{body}" if prefix else body

        final_df["contextualized_chunk_text"] = final_df.apply(_build_contextualized, axis=1)

        base_cols = [
            "page_number", "chunk_number", "chunk_text", "contextualized_chunk_text",
            "source_info", "freshness_date", "freshness_date_unix", "summary",
            "topic", "section", "universal", "sub_topics", "customer_specific_tags",
            "section_reference",
            # Stitching metadata for retrieval-time sibling reassembly.
            "parent_section_id", "split_index", "split_total",
            "debug_info",
        ]
        internal_cols = {"chunker_result", "content", "debug_info_x", "debug_info_y"}
        present_base = [c for c in base_cols if c in final_df.columns]
        extras = [c for c in final_df.columns if c not in base_cols and c not in internal_cols]
        return final_df[present_base + extras]

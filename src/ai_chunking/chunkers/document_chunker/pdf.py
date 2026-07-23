# Extracted verbatim from chunking_code.py — see PROVENANCE.md
import base64
from io import BytesIO
import numpy as np
import pdfplumber
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ai_chunking.chunkers.document_chunker")


# ==== PDF Profiler ====
class PDFProfiler:
    def profile(self, pdf_path: str):
        stats = {
            "curves": [],
            "rects": [],
            "lines": [],
            "images": [],
            "total_features": [],
        }
        per_page = []
        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"PDF opened, {len(pdf.pages)} pages.")
            for i, page in enumerate(pdf.pages):
                try:
                    cur, rec, lin, img = (
                        len(page.curves), len(page.rects), len(page.lines), len(page.images)
                    )
                    single_corner_image = False
                    if img == 1 and cur == 0 and rec == 0:
                        image = page.images[0]
                        page_height = float(page.height)
                        page_width = float(page.width)
                        corner_threshold = 0.1
                        x0, y0 = float(image["x0"]), float(image["y0"])
                        in_left = x0 < (page_width * corner_threshold)
                        in_right = x0 > (page_width * (1 - corner_threshold))
                        in_top = y0 < (page_height * corner_threshold)
                        in_bottom = y0 > (page_height * (1 - corner_threshold))
                        single_corner_image = (in_left or in_right) and (in_top or in_bottom)
                except Exception as e:
                    logger.error(f"Profiler failed on page {i+1}: {e}")
                    cur = rec = lin = img = 0
                    single_corner_image = False

                stats["curves"].append(cur)
                stats["rects"].append(rec)
                stats["lines"].append(lin)
                stats["images"].append(img)
                stats["total_features"].append(cur + rec + lin + img)
                per_page.append({
                    "page_number": i + 1,
                    "curves": cur, "rects": rec, "lines": lin, "images": img,
                    "total_features": cur + rec + lin + img,
                    "single_corner_image": single_corner_image,
                })
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(pdf.pages)} pages.")

        def mad_thr(vals):
            v = np.array(vals)
            med = np.median(v)
            mad = np.median(np.abs(v - med))
            return med + 2.1 * mad

        thresholds = {k: mad_thr(v) for k, v in stats.items()}

        image_vals = np.array(stats["images"])
        image_median = np.median(image_vals)
        image_mad = np.median(np.abs(image_vals - image_median))

        if image_mad == 0 and image_median >= 1:
            sample_indices = np.linspace(0, len(per_page) - 1, min(5, len(per_page)), dtype=int)
            max_text_chars = 0
            with pdfplumber.open(pdf_path) as pdf:
                for idx in sample_indices:
                    try:
                        text = pdf.pages[idx].extract_text() or ""
                        max_text_chars = max(max_text_chars, len(text.strip()))
                    except Exception:
                        pass

            if max_text_chars <= 20:
                logger.warning(
                    f"Scanned-PDF detected: image MAD=0, median_images={image_median}, "
                    f"max sampled text chars={max_text_chars}. "
                    f"Forcing all image-bearing pages to likely_visual=True."
                )
                for p in per_page:
                    if p["images"] >= 1:
                        p["likely_visual"] = True
                        p["triggered_features"] = ["scanned_pdf_detected"]
                    else:
                        p["likely_visual"] = False
                        p["triggered_features"] = []

                visual_page_count = sum(1 for p in per_page if p["likely_visual"])
                logger.info(f"Visual pages identified (scanned-PDF override): {visual_page_count}/{len(per_page)}")
                return per_page, thresholds
        # --- End scanned-PDF detection ---

        visual_page_count = 0
        for p in per_page:
            flags = [
                k
                for k in ("curves", "rects", "lines", "total_features")
                if p[k] > thresholds[k]
            ]
            if not p["single_corner_image"] and p["images"] > thresholds["images"]:
                flags.append("images")
            p["likely_visual"] = bool(flags)
            p["triggered_features"] = flags
            if p["likely_visual"]:
                visual_page_count += 1

        logger.info(f"Visual pages identified: {visual_page_count}/{len(per_page)}")
        return per_page, thresholds

# ==== Page Processor ====
class PageProcessor:
    @staticmethod
    def extract_text(page_obj) -> str:
        return page_obj.extract_text() or ""

    @staticmethod
    def to_png_b64(page_obj, resolution: int = 300) -> str:
        img = page_obj.to_image(resolution=resolution).original
        buf = BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        b64_string = base64.b64encode(png_bytes).decode("utf-8")

        # Debug logging
        logger.debug(f"PNG generation: Image size = {img.size}, PNG bytes = {len(png_bytes)}, Base64 length = {len(b64_string)}")

        return b64_string

    @staticmethod
    def post_process_vision_response(text: str) -> dict:
        """Post-process vision response text to extract enriched_text and classification"""
        if not isinstance(text, str):
            logger.error(f"Vision response is not a string, got {type(text)}: {text}")
            return {"enriched_text": "", "classification": "Error"}

        # Hard error from gen_vision
        if text.startswith("ERROR:"):
            return {"enriched_text": "", "classification": "Error"}

        # Debug: log the full response for troubleshooting
        logger.debug(f"Full vision response ({len(text)} chars): {text}")

        et, cls = "", ""

        # Enhanced XML parsing with better error handling
        try:
            if "<enriched_text>" in text and "</enriched_text>" in text:
                start_tag = "<enriched_text>"
                end_tag = "</enriched_text>"
                start_idx = text.find(start_tag) + len(start_tag)
                end_idx = text.find(end_tag)
                if start_idx > len(start_tag) - 1 and end_idx > start_idx:
                    et = text[start_idx:end_idx].strip()
                    logger.debug(f"Extracted enriched_text ({len(et)} chars): {et[:100]}...")
                else:
                    logger.warning("Found enriched_text tags but failed to extract content")

            if "<classification>" in text and "</classification>" in text:
                start_tag = "<classification>"
                end_tag = "</classification>"
                start_idx = text.find(start_tag) + len(start_tag)
                end_idx = text.find(end_tag)
                if start_idx > len(start_tag) - 1 and end_idx > start_idx:
                    cls = text[start_idx:end_idx].strip()
                    logger.debug(f"Extracted classification: {cls}")
                else:
                    logger.warning("Found classification tags but failed to extract content")

        except Exception as e:
            logger.error(f"Error parsing XML tags: {e}")

        # If no XML tags found, use the entire response as enriched text
        if not et and not cls:
            logger.warning("No XML tags found or extraction failed, using full text as enriched_text")
            et = text.strip()
            cls = "NECESSARY" if et else "UNNECESSARY"

        # Default classification if missing
        cls = cls or "UNNECESSARY"

        # Final validation
        if not et:
            logger.warning("No enriched text extracted from vision response")
            return {"enriched_text": "", "classification": "Error"}

        logger.info(f"Vision processing successful: enriched_text={len(et)} chars, classification={cls}")
        return {"enriched_text": et, "classification": cls}

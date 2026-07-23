# Extracted verbatim from chunking_code.py — see PROVENANCE.md
import asyncio
import base64
import json
import random
from google import genai
from google.genai.types import GenerateContentConfig, Content, Part
import logging
from typing import Any, Dict, List, Optional, Tuple
from .chunker_utils import Utils
from .config import Config

logger = logging.getLogger("ai_chunking.chunkers.document_chunker")


# ==== LLM Service ====
class LLMService:
    def __init__(self, cfg: Config, api_key: str, use_vertex: bool = True):
        self.cfg = cfg
        self.use_vertex = use_vertex
        if use_vertex:
            # Agent Platform (aiplatform.googleapis.com) express-mode key auth.
            # The key's API restrictions must allow aiplatform.googleapis.com.
            # No api_version pin: v1beta is Developer-API-specific and breaks
            # Agent Platform routing.
            self.client = genai.Client(
                vertexai=True,
                api_key=api_key,
            )
        else:
            # Plain Gemini Developer API key.
            self.client = genai.Client(api_key=api_key)

    def count_tokens(self, text: str) -> int:
        try:
            return self.client.models.count_tokens(
                model=self.cfg.GEMINI_MODEL, contents=text
            ).total_tokens or 0
        except Exception:
            return Utils.token_estimate(text)

    async def agen_text(self, prompt: str, max_retries: int = 5, timeout_s: int = 180, model: Optional[str] = None) -> str:
        model_id = model or self.cfg.GEMINI_MODEL
        base_delay = 1.0
        for attempt in range(max_retries):
            try:
                async def _call():
                    # temperature=0 for determinism; response_mime_type=json
                    # forces Gemini to validate the output as JSON before
                    # returning, eliminating unescaped-quote / unescaped-
                    # newline bugs that break strict json.loads. Every caller
                    # of agen_text expects JSON output (docmeta, chunking,
                    # heading detection, bridges, metadata, verification,
                    # table extraction). freshness_extraction also tolerates
                    # JSON-quoted strings (existing strip('"') handles it).
                    resp = await self.client.aio.models.generate_content(
                        model=model_id,
                        contents=[{"role": "user", "parts": [{"text": prompt}]}],
                        config=GenerateContentConfig(
                            temperature=0.0,
                            response_mime_type="application/json",
                        ),
                    )

                    # Extract text from response candidates
                    if resp.candidates and len(resp.candidates) > 0:
                        if resp.candidates[0].content and resp.candidates[0].content.parts:
                            return resp.candidates[0].content.parts[0].text
                    return ""
                return await asyncio.wait_for(_call(), timeout=timeout_s)
            except asyncio.TimeoutError:
                if attempt == max_retries - 1:
                    logger.error(f"Async Gemini call timed out after {max_retries} attempts (timeout={timeout_s}s)")
                    raise RuntimeError(f"Async Gemini call timed out after {max_retries} attempts")
                delay = base_delay * (3 ** attempt) + random.uniform(1, 3)  # Longer delays for timeouts
                logger.warning(f"Timeout on attempt {attempt+1}/{max_retries}, retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
            except Exception as e:
                err = str(e).lower()
                retryable = any(t in err for t in ["quota", "rate limit", "backend error", "unavailable", "handler is closed"])
                non_retryable = "invalid argument" in err
                if non_retryable or attempt == max_retries - 1 or not retryable:
                    raise RuntimeError(f"Async Gemini call failed: {e}")
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                await asyncio.sleep(delay)

    async def agen_vision(self, prompt: str, base64_png: str, timeout_s: int = 120, model: Optional[str] = None) -> str:
        model_id = model or self.cfg.GEMINI_MODEL
        try:
            # Decode base64 to bytes for the image part
            import base64
            image_bytes = base64.b64decode(base64_png)

            # Debug: Log image size
            logger.debug(f"Vision API: Image size = {len(image_bytes)} bytes, Prompt length = {len(prompt)} chars")

            # Create content with proper types
            content = Content(
                role="user",
                parts=[
                    Part.from_text(text=prompt),
                    Part.from_bytes(data=image_bytes, mime_type="image/png")
                ]
            )

            # Create config with generation settings
            config = GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=4000  # Increased for detailed visual descriptions
            )

            resp = await self.client.aio.models.generate_content(
                model=model_id,
                contents=[content],
                config=config
            )

            # Debug: Log full response structure
            logger.debug(f"Vision API Response: candidates={len(resp.candidates) if resp.candidates else 0}")

            # Extract text from response candidates
            if resp.candidates and len(resp.candidates) > 0:
                candidate = resp.candidates[0]
                logger.debug(f"Candidate finish_reason: {candidate.finish_reason if hasattr(candidate, 'finish_reason') else 'N/A'}")

                if candidate.content and candidate.content.parts:
                    response_text = candidate.content.parts[0].text
                    logger.debug(f"Vision response length: {len(response_text)} chars")
                    return response_text
                else:
                    logger.warning("Vision API: No content.parts in response")
                    return ""
            else:
                logger.warning("Vision API: No candidates in response")
            return ""
        except Exception as e:
            logger.error(f"Gemini vision API error: {e}")
            return f"ERROR: Gemini API failed - {str(e)}"

    async def agen_vision_batch(self, prompt: str, base64_pngs: List[str], timeout_s: int = 240, model: Optional[str] = None) -> str:
        model_id = model or self.cfg.GEMINI_MODEL
        import base64 as _b64
        try:
            parts = [Part.from_text(text=prompt)]
            for b64 in base64_pngs:
                parts.append(Part.from_bytes(data=_b64.b64decode(b64), mime_type="image/png"))
            content = Content(role="user", parts=parts)
            config = GenerateContentConfig(temperature=0.0, max_output_tokens=8000)
            resp = await self.client.aio.models.generate_content(
                model=model_id, contents=[content], config=config,
            )
            if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
                return resp.candidates[0].content.parts[0].text or ""
            return ""
        except Exception as e:
            logger.error(f"Gemini vision-batch API error ({len(base64_pngs)} images): {e}")
            return f"ERROR: Gemini batch vision failed - {str(e)}"

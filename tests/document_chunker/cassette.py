"""Cassette record/replay layer for LLMService.

Wraps the verbatim ``LLMService`` from
``ai_chunking.chunkers.document_chunker.llm`` so that live Gemini responses
can be recorded to a JSON cassette and replayed deterministically in tests.

Mirrored method signatures (exactly as in llm.py):

    count_tokens(self, text: str) -> int                                   # SYNC
    async agen_text(self, prompt: str, max_retries: int = 5,
                    timeout_s: int = 180, model: Optional[str] = None) -> str
    async agen_vision(self, prompt: str, base64_png: str,
                      timeout_s: int = 120, model: Optional[str] = None) -> str
    async agen_vision_batch(self, prompt: str, base64_pngs: List[str],
                            timeout_s: int = 240, model: Optional[str] = None) -> str

Model resolution for cache keys: ``agen_text``/``agen_vision``/
``agen_vision_batch`` take ``model: Optional[str]``; the effective model used
for keying is ``model or cfg.GEMINI_MODEL``. The recorder reads the fallback
from ``inner.cfg``; the replayer must key identically, so a ``cfg`` object
(anything with a ``GEMINI_MODEL`` attribute) MUST be provided to
``CassetteReplayer`` for replay to resolve the same effective model.
``count_tokens`` always keys on ``cfg.GEMINI_MODEL`` (as llm.py always calls
the API with that model).
"""

import hashlib
import json
from typing import Any, Dict, List, Optional


class CassetteMiss(KeyError):
    """Raised when a replayed call has no recorded response in the cassette."""

    def __init__(self, method: str, key: str):
        super().__init__(f"Cassette miss for method={method!r} key={key}")
        self.method = method
        self.key = key


def _key(method: str, model: str, payload: dict) -> str:
    """Stable sha256 hex key for a (method, model, payload) triple."""
    blob = json.dumps({"method": method, "model": model, **payload}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _image_hashes(base64_pngs: List[str]) -> List[str]:
    """sha256 each image's b64 string (not the decoded bytes) for compact keys."""
    return [hashlib.sha256(b64.encode("utf-8")).hexdigest() for b64 in base64_pngs]


class CassetteRecorder:
    """Wraps a LIVE LLMService, records every response, delegates verbatim.

    ``count_tokens`` (sync) is recorded too: pipeline control flow depends on
    token counts, so replay must reproduce them exactly.

    Any attribute not defined here (e.g. ``cfg``, ``client``, ``use_vertex``)
    is delegated to the wrapped ``inner`` service via ``__getattr__``.
    """

    def __init__(self, inner, cassette_path):
        self._inner = inner
        self._cassette_path = cassette_path
        self._entries: Dict[str, Dict[str, Any]] = {}

    def _record(self, method: str, model: str, key: str, response) -> None:
        self._entries[key] = {"method": method, "model": model, "response": response}

    def count_tokens(self, text: str) -> int:
        model = self._inner.cfg.GEMINI_MODEL
        key = _key("count_tokens", model, {"text": text})
        response = self._inner.count_tokens(text)
        self._record("count_tokens", model, key, response)
        return response

    async def agen_text(self, prompt: str, max_retries: int = 5, timeout_s: int = 180, model: Optional[str] = None) -> str:
        effective_model = model or self._inner.cfg.GEMINI_MODEL
        key = _key("agen_text", effective_model, {"prompt": prompt})
        response = await self._inner.agen_text(
            prompt, max_retries=max_retries, timeout_s=timeout_s, model=model
        )
        self._record("agen_text", effective_model, key, response)
        return response

    async def agen_vision(self, prompt: str, base64_png: str, timeout_s: int = 120, model: Optional[str] = None) -> str:
        effective_model = model or self._inner.cfg.GEMINI_MODEL
        key = _key(
            "agen_vision",
            effective_model,
            {"prompt": prompt, "images": _image_hashes([base64_png])},
        )
        response = await self._inner.agen_vision(
            prompt, base64_png, timeout_s=timeout_s, model=model
        )
        self._record("agen_vision", effective_model, key, response)
        return response

    async def agen_vision_batch(self, prompt: str, base64_pngs: List[str], timeout_s: int = 240, model: Optional[str] = None) -> str:
        effective_model = model or self._inner.cfg.GEMINI_MODEL
        key = _key(
            "agen_vision_batch",
            effective_model,
            {"prompt": prompt, "images": _image_hashes(base64_pngs)},
        )
        response = await self._inner.agen_vision_batch(
            prompt, base64_pngs, timeout_s=timeout_s, model=model
        )
        self._record("agen_vision_batch", effective_model, key, response)
        return response

    def save(self) -> None:
        with open(self._cassette_path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, indent=2, sort_keys=True)

    def __getattr__(self, name):
        # Only reached when normal attribute lookup fails; delegate to the
        # live service (cfg, client, use_vertex, ...).
        return getattr(self._inner, name)


class CassetteReplayer:
    """Replays recorded LLMService responses from a cassette JSON file.

    ``cfg`` (anything exposing ``GEMINI_MODEL``) must be provided so that
    ``model or cfg.GEMINI_MODEL`` resolves to the same effective model the
    recorder used for keying. Retry/timeout kwargs are accepted and ignored.
    """

    def __init__(self, cassette_path, cfg=None):
        with open(cassette_path, "r", encoding="utf-8") as f:
            self._entries: Dict[str, Dict[str, Any]] = json.load(f)
        self.cfg = cfg

    def _default_model(self) -> str:
        if self.cfg is None:
            raise ValueError(
                "CassetteReplayer needs cfg (with GEMINI_MODEL) to resolve the "
                "effective model for keying; pass cfg= at construction."
            )
        return self.cfg.GEMINI_MODEL

    def _lookup(self, method: str, key: str):
        entry = self._entries.get(key)
        if entry is None:
            raise CassetteMiss(method, key)
        return entry["response"]

    def count_tokens(self, text: str) -> int:
        model = self._default_model()
        key = _key("count_tokens", model, {"text": text})
        return self._lookup("count_tokens", key)

    async def agen_text(self, prompt: str, max_retries: int = 5, timeout_s: int = 180, model: Optional[str] = None) -> str:
        effective_model = model or self._default_model()
        key = _key("agen_text", effective_model, {"prompt": prompt})
        return self._lookup("agen_text", key)

    async def agen_vision(self, prompt: str, base64_png: str, timeout_s: int = 120, model: Optional[str] = None) -> str:
        effective_model = model or self._default_model()
        key = _key(
            "agen_vision",
            effective_model,
            {"prompt": prompt, "images": _image_hashes([base64_png])},
        )
        return self._lookup("agen_vision", key)

    async def agen_vision_batch(self, prompt: str, base64_pngs: List[str], timeout_s: int = 240, model: Optional[str] = None) -> str:
        effective_model = model or self._default_model()
        key = _key(
            "agen_vision_batch",
            effective_model,
            {"prompt": prompt, "images": _image_hashes(base64_pngs)},
        )
        return self._lookup("agen_vision_batch", key)

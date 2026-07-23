"""Multi-format (PDF/CSV/Excel/image) LLM-driven document chunker, ported verbatim
from the Nexla production chunking pipeline (see PROVENANCE.md for the module-by-module
mapping). The public :class:`DocumentChunker` facade wraps the internal
``DocumentPipeline``, which profiles each file, routes it to a format-specific handler,
and uses Gemini for planning, chunking, metadata enrichment, and verification. The
``use_vertex`` flag selects how the API key is interpreted: ``True`` (the default,
matching production) means a Vertex Agent-Platform express-mode key; pass ``False``
for a plain Gemini Developer API key.
"""

import asyncio
import os
from typing import List, Optional

from ..base_chunker import BaseChunker
from ...models.chunk import Chunk
from .config import Config
from .pipeline import DocumentPipeline

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is a hard dependency of the pipeline
    pd = None

__all__ = ["DocumentChunker", "Config"]


def _ensure_no_running_loop(method_name: str) -> None:
    """Raise if called from inside a running asyncio event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop — safe to proceed with the sync API.
        return
    raise RuntimeError(
        f"DocumentChunker.{method_name}() is a synchronous API and cannot be called "
        "from inside a running asyncio event loop: the underlying pipeline handlers "
        "call asyncio.run() internally. Call it from synchronous code (e.g. via "
        "asyncio.to_thread or a worker thread)."
    )


class DocumentChunker(BaseChunker):
    """Multi-format (PDF/CSV/Excel/image) LLM-driven chunker ported from the Nexla
    production pipeline. Gemini-backed."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
        custom_instructions: Optional[str] = None,
        use_vertex: bool = True,
    ):
        super().__init__()
        api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "DocumentChunker requires a Gemini API key. Pass api_key=... or set "
                "the GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )
        self._pipeline = DocumentPipeline(
            config or Config(), api_key, custom_instructions, use_vertex=use_vertex
        )

    def chunk_document(self, file_path: str, extra_metadata: Optional[dict] = None) -> List[Chunk]:
        """Process a single file and return its chunks.

        Args:
            file_path: Path to the file (PDF/CSV/Excel/image) to process.
            extra_metadata: Optional dict merged into every chunk's metadata.

        Returns:
            List of chunks in (output_name, row-order) order.
        """
        _ensure_no_running_loop("chunk_document")
        meta = {"tags": {"display_path": str(file_path)}}
        results = self._pipeline.process_file(str(file_path), meta)

        chunks: List[Chunk] = []
        for output_name, df in results.items():
            for _, row in df.iterrows():
                row_dict = dict(row)
                text = str(row_dict.pop("chunk_text")) if "chunk_text" in row_dict else ""
                metadata = {}
                for key, value in row_dict.items():
                    try:
                        if pd is not None and pd.isna(value):
                            value = None
                    except (TypeError, ValueError):
                        # Non-scalar values (dict/list/ndarray) pass through unchanged.
                        pass
                    metadata[key] = value
                metadata["output_name"] = output_name
                if extra_metadata:
                    metadata.update(extra_metadata)
                chunks.append(Chunk(text=text, metadata=metadata))
        return chunks

    def chunk_documents(self, file_paths: List[str]) -> List[Chunk]:
        """Process multiple files and return all chunks concatenated."""
        all_chunks: List[Chunk] = []
        for file_path in file_paths:
            all_chunks.extend(self.chunk_document(file_path))
        return all_chunks

    def chunk_text(self, text: str) -> List[Chunk]:
        raise NotImplementedError(
            "DocumentChunker does not support chunking raw text: the multi-format "
            "pipeline requires a file path (PDF/CSV/Excel/image)."
        )

    def process_to_dataframes(self, file_path: str) -> dict:
        """Process a file and return the raw {output_name: DataFrame} mapping.

        This exposes the production pipeline's native output surface for
        platform compatibility and parity testing.
        """
        _ensure_no_running_loop("process_to_dataframes")
        meta = {"tags": {"display_path": str(file_path)}}
        return self._pipeline.process_file(str(file_path), meta)

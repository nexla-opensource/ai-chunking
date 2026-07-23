# document_chunker test fixtures

Small deterministic input files used for golden-output capture and parity tests
between the original monolith and the extracted `ai_chunking.chunkers.document_chunker` package.
Regenerate with `make_fixtures.py` (fixed literal data only; csv is byte-stable).
The PDF fixture is used by reference: `samples/loan-extraction.pdf` (130KB) at the repo root.

#!/usr/bin/env python
"""Capture golden outputs + LLM cassettes from the ORIGINAL monolith.

Runs the original production monolith (chunking_code.py) against a set of
fixture files, recording every LLM interaction into a "cassette" and every
output DataFrame into a golden JSON file. The migrated ai_chunking package
can then be parity-tested offline by replaying the cassettes and comparing
its outputs against the goldens.

Requires a real Gemini API key (GOOGLE_API_KEY / GEMINI_API_KEY or
--api-key). Use --dry-run to verify the monolith loads and the pipeline
constructs without making any API calls.

Example:
    python tests/document_chunker/capture_goldens.py \
        --api-key "$GOOGLE_API_KEY"
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MONOLITH = "/Users/tariq/Documents/tmp/chunking_code/chunking_code.py"
DEFAULT_OUT = str(HERE / "goldens")
FIXTURES_DIR = HERE / "fixtures"
SAMPLE_PDF = HERE.parent.parent / "samples" / "loan-extraction.pdf"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Capture golden outputs + LLM cassettes from the original monolith."
    )
    p.add_argument(
        "--monolith",
        default=DEFAULT_MONOLITH,
        help=f"Path to the monolith chunking_code.py (default: {DEFAULT_MONOLITH})",
    )
    p.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output directory for goldens + cassettes (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--fixtures",
        nargs="*",
        default=None,
        help=(
            "Fixture files to process (default: all files in "
            f"{FIXTURES_DIR} plus {SAMPLE_PDF})"
        ),
    )
    p.add_argument(
        "--no-vertex",
        action="store_true",
        default=False,
        help="Swap in a plain developer-key genai.Client for ordinary Gemini API "
        "keys (default: keep the monolith's vertexai=True express-mode client, "
        "matching production)",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key (default: GOOGLE_API_KEY or GEMINI_API_KEY env var)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Stop after loading the monolith and constructing the pipeline "
        "(no fixtures processed, no API calls)",
    )
    return p.parse_args(argv)


def resolve_api_key(args):
    key = args.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit(
            "ERROR: no API key. Pass --api-key or set GOOGLE_API_KEY / "
            "GEMINI_API_KEY in the environment."
        )
    return key


def default_fixtures():
    fixtures = []
    if FIXTURES_DIR.is_dir():
        fixtures.extend(sorted(p for p in FIXTURES_DIR.iterdir() if p.is_file()))
    else:
        print(f"WARNING: fixtures dir not found: {FIXTURES_DIR}", file=sys.stderr)
    if SAMPLE_PDF.is_file():
        fixtures.append(SAMPLE_PDF)
    else:
        print(f"WARNING: sample PDF not found: {SAMPLE_PDF}", file=sys.stderr)
    return fixtures


def load_monolith(path):
    path = Path(path)
    if not path.is_file():
        sys.exit(f"ERROR: monolith not found: {path}")
    spec = importlib.util.spec_from_file_location("chunking_monolith", str(path))
    if spec is None or spec.loader is None:
        sys.exit(f"ERROR: could not create import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["chunking_monolith"] = mod
    spec.loader.exec_module(mod)
    # The monolith calls logging.basicConfig(level=DEBUG) at import time; tame it.
    logging.getLogger().setLevel(logging.INFO)
    return mod


def serialize_df(df):
    # to_json handles NaN -> null and nested dicts/lists inside cells.
    return {
        "columns": list(df.columns),
        "records": json.loads(df.to_json(orient="records")),
    }


def recorded_call_count(rec):
    """Best-effort count of LLM interactions recorded by a CassetteRecorder."""
    for attr in ("calls", "records", "entries", "interactions", "cassette"):
        val = getattr(rec, attr, None)
        if isinstance(val, list):
            return len(val)
    return -1  # unknown; cassette API exposes no obvious list


def main(argv=None):
    args = parse_args(argv)

    if args.dry_run:
        api_key = args.api_key or "dry"
    else:
        api_key = resolve_api_key(args)

    mono = load_monolith(args.monolith)
    pipeline = mono.DocumentPipeline(mono.Config(), api_key, None)

    if args.dry_run:
        print("DRY RUN OK")
        return 0

    if args.no_vertex:
        # The monolith hardcodes genai.Client(vertexai=True, ...). Swap in a
        # plain developer-key client so ordinary Gemini API keys work without
        # editing the monolith.
        from google import genai

        pipeline.llm.client = genai.Client(api_key=api_key)

    # Make the sibling cassette.py importable.
    sys.path.insert(0, str(HERE))
    from cassette import CassetteRecorder

    fixtures = [Path(f) for f in args.fixtures] if args.fixtures else default_fixtures()
    if not fixtures:
        sys.exit("ERROR: no fixtures to process.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_llm = pipeline.llm
    summary = []  # (fixture_name, ok, n_outputs, n_rows, n_calls)
    any_failed = False

    for fixture in fixtures:
        name = fixture.name
        # Include the extension in the artifact name: fixtures like tiny.csv /
        # tiny.xlsx / tiny.png share a stem and would otherwise overwrite each other.
        stem = f"{fixture.stem}_{fixture.suffix.lstrip('.')}" if fixture.suffix else fixture.stem
        cassette_path = out_dir / f"{stem}.cassette.json"
        golden_path = out_dir / f"{stem}.golden.json"
        print(f"\n=== Capturing: {fixture} ===")
        try:
            if not fixture.is_file():
                raise FileNotFoundError(f"fixture not found: {fixture}")

            rec = CassetteRecorder(base_llm, cassette_path)
            pipeline.llm = rec

            # process_file resets token counters internally per handler.
            results = pipeline.process_file(str(fixture), {"tags": {"display_path": name}})

            outputs = {out_name: serialize_df(df) for out_name, df in (results or {}).items()}
            golden_path.write_text(
                json.dumps({"fixture": name, "outputs": outputs}, indent=2, default=str)
            )
            rec.save()

            n_rows = sum(len(o["records"]) for o in outputs.values())
            n_calls = recorded_call_count(rec)
            summary.append((name, True, len(outputs), n_rows, n_calls))
            print(f"    wrote {golden_path}")
            print(f"    wrote {cassette_path}")
        except Exception:
            traceback.print_exc()
            any_failed = True
            summary.append((name, False, 0, 0, 0))
        finally:
            pipeline.llm = base_llm

    print("\n=== Summary ===")
    print(f"{'fixture':<40} {'status':<8} {'outputs':>8} {'rows':>8} {'llm_calls':>10}")
    for name, ok, n_outputs, n_rows, n_calls in summary:
        status = "ok" if ok else "FAILED"
        calls = "?" if n_calls < 0 else str(n_calls)
        print(f"{name:<40} {status:<8} {n_outputs:>8} {n_rows:>8} {calls:>10}")

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())

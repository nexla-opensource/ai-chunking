"""Offline parity gate: migrated package vs. monolith golden outputs.

For every ``goldens/*.golden.json`` with a sibling ``*.cassette.json`` this test
replays the recorded LLM responses through the migrated
``ai_chunking.chunkers.document_chunker`` package and asserts the resulting
DataFrames match the golden byte-for-byte, modulo timing-related fields.

Goldens are captured against the original monolith by ``capture_goldens.py``
(requires a Gemini API key); this test itself is fully offline.
"""

import json
import math
import re
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

HERE = Path(__file__).resolve().parent
GOLDENS_DIR = HERE / "goldens"
FIXTURES_DIR = HERE / "fixtures"
REPO_ROOT = HERE.parent.parent
SAMPLES_DIR = REPO_ROOT / "samples"

# Keys dropped (at any nesting depth, incl. debug_info / llm_stats) before
# comparison because they are timing-dependent and legitimately differ run-to-run.
TIMING_KEY_RE = re.compile(r"(^|_)(time|elapsed|duration|timestamp|wallclock)", re.I)

MAX_REPORTED_MISMATCHES = 20


def _discover_pairs():
    """Return sorted list of (golden_path, cassette_path) pairs."""
    if not GOLDENS_DIR.is_dir():
        return []
    pairs = []
    for golden in sorted(GOLDENS_DIR.glob("*.golden.json")):
        cassette = golden.with_name(golden.name[: -len(".golden.json")] + ".cassette.json")
        if cassette.is_file():
            pairs.append((golden, cassette))
    return pairs


PAIRS = _discover_pairs()

if not PAIRS:
    pytest.skip(
        "no goldens captured yet — run capture_goldens.py with a Gemini API key",
        allow_module_level=True,
    )


def _resolve_fixture_path(recorded: str) -> Path:
    """Resolve the input-file path recorded in the golden json.

    Tries the recorded path as-is (if absolute), then relative to the fixtures
    dir, the samples dir, and the repo root; finally falls back to matching by
    basename under fixtures/ and samples/.
    """
    rec = Path(recorded)
    candidates = []
    if rec.is_absolute():
        candidates.append(rec)
    candidates.extend(
        [
            FIXTURES_DIR / recorded,
            SAMPLES_DIR / recorded,
            REPO_ROOT / recorded,
            FIXTURES_DIR / rec.name,
            SAMPLES_DIR / rec.name,
        ]
    )
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"cannot resolve fixture path {recorded!r}; tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _is_timing_key(key) -> bool:
    return isinstance(key, str) and TIMING_KEY_RE.search(key) is not None


def _compare(actual, expected, path, mismatches):
    """Recursively compare, appending '(json path, detail)' strings to mismatches."""
    if len(mismatches) >= MAX_REPORTED_MISMATCHES:
        return

    if isinstance(actual, dict) and isinstance(expected, dict):
        a_keys = [k for k in actual.keys() if not _is_timing_key(k)]
        e_keys = [k for k in expected.keys() if not _is_timing_key(k)]
        for k in e_keys:
            if k not in actual:
                mismatches.append(f"{path}.{k}: missing in actual")
        for k in a_keys:
            if k not in expected:
                mismatches.append(f"{path}.{k}: unexpected key in actual")
        for k in e_keys:
            if k in actual:
                _compare(actual[k], expected[k], f"{path}.{k}", mismatches)
        return

    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            mismatches.append(
                f"{path}: length mismatch actual={len(actual)} expected={len(expected)}"
            )
        for i in range(min(len(actual), len(expected))):
            _compare(actual[i], expected[i], f"{path}[{i}]", mismatches)
        return

    # Numeric comparison (bool is an int subclass — keep bools strict).
    a_num = isinstance(actual, (int, float)) and not isinstance(actual, bool)
    e_num = isinstance(expected, (int, float)) and not isinstance(expected, bool)
    if a_num and e_num:
        af, ef = float(actual), float(expected)
        if math.isnan(af) and math.isnan(ef):
            return
        if not math.isclose(af, ef, rel_tol=1e-9):
            mismatches.append(f"{path}: {actual!r} != {expected!r}")
        return

    if type(actual) is not type(expected) or actual != expected:
        mismatches.append(f"{path}: {actual!r} != {expected!r} (actual is what the package produced)")


def _serialize_outputs(outputs):
    """Serialize {name: DataFrame} exactly like capture_goldens.py does."""
    serialized = {}
    for name, df in outputs.items():
        serialized[name] = {
            "columns": list(df.columns),
            "records": json.loads(df.to_json(orient="records")),
        }
    return serialized


@pytest.mark.parametrize(
    "golden_path,cassette_path",
    PAIRS,
    ids=[g.name[: -len(".golden.json")] for g, _ in PAIRS],
)
def test_parity(golden_path, cassette_path):
    from ai_chunking.chunkers.document_chunker import DocumentChunker, Config  # noqa: F401
    from cassette import CassetteReplayer  # conftest puts the goldens dir on sys.path

    golden = json.loads(golden_path.read_text())

    recorded_fixture = None
    for key in ("fixture", "fixture_path", "input", "input_path", "source_file", "file_path"):
        if golden.get(key):
            recorded_fixture = golden[key]
            break
    assert recorded_fixture, (
        f"{golden_path.name}: no fixture path recorded in golden json "
        "(expected one of: fixture, fixture_path, input, input_path, source_file, file_path)"
    )
    fixture_path = _resolve_fixture_path(recorded_fixture)

    expected_outputs = golden.get("outputs", golden.get("dataframes"))
    assert isinstance(expected_outputs, dict), (
        f"{golden_path.name}: golden json has no 'outputs' mapping"
    )

    chunker = DocumentChunker(api_key="cassette-replay", use_vertex=False)
    cfg = chunker._pipeline.cfg
    chunker._pipeline.llm = CassetteReplayer(cassette_path, cfg=cfg)

    outputs = chunker.process_to_dataframes(str(fixture_path))
    actual_outputs = _serialize_outputs(outputs)

    mismatches = []
    _compare(actual_outputs, expected_outputs, "$", mismatches)
    assert not mismatches, (
        f"parity mismatch vs {golden_path.name} "
        f"(first {min(len(mismatches), MAX_REPORTED_MISMATCHES)} shown):\n"
        + "\n".join(mismatches[:MAX_REPORTED_MISMATCHES])
    )

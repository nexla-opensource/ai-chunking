"""Self-test for the cassette record/replay layer (uses a fake inner service)."""

import asyncio

import pytest

from cassette import CassetteMiss, CassetteRecorder, CassetteReplayer


class _FakeCfg:
    GEMINI_MODEL = "gemini-test-model"


class _FakeInner:
    """Stub LLMService with deterministic returns; mirrors llm.py signatures."""

    def __init__(self):
        self.cfg = _FakeCfg()
        self.use_vertex = True  # exercised via __getattr__ delegation

    def count_tokens(self, text):
        return len(text.split())

    async def agen_text(self, prompt, max_retries=5, timeout_s=180, model=None):
        return f"text::{model or self.cfg.GEMINI_MODEL}::{prompt}"

    async def agen_vision(self, prompt, base64_png, timeout_s=120, model=None):
        return f"vision::{prompt}::{len(base64_png)}"

    async def agen_vision_batch(self, prompt, base64_pngs, timeout_s=240, model=None):
        return f"vision_batch::{prompt}::{len(base64_pngs)}"


def test_round_trip_record_then_replay(tmp_path):
    cassette_path = tmp_path / "cassette.json"
    inner = _FakeInner()
    rec = CassetteRecorder(inner, cassette_path)

    # __getattr__ delegation to inner
    assert rec.cfg.GEMINI_MODEL == "gemini-test-model"
    assert rec.use_vertex is True

    # Record (count_tokens sync; agen_* async)
    n = rec.count_tokens("one two three")
    t_default = asyncio.run(rec.agen_text("hello world"))
    t_override = asyncio.run(rec.agen_text("hello world", model="other-model"))
    v = asyncio.run(rec.agen_vision("describe", "QUJD"))
    vb = asyncio.run(rec.agen_vision_batch("describe all", ["QUJD", "REVG"]))
    rec.save()
    assert cassette_path.exists()

    # Replay must reproduce identical results
    rep = CassetteReplayer(cassette_path, cfg=_FakeCfg())
    assert rep.count_tokens("one two three") == n == 3
    assert asyncio.run(rep.agen_text("hello world")) == t_default
    # Explicit model equal to cfg default keys identically to model=None
    assert asyncio.run(rep.agen_text("hello world", model="gemini-test-model")) == t_default
    assert asyncio.run(rep.agen_text("hello world", model="other-model")) == t_override
    assert asyncio.run(rep.agen_vision("describe", "QUJD")) == v
    assert asyncio.run(rep.agen_vision_batch("describe all", ["QUJD", "REVG"])) == vb

    # Ignored kwargs must not affect keying
    assert asyncio.run(rep.agen_text("hello world", max_retries=1, timeout_s=5)) == t_default


def test_cassette_miss_on_unknown_prompt(tmp_path):
    cassette_path = tmp_path / "cassette.json"
    rec = CassetteRecorder(_FakeInner(), cassette_path)
    asyncio.run(rec.agen_text("known prompt"))
    rec.save()

    rep = CassetteReplayer(cassette_path, cfg=_FakeCfg())
    with pytest.raises(CassetteMiss) as exc_info:
        asyncio.run(rep.agen_text("unknown prompt"))
    assert "agen_text" in str(exc_info.value)

    with pytest.raises(CassetteMiss):
        rep.count_tokens("never recorded")

    with pytest.raises(CassetteMiss):
        asyncio.run(rep.agen_vision("known prompt", "ZGlmZmVyZW50LWltYWdl"))


def test_replayer_requires_cfg_for_default_model(tmp_path):
    cassette_path = tmp_path / "cassette.json"
    rec = CassetteRecorder(_FakeInner(), cassette_path)
    asyncio.run(rec.agen_text("p", model="explicit-model"))
    rec.save()

    rep = CassetteReplayer(cassette_path)  # no cfg
    # Explicit model works without cfg...
    assert asyncio.run(rep.agen_text("p", model="explicit-model")) == "text::explicit-model::p"
    # ...but default-model resolution needs cfg
    with pytest.raises(ValueError):
        asyncio.run(rep.agen_text("p"))

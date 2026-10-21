"""Vendor adapters against a mocked transport: parsing, retries, timing, env keys."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf

from harfiyyan.models.api.deepgram import Deepgram
from harfiyyan.models.api.elevenlabs import ElevenLabsScribe
from harfiyyan.models.api.openai import OpenAITranscribe


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    p = tmp_path / "a.wav"
    sf.write(str(p), np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
    return p


def _mount(model, handler):
    model.client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5)


def test_deepgram_parse_and_params(monkeypatch, wav):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "k")
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["auth"] = req.headers["authorization"]
        body = {"results": {"channels": [{"alternatives": [{"transcript": "مرحبا"}]}]}}
        return httpx.Response(200, json=body)

    m = Deepgram()
    _mount(m, handler)
    t = m.transcribe_path(wav)
    assert t.text == "مرحبا" and t.attempts == 1 and t.wall_s >= 0
    assert "model=nova-3" in seen["url"] and "language=ar" in seen["url"]
    assert seen["auth"] == "Token k"


def test_elevenlabs_multipart(monkeypatch, wav):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setenv("ELEVENLABS_STT_MODEL", "scribe_v1")

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["xi-api-key"] == "k"
        assert b'name="model_id"' in req.content and b"scribe_v1" in req.content
        assert b'name="language_code"' in req.content
        return httpx.Response(200, json={"text": " نص "})

    m = ElevenLabsScribe()
    _mount(m, handler)
    assert m.transcribe_path(wav).text == "نص"


def test_openai_retries_on_429_then_succeeds(monkeypatch, wav):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json={"text": "ok"})

    m = OpenAITranscribe()
    m.max_attempts = 5
    _mount(m, handler)
    # Make the retry wait instant.
    import harfiyyan.models.api._http as h

    monkeypatch.setattr(h, "wait_exponential_jitter", lambda **k: __import__("tenacity").wait_none())
    t = m.transcribe_path(wav)
    assert t.text == "ok" and t.attempts == 3 and calls["n"] == 3


def test_non_retryable_error_raises(monkeypatch, wav):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "k")
    m = Deepgram()
    _mount(m, lambda req: httpx.Response(401, text="bad key"))
    with pytest.raises(httpx.HTTPStatusError):
        m.transcribe_path(wav)


def test_missing_key_is_clear_error(monkeypatch, wav):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    m = Deepgram()
    _mount(m, lambda req: httpx.Response(200, json={}))
    with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
        m.transcribe_path(wav)

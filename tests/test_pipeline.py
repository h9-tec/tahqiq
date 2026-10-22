"""End to end on synthetic data: manifest → runner (fake model) → cache → score.

No GPU, no network. Guards the contract between runner, io, report and metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

from harfiyyan import io as hio
from harfiyyan.manifest import Utterance, make_id, write_manifest, write_manifest_hash
from harfiyyan.models import registry
from harfiyyan.models.base import ASRModel
from harfiyyan.report import markdown_table, score_model
from harfiyyan.runner import run_model

REFS = ["السلام عليكم ورحمة الله", "رقم الهاتف ٠٥٠ ١٢٣", "كيف الحال اليوم", "الحمد لله على كل حال"]


class Echo(ASRModel):
    """Returns a canned hypothesis per utterance id; errors on one id."""

    kind = "open"

    def __init__(self, model_id: str, table: dict[str, str], fail_id: str | None = None):
        self.model_id = model_id
        self.display_name = model_id
        self.table = table
        self.fail_id = fail_id
        self._current = None

    def _marker(self) -> None:
        pass

    def transcribe_path(self, wav_path: Path):
        from harfiyyan.models.base import Transcript

        uid = wav_path.stem
        if uid == self.fail_id:
            raise RuntimeError("boom")
        return Transcript(text=self.table[uid], wall_s=0.001)


@pytest.fixture
def world(tmp_path: Path):
    manifests, audio, results = tmp_path / "m", tmp_path / "a", tmp_path / "r"
    utts = []
    for i, ref in enumerate(REFS):
        uid = make_id("toy", str(i))
        rel = f"toy/{uid}.wav"
        p = audio / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), np.zeros(8000, dtype=np.float32), 16000, subtype="PCM_16")
        utts.append(Utterance(uid, "toy", "test", str(i), rel, "0" * 64, 0.5, ref,
                              "Egypt", "egyptian", has_number=("٠" in ref), in_api_slice=True))
    write_manifest(manifests / "toy.jsonl", utts)
    write_manifest_hash(manifests)
    return manifests, audio, results, utts


def test_run_score_roundtrip(world, monkeypatch):
    manifests, audio, results, utts = world
    perfect = {u.id: u.reference for u in utts}
    worse = dict(perfect)
    worse[utts[0].id] = "السلام عليكم"              # 2 deletions
    worse[utts[1].id] = "رقم الهاتف ٠٥٠ ٣٢١"          # digits wrong
    monkeypatch.setitem(registry.REGISTRY, "base", lambda: Echo("base", perfect))
    monkeypatch.setitem(registry.REGISTRY, "cand", lambda: Echo("cand", worse, fail_id=utts[3].id))

    run_model("base", ["toy"], manifests_dir=manifests, audio_dir=audio, results_dir=results)
    run = run_model("cand", ["toy"], manifests_dir=manifests, audio_dir=audio, results_dir=results)
    assert run["totals"]["errors"] == 1
    assert run["manifest_sha256"] == (manifests / "MANIFEST_SHA256").read_text().strip()

    # Resume: a second run only retries the errored utterance.
    run2 = run_model("cand", ["toy"], manifests_dir=manifests, audio_dir=audio, results_dir=results)
    assert run2["totals"]["done"] == 1 and run2["totals"]["skipped"] == 3

    s_base = score_model("base", ["toy"], manifests_dir=manifests, results_dir=results, n_boot=50, baseline_id="base")
    s_cand = score_model("cand", ["toy"], manifests_dir=manifests, results_dir=results, n_boot=50, baseline_id="base")
    assert s_base["datasets"]["toy"]["elm"]["wer"]["point"] == 0.0
    assert s_cand["datasets"]["toy"]["elm"]["wer"]["point"] > 0.0
    assert s_cand["coverage"]["toy"] == {"n": 4, "scored": 3}
    assert s_cand["probes"]["numeral"] == {"acc": s_cand["probes"]["numeral"]["acc"], "k": 0, "n": 1}
    assert s_base["probes"]["numeral"]["k"] == 1
    assert s_cand["vs_baseline"]["delta_wer"]["point"] > 0
    assert s_cand["dialects"] == {}  # fewer than 20 utts per dialect → no dialect row

    md = markdown_table([s_base, s_cand], ["toy"])
    assert "| base |" in md and "| cand |" in md

    # Cache file is the documented schema.
    line = json.loads((results / "cand" / "toy.jsonl").read_text().splitlines()[0])
    assert set(line) == {"id", "hyp", "wall_s", "attempts", "error", "run_id"}


def test_entity_probe_containment():
    from harfiyyan.report import entity_probe
    from harfiyyan import io as hio
    u = Utterance("a", "toy", "test", "0", "toy/a.wav", "0" * 64, 1.0, "سافر أحمد إلى الرياض", "", "unknown")
    ents = {"a": ["أحمد", "الرياض"]}
    good = {"a": hio.Prediction("a", "سافر احمد الى الرياض", 0.1)}
    half = {"a": hio.Prediction("a", "سافر محمد الى الرياض", 0.1)}
    assert entity_probe([u], good, ents)["k"] == 2
    r = entity_probe([u], half, ents)
    assert (r["k"], r["n"]) == (1, 2)


def test_load_entities_filters(tmp_path):
    from harfiyyan.report import load_entities
    p = tmp_path / "entities.jsonl"
    p.write_text(json.dumps({"id": "a", "dataset": "x", "entities": [
        {"text": "أحمد", "type": "PERS", "score": 0.95},
        {"text": "##ان", "type": "PERS", "score": 0.95},
        {"text": "الله", "type": "PERS", "score": 0.99},
        {"text": "سق", "type": "LOC", "score": 0.9},
        {"text": "القاهرة", "type": "LOC", "score": 0.5},
        {"text": "شيء", "type": "MISC", "score": 0.99},
    ]}, ensure_ascii=False) + "\n", encoding="utf-8")
    assert load_entities(p) == {"a": ["أحمد"]}

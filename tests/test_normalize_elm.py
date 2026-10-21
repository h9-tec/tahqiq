"""The ``elm`` normalizer must equal Elm's original function, string for string.

The oracle is Elm's ``eval.py`` vendored verbatim under ``tests/oracle``; only
its NeMo import is stubbed so the file loads without NeMo.
"""

from __future__ import annotations

import json
import random
import sys
import types
from pathlib import Path

import pytest

from harfiyyan.normalize.elm import normalize_elm

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests" / "oracle" / "elm_eval_original.py"
ELM_MANIFESTS = sorted((ROOT / "data" / "elm").glob("*.json"))


def _load_oracle():
    # Stub the NeMo import chain so the vendored file executes unchanged.
    nemo = types.ModuleType("nemo")
    coll = types.ModuleType("nemo.collections")
    asr = types.ModuleType("nemo.collections.asr")
    metrics = types.ModuleType("nemo.collections.asr.metrics")
    wer_mod = types.ModuleType("nemo.collections.asr.metrics.wer")
    wer_mod.word_error_rate = lambda *a, **k: None
    for name, mod in [
        ("nemo", nemo), ("nemo.collections", coll), ("nemo.collections.asr", asr),
        ("nemo.collections.asr.metrics", metrics), ("nemo.collections.asr.metrics.wer", wer_mod),
    ]:
        sys.modules.setdefault(name, mod)
    ns: dict = {}
    exec(compile(ORACLE.read_text(encoding="utf-8"), str(ORACLE), "exec"), ns)
    return ns["normalize_arabic_text"]


oracle = _load_oracle()

HAND_CASES = [
    "",
    "   ",
    "و الله",                      # standalone waw glued
    "و  و كلمة",                  # two standalone waws
    "قال: «نعم»، ثم ذهب؟!",        # punctuation deleted, no space inserted
    "كلمة،كلمة",                   # punctuation between words, no space
    "اَلسَّلَامُ عَلَيْكُم",         # tashkeel
    "أحمد إبراهيم آمن",            # alef variants
    "سؤال قائل شيء",               # hamza forms
    "پاكستان ڤيديو",               # Persian letters
    "٠١٢٣٤٥٦٧٨٩ 0123456789",       # digits
    "مدرسة على",                   # ة and ى NOT folded by Elm
    "hello مرحبا world",           # Latin kept
    "a-b c_d e/f",                 # Latin punctuation
    "ـــ تطويل ـــ",               # tatweel NOT stripped by Elm
    "ٱلله",                        # wasla NOT folded by Elm
    "۱۲",                # Persian digits NOT mapped by Elm
    "\t tabs \n and nbsp ",   # whitespace kinds
]


@pytest.mark.parametrize("text", HAND_CASES)
def test_hand_cases_match_oracle(text):
    assert normalize_elm(text) == oracle(text)


def test_idempotent_on_hand_cases():
    for t in HAND_CASES:
        once = normalize_elm(t)
        assert normalize_elm(once) == once


@pytest.mark.skipif(not ELM_MANIFESTS, reason="Elm manifests not present")
def test_manifest_references_match_oracle():
    rng = random.Random(0)
    checked = 0
    for mf in ELM_MANIFESTS:
        lines = mf.read_text(encoding="utf-8").splitlines()
        for line in rng.sample(lines, min(300, len(lines))):
            text = json.loads(line)["text"]
            assert normalize_elm(text) == oracle(text), (mf.name, text)
            checked += 1
    assert checked >= 1000

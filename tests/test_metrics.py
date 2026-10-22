import pytest

from tahqiq.metrics import (
    bootstrap_ci,
    cer,
    mcnemar_pair,
    paired_delta_ci,
    proportion_ci,
    wer,
)


def test_wer_known_answers():
    assert wer(["hello world"], ["hello world"]) == 0.0
    assert wer(["hello"], ["hello world"]) == 1.0
    assert wer(["hello world"], ["hello there"]) == 0.5
    # Corpus-level, not mean of per-utterance: (0 + 1) / (1 + 2) = 1/3
    assert wer(["a", "b c"], ["a", "b"]) == pytest.approx(1 / 3)


def test_cer_known_answers():
    assert cer(["abc"], ["abc"]) == 0.0
    assert cer(["abc"], ["abd"]) == pytest.approx(1 / 3)


def test_bootstrap_identical_is_zero():
    ci = bootstrap_ci(["a b", "c d"], ["a b", "c d"], n_iter=100)
    assert (ci.lo, ci.point, ci.hi) == (0.0, 0.0, 0.0)


def test_bootstrap_brackets_point_and_is_deterministic():
    refs = ["السلام عليكم", "كيف الحال", "الحمد لله", "مرحبا بكم"] * 5
    hyps = ["السلام عليكم", "كيف الحال", "الحمد", "مرحبا"] * 5
    a = bootstrap_ci(refs, hyps, n_iter=300, seed=42)
    b = bootstrap_ci(refs, hyps, n_iter=300, seed=42)
    assert a == b
    assert a.lo <= a.point <= a.hi and 0 < a.point < 1


def test_paired_delta_sign():
    refs = ["a b c"] * 30
    good = ["a b c"] * 30
    bad = ["a b"] * 30
    d = paired_delta_ci(refs, good, bad, n_iter=200)
    assert d.point == pytest.approx(1 / 3)
    assert d.lo > 0  # bad is worse than good, stably


def test_proportion_ci():
    ci = proportion_ci(50, 100)
    assert ci.point == 0.5 and ci.lo < 0.5 < ci.hi
    assert proportion_ci(0, 0).hi == 0.0


def test_mcnemar():
    refs = ["a"] * 20
    a = ["a"] * 20
    b = ["a"] * 5 + ["b"] * 15
    out = mcnemar_pair(refs, a, b)
    assert out["b"] == 0 and out["c"] == 15 and out["p_value"] < 0.001
    assert mcnemar_pair(refs, a, a)["p_value"] == 1.0

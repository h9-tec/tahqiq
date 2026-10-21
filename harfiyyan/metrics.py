"""WER / CER with bootstrap confidence intervals and paired tests.

Ported from the Saudi TTS bench (``bench/metrics.py``).

Conventions (match Elm's board so the ``elm`` column is comparable):
- WER and CER are corpus-level: total edits / total reference tokens.
  jiwer computes exactly this, as does NeMo's ``word_error_rate``.
- CER tokens are characters including spaces (NeMo ``use_cer=True`` behaviour).
- Inputs are already-normalized strings; this module never normalizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from jiwer import cer as _jiwer_cer
from jiwer import wer as _jiwer_wer

Metric = Literal["wer", "cer"]


def wer(refs: list[str], hyps: list[str]) -> float:
    return float(_jiwer_wer(refs, hyps))


def cer(refs: list[str], hyps: list[str]) -> float:
    return float(_jiwer_cer(refs, hyps))


_METRIC_FN = {"wer": wer, "cer": cer}


@dataclass(frozen=True)
class CI:
    lo: float
    point: float
    hi: float

    def __repr__(self) -> str:
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"

    def as_dict(self) -> dict:
        return {"point": self.point, "lo": self.lo, "hi": self.hi}


def _check_same_len(*lists) -> None:
    n = {len(x) for x in lists}
    if len(n) != 1:
        raise ValueError(f"length mismatch: {[len(x) for x in lists]}")


def bootstrap_ci(
    refs: list[str],
    hyps: list[str],
    *,
    metric: Metric = "wer",
    n_iter: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> CI:
    """Utterance-level percentile bootstrap of a corpus-level metric."""
    _check_same_len(refs, hyps)
    if not refs:
        return CI(0.0, 0.0, 0.0)
    fn = _METRIC_FN[metric]
    point = fn(refs, hyps)
    n = len(refs)
    rng = np.random.default_rng(seed)
    refs_arr = np.array(refs, dtype=object)
    hyps_arr = np.array(hyps, dtype=object)
    samples = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        samples[i] = fn(refs_arr[idx].tolist(), hyps_arr[idx].tolist())
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CI(float(lo), point, float(hi))


def paired_delta_ci(
    refs: list[str],
    hyps_a: list[str],
    hyps_b: list[str],
    *,
    metric: Metric = "wer",
    n_iter: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> CI:
    """Bootstrap CI of metric(B) - metric(A) resampling the same utterances for both.

    Negative point = B is better than A. The CI excludes 0 when the difference
    is stable across resamples; this is the number the board's "vs baseline"
    column shows.
    """
    _check_same_len(refs, hyps_a, hyps_b)
    if not refs:
        return CI(0.0, 0.0, 0.0)
    fn = _METRIC_FN[metric]
    point = fn(refs, hyps_b) - fn(refs, hyps_a)
    n = len(refs)
    rng = np.random.default_rng(seed)
    r = np.array(refs, dtype=object)
    a = np.array(hyps_a, dtype=object)
    b = np.array(hyps_b, dtype=object)
    samples = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        rr = r[idx].tolist()
        samples[i] = fn(rr, b[idx].tolist()) - fn(rr, a[idx].tolist())
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CI(float(lo), float(point), float(hi))


def proportion_ci(k: int, n: int, *, alpha: float = 0.05) -> CI:
    """Wilson score interval for a probe accuracy k/n."""
    if n == 0:
        return CI(0.0, 0.0, 0.0)
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return CI(float(max(0.0, centre - half)), float(p), float(min(1.0, centre + half)))


def _per_utt_correct(refs: list[str], hyps: list[str]) -> np.ndarray:
    return np.array([r == h for r, h in zip(refs, hyps)], dtype=np.int32)


def mcnemar_pair(refs: list[str], hyps_a: list[str], hyps_b: list[str]) -> dict:
    """McNemar's test on per-utterance exact-match correctness, A vs B.

    b = A wrong & B right, c = A right & B wrong. Exact binomial when
    b + c <= 25, otherwise chi-square with continuity correction.
    """
    _check_same_len(refs, hyps_a, hyps_b)
    a = _per_utt_correct(refs, hyps_a)
    b = _per_utt_correct(refs, hyps_b)
    a_wrong_b_right = int(np.sum((a == 0) & (b == 1)))
    a_right_b_wrong = int(np.sum((a == 1) & (b == 0)))
    n_disc = a_wrong_b_right + a_right_b_wrong
    if n_disc == 0:
        return {"b": 0, "c": 0, "p_value": 1.0, "method": "no_discordant"}
    if n_disc <= 25:
        from scipy.stats import binomtest

        k = min(a_wrong_b_right, a_right_b_wrong)
        p = binomtest(k, n_disc, 0.5, alternative="two-sided").pvalue
        method = "exact"
    else:
        from scipy.stats import chi2 as _chi2

        stat = (abs(a_wrong_b_right - a_right_b_wrong) - 1) ** 2 / n_disc
        p = float(_chi2.sf(stat, df=1))
        method = "chi2_continuity"
    return {"b": a_wrong_b_right, "c": a_right_b_wrong, "p_value": float(p), "method": method}

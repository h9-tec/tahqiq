"""WER / CER with bootstrap confidence intervals and paired tests.

Conventions (match Elm's board so the ``elm`` column is comparable):
- WER and CER are corpus-level: total edits / total reference tokens, the same
  quantity jiwer and NeMo's ``word_error_rate`` report.
- CER tokens are characters including spaces.
- Inputs are already-normalized strings; this module never normalizes.
- Utterances whose normalized reference is empty are dropped (they have no
  denominator); callers report how many.

Bootstrap: per-utterance (edits, reference length) are computed once, then
resampled with numpy, so 2,000 resamples over 40k utterances take about a
second instead of re-aligning strings every iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from rapidfuzz.distance import Levenshtein

Metric = Literal["wer", "cer"]


def _tokens(s: str, metric: Metric) -> list[str]:
    return s.split() if metric == "wer" else list(s)


def utt_stats(refs: list[str], hyps: list[str], metric: Metric = "wer") -> tuple[np.ndarray, np.ndarray]:
    """Per-utterance (edits, reference_length). Empty references get length 0."""
    _check_same_len(refs, hyps)
    edits = np.empty(len(refs), dtype=np.int64)
    lens = np.empty(len(refs), dtype=np.int64)
    for i, (r, h) in enumerate(zip(refs, hyps)):
        rt, ht = _tokens(r, metric), _tokens(h, metric)
        lens[i] = len(rt)
        edits[i] = Levenshtein.distance(rt, ht)
    return edits, lens


def _ratio(edits: np.ndarray, lens: np.ndarray) -> float:
    tot = lens.sum()
    return float(edits.sum() / tot) if tot > 0 else 0.0


def wer(refs: list[str], hyps: list[str]) -> float:
    return _ratio(*utt_stats(refs, hyps, "wer"))


def cer(refs: list[str], hyps: list[str]) -> float:
    return _ratio(*utt_stats(refs, hyps, "cer"))


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


def _bootstrap_ratio(
    num: np.ndarray, den: np.ndarray, *, n_iter: int, seed: int, alpha: float, num_b: np.ndarray | None = None
) -> tuple[float, float]:
    """Percentile CI of sum(num)/sum(den) (or of the difference when num_b given) under resampling."""
    n = len(den)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_iter, n))
    d = den[idx].sum(axis=1).astype(np.float64)
    d[d == 0] = np.nan
    a = num[idx].sum(axis=1) / d
    if num_b is not None:
        a = num_b[idx].sum(axis=1) / d - a
    lo, hi = np.nanpercentile(a, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


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
    edits, lens = utt_stats(refs, hyps, metric)
    point = _ratio(edits, lens)
    if lens.sum() == 0:
        return CI(0.0, 0.0, 0.0)
    lo, hi = _bootstrap_ratio(edits, lens, n_iter=n_iter, seed=seed, alpha=alpha)
    return CI(lo, point, hi)


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
    """Bootstrap CI of metric(B) - metric(A), resampling the same utterances for both.

    Negative point = B is better than A. This is the "vs baseline" column.
    """
    _check_same_len(refs, hyps_a, hyps_b)
    if not refs:
        return CI(0.0, 0.0, 0.0)
    ea, lens = utt_stats(refs, hyps_a, metric)
    eb, _ = utt_stats(refs, hyps_b, metric)
    if lens.sum() == 0:
        return CI(0.0, 0.0, 0.0)
    point = _ratio(eb, lens) - _ratio(ea, lens)
    lo, hi = _bootstrap_ratio(ea, lens, n_iter=n_iter, seed=seed, alpha=alpha, num_b=eb)
    return CI(lo, point, hi)


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

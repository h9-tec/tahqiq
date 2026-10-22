"""Contamination checks: declared training-data overlap and an empirical anomaly test.

Two independent signals, both shown on the board per (system, test set) cell:

1. **Declared** (``data/contamination.yaml``): the model card or paper says the system was
   trained or fine-tuned on data from the same corpus as a test set (SADA, MGB-2, Common
   Voice, MASC, Casablanca). Each entry cites its source. A declared cell is marked
   "in-domain" and excluded from the system's average by default.

2. **Empirical** (``anomalies``): with many systems on the board, log WER is well described
   by an additive model  log WER[m, d] ≈ a[m] + b[d]  (system difficulty + set difficulty).
   The fit is a median polish, which is robust to a few contaminated cells. A cell whose
   residual is far below the fit (the system does much better on that set than its own
   level and the set's level predict) is flagged "anomalous advantage". This cannot prove
   contamination, but it is exactly the pattern a test-set leak produces, and it is the
   check a reader cannot run from a single self-reported number. Thresholds are published:
   residual ≤ -0.30 (≈ 26% lower WER than expected) and robust z ≤ -2.5.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DECLARED_YAML = ROOT / "data" / "contamination.yaml"
RESIDUAL_T = -0.30
Z_T = -2.5
MIN_MODELS = 6


def load_declared(path: Path = DECLARED_YAML) -> dict[str, dict[str, dict]]:
    """model_id → dataset → {source, note}."""
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, dict[str, dict]] = {}
    for model_id, entry in (raw.get("declared") or {}).items():
        out[model_id] = {}
        for ds in entry.get("datasets") or []:
            out[model_id][ds] = {"source": entry.get("source", ""), "note": entry.get("note", "")}
    return out


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def median_polish(table: dict[str, dict[str, float]], iters: int = 10) -> dict[str, dict[str, float]]:
    """Residuals of an additive fit on a sparse model × dataset table (values = log WER)."""
    models = list(table)
    datasets = sorted({d for row in table.values() for d in row})
    a = {m: 0.0 for m in models}
    b = {d: 0.0 for d in datasets}
    mu = _median([v for row in table.values() for v in row.values()])
    for _ in range(iters):
        for m in models:
            vals = [table[m][d] - mu - b[d] for d in table[m]]
            if vals:
                a[m] = _median(vals)
        for d in datasets:
            vals = [table[m][d] - mu - a[m] for m in models if d in table[m]]
            if vals:
                b[d] = _median(vals)
    return {m: {d: table[m][d] - mu - a[m] - b[d] for d in table[m]} for m in models}


def anomalies(table: dict[str, dict[str, float]]) -> dict[str, dict[str, dict]]:
    """Flag cells with a large negative residual. ``table`` holds WER (0..1+) per model and dataset."""
    if len(table) < MIN_MODELS:
        return {}
    logs = {m: {d: math.log(max(w, 1e-3)) for d, w in row.items()} for m, row in table.items()}
    res = median_polish(logs)
    all_res = [v for row in res.values() for v in row.values()]
    med = _median(all_res)
    mad = _median([abs(v - med) for v in all_res]) or 1e-6
    out: dict[str, dict[str, dict]] = {}
    for m, row in res.items():
        for d, r in row.items():
            z = 0.6745 * (r - med) / mad
            if r <= RESIDUAL_T and z <= Z_T:
                out.setdefault(m, {})[d] = {"residual": round(r, 3), "z": round(z, 2), "expected_wer": round(math.exp(logs[m][d] - r), 4)}
    return out


def contamination_report(scores: list[dict], *, slice_pref: tuple[str, ...] = ("full", "api"), normalizer: str = "elm") -> dict:
    """Combine declared overlap and empirical anomalies for every system on the board."""
    declared = load_declared()
    table: dict[str, dict[str, float]] = {}
    for s in scores:
        sl = next((s["scores"][k] for k in slice_pref if k in s.get("scores", {})), None)
        if not sl:
            continue
        row = {d: v[normalizer]["wer"]["point"] for d, v in sl["datasets"].items() if normalizer in v}
        if row:
            table[s["model_id"]] = row
    anom = anomalies(table)
    cells: dict[str, dict[str, dict]] = {}
    for m in set(declared) | set(anom):
        cells[m] = {}
        for d in set(declared.get(m, {})) | set(anom.get(m, {})):
            cells[m][d] = {"declared": d in declared.get(m, {}), **declared.get(m, {}).get(d, {}),
                           "anomaly": d in anom.get(m, {}), **anom.get(m, {}).get(d, {})}
    return {"cells": cells, "thresholds": {"residual": RESIDUAL_T, "z": Z_T, "min_models": MIN_MODELS},
            "n_models_in_fit": len(table), "normalizer": normalizer}

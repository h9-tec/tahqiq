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


REF_CONVENTIONS = ROOT / "data" / "probes" / "ref_conventions.json"


def fingerprint_flags(scores: list[dict]) -> dict:
    """Convention-fingerprint report from the per-system vectors stored at publish time."""
    import json

    from .fingerprint import fingerprint_report

    if not REF_CONVENTIONS.exists():
        return {}
    refs = json.loads(REF_CONVENTIONS.read_text(encoding="utf-8"))
    hyp: dict[str, dict[str, dict]] = {}
    for s in scores:
        fp = s.get("fingerprint") or {}
        sl = "all" if "full" in s.get("scores", {}) else "api"
        row = {d: v[sl] for d, v in fp.items() if sl in v}
        if row:
            hyp[s["model_id"]] = row
    ref_vec = {d: r["all"] for d, r in refs.items()}
    return fingerprint_report(ref_vec, hyp)


def _robust_z(values: dict[str, float]) -> dict[str, float]:
    if len(values) < 3:
        return {}
    med = _median(list(values.values()))
    mad = _median([abs(v - med) for v in values.values()]) or 1e-6
    return {k: 0.6745 * (v - med) / mad for k, v in values.items()}


def dev_gap_flags(scores: list[dict], *, gap_t: float = -0.25, z_t: float = -2.5) -> dict[str, dict[str, dict]]:
    """Dev-vs-test gap: log WER(test common slice) - log WER(dev slice) per corpus.

    A clean system scores about the same on both (same distribution). A system that saw the test
    split scores markedly better on test. Flag when the gap is <= gap_t and, across systems on that
    corpus, the robust z of the gap is <= z_t (so a corpus whose dev split is simply harder for
    everyone is not flagged)."""
    gaps: dict[str, dict[str, float]] = {}   # corpus -> model -> gap
    for s in scores:
        api = (s.get("scores") or {}).get("api") or {}
        dev = (s.get("extras") or {}).get("dev") or {}
        for ds, cell in api.get("datasets", {}).items():
            w_test = cell.get("elm", {}).get("wer", {}).get("point")
            w_dev = dev.get(ds, {}).get("wer")
            if w_test and w_dev:
                gaps.setdefault(ds, {})[s["model_id"]] = math.log(max(w_test, 1e-3)) - math.log(max(w_dev, 1e-3))
    out: dict[str, dict[str, dict]] = {}
    for ds, per in gaps.items():
        z = _robust_z(per)
        for m, g in per.items():
            if g <= gap_t and z.get(m, 0) <= z_t:
                out.setdefault(m, {})[ds] = {"gap": round(g, 3), "z": round(z[m], 2), "wer_dev": None}
    # attach the dev WER for the flagged cells
    for s in scores:
        dev = (s.get("extras") or {}).get("dev") or {}
        for ds in out.get(s["model_id"], {}):
            out[s["model_id"]][ds]["wer_dev"] = dev.get(ds, {}).get("wer")
    return out


def pert_gap_flags(scores: list[dict], *, delta_t: float = 0.15, z_t: float = 2.5) -> dict[str, dict[str, dict]]:
    """Perturbation gap: mean over variants of log WER(perturbed) - log WER(original) per corpus.

    Memorized audio/transcripts lose their advantage under word-preserving perturbation, so a
    contaminated system degrades much more on that corpus than it does elsewhere and than peers do.
    Flag when the corpus delta exceeds the system's own median delta by >= delta_t and the robust z
    across systems on that corpus is >= z_t."""
    deltas: dict[str, dict[str, float]] = {}
    for s in scores:
        api = (s.get("scores") or {}).get("api") or {}
        pert = (s.get("extras") or {}).get("pert") or {}
        per_ds: dict[str, list[float]] = {}
        for key, v in pert.items():               # key = "<variant>__<ds>"
            variant, _, ds = key.partition("__")
            w0 = api.get("datasets", {}).get(ds, {}).get("elm", {}).get("wer", {}).get("point")
            w1 = v.get("wer")
            if w0 and w1:
                per_ds.setdefault(ds, []).append(math.log(max(w1, 1e-3)) - math.log(max(w0, 1e-3)))
        if per_ds:
            deltas[s["model_id"]] = {ds: sum(x) / len(x) for ds, x in per_ds.items()}
    out: dict[str, dict[str, dict]] = {}
    by_ds: dict[str, dict[str, float]] = {}
    for m, per in deltas.items():
        for ds, dl in per.items():
            by_ds.setdefault(ds, {})[m] = dl
    for m, per in deltas.items():
        own = _median(list(per.values()))
        for ds, dl in per.items():
            z = _robust_z(by_ds[ds]).get(m, 0)
            if dl - own >= delta_t and z >= z_t:
                out.setdefault(m, {})[ds] = {"delta": round(dl, 3), "own_median": round(own, 3), "z": round(z, 2)}
    return out


def likelihood_flags(scores: list[dict], *, rate_t: float = 0.15, z_t: float = -2.5) -> dict[str, dict[str, dict]]:
    """Likelihood probe (seq2seq decoders): share of utterances where the reference transcript is
    *more* probable under the model than the model's own greedy output, per corpus. Normal systems
    prefer their own output almost always; memorized references flip that. Flag when a corpus's
    rate exceeds the system's own median rate by >= rate_t and its robust z (lower = more normal)
    across corpora for that system is beyond z_t in the suspicious direction."""
    out: dict[str, dict[str, dict]] = {}
    for s in scores:
        lik = (s.get("extras") or {}).get("likelihood") or {}
        rates = {ds: v.get("ref_preferred_rate") for ds, v in lik.items() if v.get("ref_preferred_rate") is not None}
        if len(rates) < 3:
            continue
        own = _median(list(rates.values()))
        z = _robust_z({k: -v for k, v in rates.items()})   # negate so "more suspicious" is more negative
        for ds, r in rates.items():
            if r - own >= rate_t and z.get(ds, 0) <= z_t:
                out.setdefault(s["model_id"], {})[ds] = {"ref_preferred_rate": round(r, 3), "own_median": round(own, 3), "z": round(z[ds], 2)}
    return out


def contamination_report(scores: list[dict], *, slice_pref: tuple[str, ...] = ("full", "api"), normalizer: str = "elm") -> dict:
    """Combine declared overlap, empirical WER anomalies and convention fingerprints for every system."""
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
    try:
        fpf = fingerprint_flags(scores).get("flags", {})
    except Exception:  # noqa: BLE001
        fpf = {}
    gaps = dev_gap_flags(scores)
    perts = pert_gap_flags(scores)
    liks = likelihood_flags(scores)
    cells: dict[str, dict[str, dict]] = {}
    for m in set(declared) | set(anom) | set(fpf) | set(gaps) | set(perts) | set(liks):
        cells[m] = {}
        for d in set(declared.get(m, {})) | set(anom.get(m, {})) | set(fpf.get(m, {})) | set(gaps.get(m, {})) | set(perts.get(m, {})) | set(liks.get(m, {})):
            cells[m][d] = {"declared": d in declared.get(m, {}), **declared.get(m, {}).get(d, {}),
                           "anomaly": d in anom.get(m, {}), **anom.get(m, {}).get(d, {}),
                           "fingerprint": d in fpf.get(m, {}), **({"fp_" + k: v for k, v in fpf[m][d].items()} if d in fpf.get(m, {}) else {}),
                           "devgap": d in gaps.get(m, {}), **({"dev_" + k: v for k, v in gaps[m][d].items()} if d in gaps.get(m, {}) else {}),
                           "pertgap": d in perts.get(m, {}), **({"pert_" + k: v for k, v in perts[m][d].items()} if d in perts.get(m, {}) else {}),
                           "likelihood": d in liks.get(m, {}), **({"lik_" + k: v for k, v in liks[m][d].items()} if d in liks.get(m, {}) else {})}
    return {"cells": cells, "thresholds": {"residual": RESIDUAL_T, "z": Z_T, "min_models": MIN_MODELS, "fingerprint_z": -2.5,
                                            "dev_gap": -0.25, "pert_delta": 0.15, "likelihood_rate": 0.15},
            "n_models_in_fit": len(table), "normalizer": normalizer}

"""Score cached predictions against manifests.

This is the single scoring path for the pilot, the site build, and any
replication run. Inputs are manifests + results JSONL; outputs are plain dicts
so they serialize to ``dist/summary.json`` unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from . import io as hio
from .manifest import Utterance, load_manifest
from .metrics import bootstrap_ci, mcnemar_pair, paired_delta_ci, proportion_ci
from .normalize import NORMALIZERS
from .normalize.numbers import digit_string, has_number

BASELINE = "whisper_large_v3"


def load_entities(path: Path) -> dict[str, list[str]]:
    """id → list of reference entity strings (from scripts/make_entities.py); {} if absent."""
    if not path.exists():
        return {}
    import json

    out: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                if d["entities"]:
                    out[d["id"]] = [e["text"] for e in d["entities"]]
    return out


def entity_probe(utts: list[Utterance], preds: dict[str, hio.Prediction], entities: dict[str, list[str]]) -> dict:
    """Per-entity containment after tolerant normalization. Returns k, n and Wilson CI."""
    norm = NORMALIZERS["tolerant"]
    k = n = 0
    for u in utts:
        ents = entities.get(u.id)
        p = preds.get(u.id)
        if not ents or p is None or p.error:
            continue
        hyp = f" {norm(p.hyp)} "
        for e in ents:
            ne = norm(e)
            if not ne:
                continue
            n += 1
            k += int(f" {ne} " in hyp or ne in hyp)
    return {"acc": proportion_ci(k, n).as_dict(), "k": k, "n": n}


def _pairs(utts: list[Utterance], preds: dict[str, hio.Prediction], norm) -> tuple[list[str], list[str], list[Utterance]]:
    refs, hyps, kept = [], [], []
    for u in utts:
        p = preds.get(u.id)
        if p is None or p.error:
            continue  # scored on the utterances a system produced output for; coverage is reported
        refs.append(norm(u.reference))
        hyps.append(norm(p.hyp))
        kept.append(u)
    return refs, hyps, kept


def score_model(
    model_id: str,
    datasets: list[str],
    *,
    manifests_dir: Path,
    results_dir: Path,
    slice_name: str | None = None,
    normalizers: tuple[str, ...] = ("elm", "tolerant"),
    n_boot: int = 2000,
    baseline_id: str | None = BASELINE,
) -> dict:
    """Return {dataset: {normalizer: {...}}, dialect: {...}, probes: {...}, coverage: {...}}."""
    from .runner import select_utts

    out: dict = {"model_id": model_id, "datasets": {}, "dialects": {}, "probes": {}, "coverage": {}}
    all_utts: list[Utterance] = []
    all_preds: dict[str, hio.Prediction] = {}
    base_preds: dict[str, hio.Prediction] = {}
    for ds in datasets:
        utts = select_utts(load_manifest(manifests_dir / f"{ds}.jsonl"), slice_name)
        preds = hio.load_predictions(results_dir, model_id, ds)
        if baseline_id and baseline_id != model_id:
            base_preds.update(hio.load_predictions(results_dir, baseline_id, ds))
        n_ok = sum(1 for u in utts if u.id in preds and not preds[u.id].error)
        out["coverage"][ds] = {"n": len(utts), "scored": n_ok}
        out["datasets"][ds] = {}
        for nm in normalizers:
            norm = NORMALIZERS[nm]
            refs, hyps, _ = _pairs(utts, preds, norm)
            if not refs:
                continue
            w = bootstrap_ci(refs, hyps, metric="wer", n_iter=n_boot)
            c = bootstrap_ci(refs, hyps, metric="cer", n_iter=n_boot)
            out["datasets"][ds][nm] = {"wer": w.as_dict(), "cer": c.as_dict(), "n": len(refs)}
        all_utts.extend(utts)
        all_preds.update(preds)

    # Per-dialect (tolerant + elm), pooled across datasets.
    by_dialect: dict[str, list[Utterance]] = defaultdict(list)
    for u in all_utts:
        by_dialect[u.dialect].append(u)
    for dialect, utts in sorted(by_dialect.items()):
        cell: dict = {}
        for nm in normalizers:
            refs, hyps, _ = _pairs(utts, all_preds, NORMALIZERS[nm])
            if len(refs) < 20:
                continue
            w = bootstrap_ci(refs, hyps, metric="wer", n_iter=n_boot)
            cell[nm] = {"wer": w.as_dict(), "n": len(refs)}
        if cell:
            out["dialects"][dialect] = cell

    # Numeral probe: references with digits or an unambiguous number word; compared
    # as digit strings with number words mapped on both sides. Computed from the
    # reference at score time so the manifest hash does not depend on the word list.
    num_utts = [u for u in all_utts if has_number(u.reference)]
    k = n = 0
    for u in num_utts:
        p = all_preds.get(u.id)
        if p is None or p.error:
            continue
        n += 1
        k += int(digit_string(u.reference, map_number_words=True) == digit_string(p.hyp, map_number_words=True))
    out["probes"]["numeral"] = {"acc": proportion_ci(k, n).as_dict(), "k": k, "n": n}

    # Entity probe (only when data/probes/entities.jsonl exists).
    entities = load_entities(manifests_dir.parent / "probes" / "entities.jsonl")
    if entities:
        out["probes"]["entity"] = entity_probe(all_utts, all_preds, entities)

    # Paired comparison against the baseline on shared utterances.
    if base_preds and baseline_id != model_id:
        norm = NORMALIZERS["elm"]
        refs, a, b = [], [], []
        for u in all_utts:
            pa, pb = base_preds.get(u.id), all_preds.get(u.id)
            if pa is None or pb is None or pa.error or pb.error:
                continue
            refs.append(norm(u.reference))
            a.append(norm(pa.hyp))
            b.append(norm(pb.hyp))
        if refs:
            d = paired_delta_ci(refs, a, b, metric="wer", n_iter=n_boot)
            out["vs_baseline"] = {
                "baseline": baseline_id,
                "delta_wer": d.as_dict(),
                "mcnemar": mcnemar_pair(refs, a, b),
                "n": len(refs),
            }
    return out


def markdown_table(scores: list[dict], datasets: list[str], normalizer: str = "elm") -> str:
    """Compact table: one row per model, WER ± CI per dataset, numeral probe, delta vs baseline."""
    head = ["model"] + datasets + ["numeral acc", "ΔWER vs base"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in scores:
        row = [s["model_id"]]
        for ds in datasets:
            cell = s["datasets"].get(ds, {}).get(normalizer)
            if cell:
                w = cell["wer"]
                row.append(f"{100*w['point']:.1f} [{100*w['lo']:.1f}, {100*w['hi']:.1f}]")
            else:
                row.append("—")
        pr = s["probes"].get("numeral")
        row.append(f"{100*pr['acc']['point']:.0f}% (n={pr['n']})" if pr and pr["n"] else "—")
        vb = s.get("vs_baseline")
        row.append(f"{100*vb['delta_wer']['point']:+.1f} [{100*vb['delta_wer']['lo']:+.1f}, {100*vb['delta_wer']['hi']:+.1f}]" if vb else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

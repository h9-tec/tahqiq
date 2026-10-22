"""Build dist/ (summary.json, models.json, meta.json) from results/ + manifests, or check consistency.

    uv run python scripts/build.py build            # scores every results/<model>/ with a run.json
    uv run python scripts/build.py check            # every run.json hash == current manifest hash
    uv run python scripts/build.py table            # print the headline markdown table

Scoring runs through harfiyyan.report.score_model; nothing here computes a number itself.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from harfiyyan.manifest import load_manifest, manifest_paths, read_manifest_hash  # noqa: E402
from harfiyyan.report import BASELINE, markdown_table, score_model  # noqa: E402

MANIFESTS = ROOT / "data" / "manifests"
RESULTS = ROOT / "results"
DIST = ROOT / "dist"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _runs() -> dict[str, dict]:
    out = {}
    for d in sorted(RESULTS.iterdir()) if RESULTS.exists() else []:
        rj = d / "run.json"
        if rj.exists():
            out[d.name] = json.loads(rj.read_text(encoding="utf-8"))
    return out


def dataset_cards() -> dict:
    cards = {}
    for p in manifest_paths(MANIFESTS):
        utts = load_manifest(p)
        cards[p.stem] = {
            "n": len(utts),
            "hours": round(sum(u.duration_s for u in utts) / 3600, 2),
            "elm_members": sum(1 for u in utts if u.elm_member),
            "n_api_slice": sum(1 for u in utts if u.in_api_slice),
            "dialects": sorted({u.dialect for u in utts}),
        }
    return cards


def cmd_check() -> int:
    current = read_manifest_hash(MANIFESTS)
    bad = []
    for mid, run in _runs().items():
        if run.get("manifest_sha256") != current:
            bad.append((mid, run.get("manifest_sha256")))
    if bad:
        for mid, h in bad:
            print(f"STALE {mid}: run hash {h} != current {current}")
        return 1
    print(f"ok: {len(_runs())} runs carry manifest {current}")
    return 0


def cmd_build(n_boot: int) -> None:
    current = read_manifest_hash(MANIFESTS)
    datasets = [p.stem for p in manifest_paths(MANIFESTS)]
    runs = _runs()
    models_meta = (yaml.safe_load((ROOT / "data" / "models.yaml").read_text(encoding="utf-8")) or {}).get("models", {})
    rows = []
    for mid, run in runs.items():
        if run.get("manifest_sha256") != current:
            print(f"skip {mid}: stale manifest hash")
            continue
        scores = {}
        for slice_name in ("full", "api"):
            s = score_model(mid, datasets, manifests_dir=MANIFESTS, results_dir=RESULTS,
                            slice_name=slice_name, n_boot=n_boot, baseline_id=BASELINE)
            if any(s["datasets"].values()):
                scores[slice_name] = s
        rows.append({
            "model_id": mid,
            "display_name": run.get("display_name") or mid,
            "kind": run.get("kind"),
            "hf_id": run.get("hf_id"),
            "card": models_meta.get(mid, {}),
            "run": {k: run.get(k) for k in ("run_id", "git_sha", "finished", "env", "decode_params", "language_hint", "slice")},
            "scores": scores,
        })
        print(f"scored {mid}")
    DIST.mkdir(exist_ok=True)
    meta = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_sha": _git_sha(),
            "manifest_sha256": current, "n_boot": n_boot, "baseline": BASELINE}
    (DIST / "summary.json").write_text(json.dumps({"meta": meta, "datasets": dataset_cards(), "models": rows},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    (DIST / "models.json").write_text(json.dumps(models_meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (DIST / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"dist/ written: {len(rows)} models, {len(datasets)} datasets")


def cmd_table(slice_name: str, normalizer: str) -> None:
    summary = json.loads((DIST / "summary.json").read_text(encoding="utf-8"))
    datasets = list(summary["datasets"])
    scores = [m["scores"][slice_name] for m in summary["models"] if slice_name in m["scores"]]
    scores.sort(key=lambda s: sum(s["datasets"][d][normalizer]["wer"]["point"] for d in datasets if normalizer in s["datasets"].get(d, {})) / max(1, len(datasets)))
    print(markdown_table(scores, datasets, normalizer))


README_START = "<!-- harfiyyan:table:start -->"
README_END = "<!-- harfiyyan:table:end -->"


def readme_table(slice_name: str = "full", normalizer: str = "elm") -> str:
    """Headline table for README: avg WER, per-set WER, probes, Δ vs baseline. Generated only."""
    summary = json.loads((DIST / "summary.json").read_text(encoding="utf-8"))
    datasets = list(summary["datasets"])
    label = {"sada": "SADA", "casablanca": "Casablanca", "commonvoice": "CV18", "masc_clean": "MASC-c", "masc_noisy": "MASC-n", "mgb2": "MGB-2"}
    rows = []
    for m in summary["models"]:
        sc = m["scores"].get(slice_name)
        if not sc:
            continue
        cells, pts = [], []
        for d in datasets:
            c = sc["datasets"].get(d, {}).get(normalizer)
            if c:
                pts.append(c["wer"]["point"])
                cells.append(f"{100*c['wer']['point']:.1f}")
            else:
                cells.append("·")
        avg = sum(pts) / len(pts) if pts else None
        num = sc["probes"].get("numeral"); ent = sc["probes"].get("entity"); vb = sc.get("vs_baseline")
        rows.append((avg if avg is not None else 9e9, [
            m["display_name"], f"{100*avg:.1f}" if avg is not None else "·", *cells,
            f"{100*num['acc']['point']:.0f}%" if num and num["n"] else "·",
            f"{100*ent['acc']['point']:.0f}%" if ent and ent["n"] else "·",
            f"{100*vb['delta_wer']['point']:+.1f}" if vb else "base",
        ]))
    rows.sort(key=lambda r: r[0])
    head = ["System", "Avg WER", *[label.get(d, d) for d in datasets], "Numerals", "Entities", "Δ vs Whisper v3"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r[1]) + " |" for r in rows]
    meta = summary["meta"]
    out.append("")
    out.append(f"WER % on the {'full test sets' if slice_name == 'full' else 'common stratified slice'}, `{normalizer}` normalizer, "
               f"corpus-level, sorted by unweighted average over {len(datasets)} sets. Confidence intervals, CER, the tolerant "
               f"normalizer and per-dialect numbers are on the site. Manifest `{meta['manifest_sha256'][:12]}`, built {meta['generated_at'][:10]}.")
    return "\n".join(out)


def cmd_readme() -> None:
    p = ROOT / "README.md"
    text = p.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise SystemExit("README.md lacks the harfiyyan table markers")
    pre, rest = text.split(README_START, 1)
    _, post = rest.split(README_END, 1)
    new = f"{pre}{README_START}\n{readme_table()}\n{README_END}{post}"
    p.write_text(new, encoding="utf-8")
    print("README table updated")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n-boot", type=int, default=2000)
    sub.add_parser("check")
    sub.add_parser("readme")
    t = sub.add_parser("table")
    t.add_argument("--slice", default="full")
    t.add_argument("--normalizer", default="elm")
    args = ap.parse_args()
    if args.cmd == "build":
        cmd_build(args.n_boot)
    elif args.cmd == "check":
        sys.exit(cmd_check())
    elif args.cmd == "readme":
        cmd_readme()
    else:
        cmd_table(args.slice, args.normalizer)


if __name__ == "__main__":
    main()

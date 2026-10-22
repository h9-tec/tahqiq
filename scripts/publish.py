"""Score a finished local run and publish it to the Hub results store.

    uv run python scripts/publish.py whisper_large_v3_turbo
    uv run python scripts/publish.py whisper_large_v3_turbo --hf-id openai/whisper-large-v3-turbo --display-name "Whisper large-v3-turbo"

Writes results/<slug>/scores.json locally too. Needs HF_TOKEN in .env.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import yaml  # noqa: E402

from tahqiq import hub  # noqa: E402
from tahqiq.profile import profile  # noqa: E402
from tahqiq.fingerprint import mean_vector  # noqa: E402
from tahqiq import io as hio  # noqa: E402
from tahqiq.manifest import load_manifest  # noqa: E402
from tahqiq.manifest import manifest_paths, read_manifest_hash  # noqa: E402
from tahqiq.report import BASELINE, score_model  # noqa: E402

MANIFESTS = ROOT / "data" / "manifests"
RESULTS = ROOT / "results"


def build_scores(slug: str, *, n_boot: int = 2000, extra: dict | None = None, slices: tuple[str, ...] = ("full", "api")) -> dict:
    run = json.loads((RESULTS / slug / "run.json").read_text(encoding="utf-8"))
    current = read_manifest_hash(MANIFESTS)
    if run.get("manifest_sha256") != current:
        raise SystemExit(f"{slug}: run manifest {run.get('manifest_sha256')} != current {current}")
    datasets = [p.stem for p in manifest_paths(MANIFESTS)]
    cards = (yaml.safe_load((ROOT / "data" / "models.yaml").read_text(encoding="utf-8")) or {}).get("models", {})
    scores = {}
    for slice_name in slices:
        s = score_model(slug, datasets, manifests_dir=MANIFESTS, results_dir=RESULTS, slice_name=slice_name,
                        n_boot=n_boot, baseline_id=BASELINE)
        if any(s["datasets"].values()):
            scores[slice_name] = s
    hf_id = (extra or {}).get("hf_id") or run.get("hf_id")
    timing_path = RESULTS / slug / "timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else None
    # Transcript-convention fingerprint: mean feature vector of this system's hypotheses per set and slice.
    fp = {}
    for ds in datasets:
        preds = hio.load_predictions(RESULTS, slug, ds)
        utts = load_manifest(MANIFESTS / f"{ds}.jsonl")
        for sl, sel in (("all", utts), ("api", [u for u in utts if u.in_api_slice])):
            hyps = [preds[u.id].hyp for u in sel if u.id in preds and not preds[u.id].error]
            v = mean_vector(hyps) if len(hyps) >= 50 else None
            if v:
                fp.setdefault(ds, {})[sl] = v
    # Extra passes for the contamination suite (common slice): dev split, perturbed audio, likelihood probe.
    from tahqiq.metrics import wer as _wer
    from tahqiq.normalize import NORMALIZERS

    extras: dict = {}
    sub = RESULTS / "_sub"
    def _wer_for(manifest_path: Path, preds_dir_slug: str, results_root: Path) -> dict | None:
        if not manifest_path.exists():
            return None
        utts = [u for u in load_manifest(manifest_path) if u.in_api_slice]
        preds = hio.load_predictions(results_root, preds_dir_slug, manifest_path.stem)
        pairs = [(NORMALIZERS["elm"](u.reference), NORMALIZERS["elm"](preds[u.id].hyp)) for u in utts if u.id in preds and not preds[u.id].error]
        if len(pairs) < 50:
            return None
        return {"wer": round(_wer([r for r, _ in pairs], [h for _, h in pairs]), 4), "n": len(pairs)}
    for ds in datasets:
        v = _wer_for(ROOT / "data" / "dev_manifests" / f"dev_{ds}.jsonl", slug, sub / "dev")
        if v:
            extras.setdefault("dev", {})[ds] = v
    for mp in sorted((ROOT / "data" / "pert_manifests").glob("*__*.jsonl")) if (ROOT / "data" / "pert_manifests").exists() else []:
        v = _wer_for(mp, slug, sub / "pert")
        if v:
            extras.setdefault("pert", {})[mp.stem] = v
    lik_dir = sub / "lik" / slug
    if lik_dir.exists():
        for lp in sorted(lik_dir.glob("*.jsonl")):
            rows = [json.loads(l) for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
            rows = [r for r in rows if r.get("nll_ref") is not None and r.get("nll_hyp") is not None]
            if len(rows) >= 50:
                pref = sum(1 for r in rows if r["nll_ref"] < r["nll_hyp"]) / len(rows)
                extras.setdefault("likelihood", {})[lp.stem] = {"ref_preferred_rate": round(pref, 4), "n": len(rows)}
    doc = {
        "extras": extras,
        "fingerprint": fp,
        "profile": profile(slug, hf_id),
        "timing": timing,
        "model_id": slug,
        "display_name": (extra or {}).get("display_name") or run.get("display_name") or slug,
        "hf_id": (extra or {}).get("hf_id") or run.get("hf_id"),
        "kind": run.get("kind"),
        "card": cards.get(slug, {}),
        "submitted_by": (extra or {}).get("submitted_by", "maintainer"),
        "run": {k: run.get(k) for k in ("run_id", "git_sha", "finished", "env", "decode_params", "language_hint", "slice", "totals")},
        "manifest_sha256": current,
        "baseline": BASELINE,
        "n_boot": n_boot,
        "scores": scores,
        "published_at": hub.now(),
    }
    (RESULTS / slug / "scores.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--hf-id", default=None)
    ap.add_argument("--display-name", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--slices", default="full,api", help="which slices to score, e.g. 'api' for a common-slice-only run")
    args = ap.parse_args()
    doc = build_scores(args.slug, n_boot=args.n_boot, extra={"hf_id": args.hf_id, "display_name": args.display_name},
                       slices=tuple(x for x in args.slices.split(",") if x))
    first = next(iter(doc["scores"].values()))
    print(f"scored {args.slug}: " + ", ".join(
        f"{d} {100*v['elm']['wer']['point']:.1f}" for d, v in first["datasets"].items() if "elm" in v))
    if not args.no_upload:
        hub.upload_results(args.slug, RESULTS / args.slug, doc)
        print("uploaded:", hub.results_url(args.slug))


if __name__ == "__main__":
    main()

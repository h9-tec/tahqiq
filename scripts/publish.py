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
from tahqiq.manifest import manifest_paths, read_manifest_hash  # noqa: E402
from tahqiq.report import BASELINE, score_model  # noqa: E402

MANIFESTS = ROOT / "data" / "manifests"
RESULTS = ROOT / "results"


def build_scores(slug: str, *, n_boot: int = 2000, extra: dict | None = None) -> dict:
    run = json.loads((RESULTS / slug / "run.json").read_text(encoding="utf-8"))
    current = read_manifest_hash(MANIFESTS)
    if run.get("manifest_sha256") != current:
        raise SystemExit(f"{slug}: run manifest {run.get('manifest_sha256')} != current {current}")
    datasets = [p.stem for p in manifest_paths(MANIFESTS)]
    cards = (yaml.safe_load((ROOT / "data" / "models.yaml").read_text(encoding="utf-8")) or {}).get("models", {})
    scores = {}
    for slice_name in ("full", "api"):
        s = score_model(slug, datasets, manifests_dir=MANIFESTS, results_dir=RESULTS, slice_name=slice_name,
                        n_boot=n_boot, baseline_id=BASELINE)
        if any(s["datasets"].values()):
            scores[slice_name] = s
    hf_id = (extra or {}).get("hf_id") or run.get("hf_id")
    timing_path = RESULTS / slug / "timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else None
    doc = {
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
    args = ap.parse_args()
    doc = build_scores(args.slug, n_boot=args.n_boot, extra={"hf_id": args.hf_id, "display_name": args.display_name})
    print(f"scored {args.slug}: " + ", ".join(
        f"{d} {100*v['elm']['wer']['point']:.1f}" for d, v in doc["scores"]["full"]["datasets"].items() if "elm" in v))
    if not args.no_upload:
        hub.upload_results(args.slug, RESULTS / args.slug, doc)
        print("uploaded:", hub.results_url(args.slug))


if __name__ == "__main__":
    main()

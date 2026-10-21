"""M0 pilot gate: a ~300-utterance slice through Whisper + 3 vendor APIs.

Everything the pilot writes goes under ``--out`` (default ~/research/harfiyyan/pilot),
NOT under results/, so the public manifest hash and results stay clean.

    uv run python scripts/pilot.py build  --per-dialect 19          # 8 dialects × 19 ≈ 150 utts
    uv run python scripts/pilot.py run    --model whisper_large_v3_turbo
    uv run python scripts/pilot.py run    --model deepgram_nova3
    uv run python scripts/pilot.py score

Requires data/manifests/casablanca.jsonl (scripts/prepare_data.py casablanca) and audio/.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from harfiyyan.manifest import load_manifest, write_manifest, write_manifest_hash  # noqa: E402
from harfiyyan.report import markdown_table, score_model  # noqa: E402
from harfiyyan.runner import AUDIO, MANIFESTS, run_model  # noqa: E402

DEFAULT_OUT = Path.home() / "research" / "harfiyyan" / "pilot"
PILOT_MODELS = ["whisper_large_v3_turbo", "deepgram_nova3", "elevenlabs_scribe", "openai_gpt4o_transcribe"]


def cmd_build(args: argparse.Namespace) -> None:
    out = Path(args.out)
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for name in args.dataset:
        utts = load_manifest(MANIFESTS / f"{name}.jsonl")
        by: dict[str, list] = defaultdict(list)
        for u in utts:
            if 1.0 <= u.duration_s <= 20.0 and u.reference.strip():
                by[u.dialect_raw or u.dialect].append(u)
        picked = []
        for key, items in sorted(by.items()):
            items = sorted(items, key=lambda u: u.id)
            picked.extend(rng.sample(items, min(args.per_dialect, len(items))))
        for u in picked:
            u.in_api_slice = True
        picked.sort(key=lambda u: u.id)
        write_manifest(out / "manifests" / f"{name}.jsonl", picked)
        hours = sum(u.duration_s for u in picked) / 3600
        print(f"{name}: {len(picked)} utts, {hours*60:.1f} min, groups={ {k: len(v) for k, v in by.items()} }")
    write_manifest_hash(out / "manifests")


def cmd_run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    models = PILOT_MODELS if args.model == "all" else [args.model]
    datasets = [p.stem for p in (out / "manifests").glob("*.jsonl")]
    for m in models:
        run_model(m, datasets, slice_name="api", limit=args.limit,
                  manifests_dir=out / "manifests", audio_dir=AUDIO, results_dir=out / "results")


def cmd_score(args: argparse.Namespace) -> None:
    out = Path(args.out)
    datasets = sorted(p.stem for p in (out / "manifests").glob("*.jsonl"))
    present = sorted(p.name for p in (out / "results").iterdir() if (p / "run.json").exists())
    scores = []
    for m in present:
        s = score_model(m, datasets, manifests_dir=out / "manifests", results_dir=out / "results",
                        slice_name="api", baseline_id="whisper_large_v3_turbo")
        scores.append(s)
    (out / "scores.json").write_text(json.dumps(scores, indent=2, ensure_ascii=False))
    md = ["# Pilot gate scores", "", f"Datasets: {datasets}", ""]
    for nm in ("elm", "tolerant"):
        md += [f"## WER % with `{nm}` normalizer (95% bootstrap CI)", "", markdown_table(scores, datasets, nm), ""]
    md += ["## Per-dialect WER % (elm)", ""]
    dialects = sorted({d for s in scores for d in s["dialects"]})
    md.append("| model | " + " | ".join(dialects) + " |")
    md.append("|---|" + "---|" * len(dialects))
    for s in scores:
        cells = []
        for d in dialects:
            c = s["dialects"].get(d, {}).get("elm")
            cells.append(f"{100*c['wer']['point']:.1f}" if c else "—")
        md.append(f"| {s['model_id']} | " + " | ".join(cells) + " |")
    (out / "pilot.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--dataset", action="append", default=None)
    b.add_argument("--per-dialect", type=int, default=19)
    b.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--model", default="all")
    r.add_argument("--limit", type=int, default=None)
    sub.add_parser("score")
    args = ap.parse_args()
    if args.cmd == "build":
        args.dataset = args.dataset or ["casablanca"]
        cmd_build(args)
    elif args.cmd == "run":
        cmd_run(args)
    else:
        cmd_score(args)


if __name__ == "__main__":
    main()

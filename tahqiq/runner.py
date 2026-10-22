"""Run one model over one or more manifests, caching predictions.

    uv run python -m tahqiq.runner --model whisper_large_v3_turbo --dataset casablanca
    uv run python -m tahqiq.runner --model deepgram_nova3 --slice api
    uv run python -m tahqiq.runner --model X --limit 5          # smoke test

Writes ``results/<model>/<dataset>.jsonl`` (resumable) and ``results/<model>/run.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from . import io as hio
from .manifest import Utterance, load_manifest, manifest_paths, read_manifest_hash
from .models import registry

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
AUDIO = ROOT / "audio"
RESULTS = ROOT / "results"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def select_utts(utts: list[Utterance], slice_name: str | None) -> list[Utterance]:
    if slice_name in (None, "full"):
        return utts
    if slice_name == "api":
        return [u for u in utts if u.in_api_slice]
    if slice_name == "timing":
        return [u for u in utts if u.in_timing_slice]
    raise ValueError(f"unknown slice {slice_name!r}")


def run_model(
    model_id: str,
    datasets: list[str],
    *,
    slice_name: str | None = None,
    limit: int | None = None,
    manifests_dir: Path = MANIFESTS,
    audio_dir: Path = AUDIO,
    results_dir: Path = RESULTS,
    hf_id: str | None = None,
    display_name: str | None = None,
    adapter: str | None = None,
    write_run_json: bool = True,
) -> dict:
    model = registry.build(model_id, hf_id=hf_id, display_name=display_name, adapter=adapter)
    run_id = uuid.uuid4().hex[:12]
    started = datetime.now(timezone.utc).isoformat()
    print(f"== {model_id} ({model.kind}) run_id={run_id}")
    model.load()
    totals = {"done": 0, "errors": 0, "skipped": 0, "audio_s": 0.0, "wall_s": 0.0}
    try:
        for ds in datasets:
            utts = select_utts(load_manifest(manifests_dir / f"{ds}.jsonl"), slice_name)
            cache = hio.load_predictions(results_dir, model_id, ds)
            todo = [u for u in utts if u.id not in cache or cache[u.id].error]
            if limit is not None:
                todo = todo[:limit]
            totals["skipped"] += len(utts) - len(todo)
            for u in tqdm(todo, desc=f"{model_id}/{ds}", unit="utt"):
                wav = audio_dir / u.audio_rel
                try:
                    t = model.transcribe_path(wav)
                    pred = hio.Prediction(u.id, t.text, t.wall_s, t.attempts, None, run_id)
                    totals["audio_s"] += u.duration_s
                    totals["wall_s"] += t.wall_s
                except Exception as e:  # noqa: BLE001
                    pred = hio.Prediction(u.id, "", 0.0, 1, f"{type(e).__name__}: {e}", run_id)
                    totals["errors"] += 1
                    traceback.print_exc(limit=1)
                hio.append_prediction(results_dir, model_id, ds, pred)
                totals["done"] += 1
    finally:
        model.unload()

    run = {
        "model_id": model_id,
        "display_name": model.display_name,
        "kind": model.kind,
        "hf_id": model.hf_id,
        "api_version": model.api_version,
        "language_hint": model.language_hint,
        "decode_params": model.decode_params,
        "run_id": run_id,
        "git_sha": _git_sha(),
        "manifest_sha256": read_manifest_hash(manifests_dir),
        "datasets": datasets,
        "slice": slice_name or "full",
        "env": {**model.env_info(), "python": platform.python_version(), "machine": platform.node()},
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "totals": totals,
    }
    out = results_dir / model_id / "run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Keep a history: one run.json (latest) plus runs/<run_id>.json.
    (out.parent / "runs").mkdir(exist_ok=True)
    (out.parent / "runs" / f"{run_id}.json").write_text(json.dumps(run, indent=2, ensure_ascii=False))
    if write_run_json:
        out.write_text(json.dumps(run, indent=2, ensure_ascii=False))
    print(json.dumps(totals))
    return run


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model_id or 'all'")
    ap.add_argument("--dataset", action="append", help="dataset name; repeatable; default all manifests")
    ap.add_argument("--slice", default=None, choices=[None, "full", "api", "timing"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--hf-id", default=None, help="run an arbitrary Hub model under --model <slug>")
    ap.add_argument("--display-name", default=None)
    ap.add_argument("--adapter", default=None, help="force adapter key (see models/auto.py)")
    ap.add_argument("--helper", action="store_true", help="fill the cache for some datasets without writing run.json")
    args = ap.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    datasets = args.dataset or [p.stem for p in manifest_paths(MANIFESTS)]
    models = registry.list_models() if args.model == "all" else [args.model]
    for m in models:
        run_model(m, datasets, slice_name=args.slice, limit=args.limit, hf_id=args.hf_id,
                  display_name=args.display_name, adapter=args.adapter, write_run_json=not args.helper)


if __name__ == "__main__":
    main()

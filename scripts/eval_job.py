"""Self-contained evaluation job: runs on any fresh GPU machine (HF Jobs, RunPod, a pod).

    python scripts/eval_job.py --slug openai_whisper-tiny --hf-id openai/whisper-tiny [--datasets sada,casablanca] [--no-timing] [--no-publish]

What it does, in order:
  1. fetch manifests + audio tars from the private audio dataset (HF_TOKEN) into ./data and ./audio
  2. run the batched runner for the requested datasets (resumable cache in ./results/<slug>/)
  3. timing pass on the fixed slice (the GPU is exclusive here, so the measurement is clean)
  4. score + upload results and scores (scripts/publish.py), rebuild board.json, mark the request DONE

Environment: HF_TOKEN (required), TAHQIQ_AUDIO_REPO (default HeshamHaroon/tahqiq-audio-private),
TAHQIQ_RESULTS_REPO / TAHQIQ_REQUESTS_REPO (defaults in tahqiq/hub.py).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

AUDIO_REPO = os.environ.get("TAHQIQ_AUDIO_REPO", "HeshamHaroon/tahqiq-audio-private")
DATASETS = ["casablanca", "commonvoice", "masc_clean", "masc_noisy", "sada"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_data(datasets: list[str]) -> None:
    token = os.environ["HF_TOKEN"]
    mdir = ROOT / "data" / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    snap = snapshot_download(AUDIO_REPO, repo_type="dataset", allow_patterns=["manifests/*"], token=token)
    for p in Path(snap, "manifests").iterdir():
        (mdir / p.name).write_bytes(p.read_bytes())
    log(f"manifests: {sorted(p.name for p in mdir.iterdir())}")
    audio = ROOT / "audio"
    audio.mkdir(exist_ok=True)
    for ds in datasets:
        if (audio / ds).exists() and any((audio / ds).iterdir()):
            continue
        t0 = time.time()
        tar_path = hf_hub_download(AUDIO_REPO, f"audio_tars/{ds}.tar", repo_type="dataset", token=token)
        with tarfile.open(tar_path) as tf:
            tf.extractall(audio, filter="data")
        log(f"audio/{ds}: {sum(1 for _ in (audio / ds).iterdir())} files in {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--hf-id", default=None, help="needed for non-registry models")
    ap.add_argument("--display-name", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--helper", action="store_true", help="partial run: write cache only, no run.json/publish")
    args = ap.parse_args()
    datasets = [d for d in args.datasets.split(",") if d]
    os.chdir(ROOT)
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    fetch_data(datasets)
    py = sys.executable
    cmd = [py, "-m", "tahqiq.runner", "--model", args.slug]
    for d in datasets:
        cmd += ["--dataset", d]
    if args.hf_id:
        cmd += ["--hf-id", args.hf_id]
    if args.display_name:
        cmd += ["--display-name", args.display_name]
    if args.adapter:
        cmd += ["--adapter", args.adapter]
    if args.batch:
        cmd += ["--batch", str(args.batch)]
    if args.helper:
        cmd += ["--helper"]
    log("run: " + " ".join(cmd[2:]))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"runner failed rc={rc}")
    if args.helper:
        log("helper mode: cache written, no publish")
        return
    if not args.no_timing:
        tcmd = [py, "scripts/timing.py", args.slug] + (["--hf-id", args.hf_id] if args.hf_id else []) \
            + (["--adapter", args.adapter] if args.adapter else []) + (["--display-name", args.display_name] if args.display_name else [])
        if subprocess.call(tcmd) != 0:
            log("timing pass failed; publishing accuracy only")
    if not args.no_publish:
        pcmd = [py, "scripts/publish.py", args.slug] + (["--hf-id", args.hf_id] if args.hf_id else []) \
            + (["--display-name", args.display_name] if args.display_name else [])
        if subprocess.call(pcmd) != 0:
            raise SystemExit("publish failed")
        from tahqiq import hub

        try:
            req = hub.get_request(args.slug)
            if req:
                doc = json.loads((ROOT / "results" / args.slug / "scores.json").read_text(encoding="utf-8"))
                errors = sum(v["n"] - v["scored"] for v in doc["scores"]["full"]["coverage"].values())
                req = hub.update_status(args.slug, "DONE", finished_at=hub.now(), worker="hf-jobs",
                                        error=(f"{errors} utterances errored" if errors else ""), results=hub.results_url(args.slug))
                full = doc["scores"]["full"]["datasets"]
                summary = ", ".join(f"{d} {100*v['elm']['wer']['point']:.1f}" for d, v in full.items() if "elm" in v)
                hub.notify(req, f"Done. WER % (elm normalizer): {summary}. Outputs: {hub.results_url(args.slug)}", close=True)
        finally:
            hub.rebuild_board()
    log("done")


if __name__ == "__main__":
    main()

"""Evaluation worker: polls the Hub request queue, runs each model, scores, publishes.

    nohup .venv/bin/python scripts/worker.py --poll 120 > logs/worker.log 2>&1 &
    .venv/bin/python scripts/worker.py --once          # one job then exit

Each job runs in a subprocess (``python -m harfiyyan.runner``) on the GPU with the
most free memory at that moment, in the main env or the NeMo env depending on the
adapter. Models that need remote code, exceed the size limit, or have an unknown
architecture are parked as NEEDS_REVIEW; set their status to APPROVED by hand to run.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from harfiyyan import hub  # noqa: E402
from harfiyyan.models.auto import detect  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from publish import build_scores  # noqa: E402

WORKER = f"{platform.node()}:{os.getpid()}"
PY_MAIN = ROOT / ".venv" / "bin" / "python"
PY_NEMO = ROOT / ".venv-nemo" / "bin" / "python"


def pick_gpu(exclude: set[int] = frozenset()) -> str:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
        rows = [(int(i), int(f)) for i, f in (l.split(",") for l in out.strip().splitlines())]
        rows = [r for r in rows if r[0] not in exclude]
        return str(max(rows, key=lambda r: r[1])[0])
    except Exception:
        return "0"


def log(msg: str) -> None:
    print(f"[{hub.now()}] {msg}", flush=True)


def run_job(req: dict, *, exclude_gpus: set[int]) -> None:
    slug, hf_id = req["slug"], req["hf_id"]
    dec = detect(hf_id, revision=req.get("revision"))
    if dec.needs_review and req["status"] != "APPROVED":
        park_for_review(req, dec.reasons, dec.adapter)
        log(f"{slug}: needs review ({dec.reasons})")
        return
    adapter = dec.adapter or req.get("adapter") or ""
    hub.update_status(slug, "RUNNING", adapter=adapter, worker=WORKER, started_at=hub.now(), error="")
    hub.rebuild_board()
    gpu = pick_gpu(exclude_gpus)
    py = PY_NEMO if dec.env == "nemo" else PY_MAIN
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
           "HARFIYYAN_AUDIO": os.environ.get("HARFIYYAN_AUDIO", str(ROOT / "audio"))}
    cmd = [str(py), "-m", "harfiyyan.runner", "--model", slug, "--hf-id", hf_id,
           "--display-name", req.get("display_name") or hf_id]
    if adapter and adapter != "nemo":
        cmd += ["--adapter", adapter]
    log(f"{slug}: running on GPU {gpu} with {py.name}: {' '.join(cmd[2:])}")
    logf = ROOT / "logs" / "runs" / f"{slug}.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    with open(logf, "ab") as f:
        rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        tail = logf.read_text(encoding="utf-8", errors="replace")[-1500:]
        req = hub.update_status(slug, "FAILED", error=f"runner exit {rc}\n{tail}", worker=WORKER)
        hub.notify(req, f"Run failed (exit {rc}). Last lines:\n```\n{tail[-800:]}\n```", close=True)
        log(f"{slug}: FAILED rc={rc}")
        return
    doc = build_scores(slug, extra={"hf_id": hf_id, "display_name": req.get("display_name"),
                                    "submitted_by": req.get("submitted_by") or "community"})
    errors = sum(v.get("n", 0) - v.get("scored", 0) for v in doc["scores"]["full"]["coverage"].values())
    hub.upload_results(slug, ROOT / "results" / slug, doc)
    req = hub.update_status(slug, "DONE", finished_at=hub.now(), worker=WORKER,
                            error=(f"{errors} utterances errored" if errors else ""), results=hub.results_url(slug))
    full = doc["scores"]["full"]["datasets"]
    summary = ", ".join(f"{d} {100*v['elm']['wer']['point']:.1f}" for d, v in full.items() if "elm" in v)
    hub.notify(req, f"Done. WER % (elm normalizer): {summary}. Outputs: {hub.results_url(slug)}", close=True)
    log(f"{slug}: DONE ({errors} errors)")


def park_for_review(req: dict, reasons: list[str], adapter: str) -> None:
    req = hub.update_status(req["slug"], "NEEDS_REVIEW", error="; ".join(reasons), adapter=adapter, worker=WORKER)
    hub.notify(req, "Parked for manual review: " + "; ".join(reasons) + ". A maintainer will approve or decline.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--exclude-gpus", default="0,3", help="GPU indexes never used (production jobs)")
    args = ap.parse_args()
    exclude = {int(x) for x in args.exclude_gpus.split(",") if x.strip()}
    log(f"worker {WORKER} up; polling {hub.REQUESTS_REPO} every {args.poll}s; excluding GPUs {sorted(exclude)}")
    while True:
        try:
            new = hub.poll_discussions()
            for r in new:
                log(f"queued from discussion: {r['slug']}")
            if new:
                hub.rebuild_board()
            reqs = hub.list_requests(force=True)
            todo = [r for r in reqs if r.get("status") in ("PENDING", "APPROVED")]
            todo.sort(key=lambda r: (r["status"] != "APPROVED", r.get("submitted_at", "")))
            if todo:
                run_job(todo[0], exclude_gpus=exclude)
                hub.rebuild_board()
                if args.once:
                    return
                continue
        except Exception:
            traceback.print_exc()
        if args.once:
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()

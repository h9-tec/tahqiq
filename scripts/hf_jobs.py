"""Run evaluations on Hugging Face Jobs (prepaid credits), nothing on our machines.

    uv run python scripts/hf_jobs.py bundle                       # upload current code as code/tahqiq.tar.gz to the private dataset
    uv run python scripts/hf_jobs.py launch --slug X --hf-id Y     # one GPU job for one model
    uv run python scripts/hf_jobs.py burst                         # all registry models, one job each (parallel)
    uv run python scripts/hf_jobs.py schedule                      # install the queue poller as a scheduled CPU job (every 10 min)
    uv run python scripts/hf_jobs.py poll                          # what the scheduled job runs: PENDING → launch job → RUNNING
    uv run python scripts/hf_jobs.py status                        # list jobs

Each GPU job: pulls the code bundle + private audio, installs deps, runs scripts/eval_job.py.
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from huggingface_hub import HfApi  # noqa: E402

from tahqiq import hub  # noqa: E402
from tahqiq.models import registry  # noqa: E402
from tahqiq.models.auto import detect  # noqa: E402

AUDIO_REPO = os.environ.get("TAHQIQ_AUDIO_REPO", "HeshamHaroon/tahqiq-audio-private")
IMAGE = os.environ.get("TAHQIQ_JOB_IMAGE", "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime")
FLAVOR = os.environ.get("TAHQIQ_JOB_FLAVOR", "l4x1")
TIMEOUT_S = int(os.environ.get("TAHQIQ_JOB_TIMEOUT", "7200"))

PIP_MAIN = ("pip install -q -U 'transformers>=5.13' accelerate soundfile scipy numpy rapidfuzz jiwer pyyaml tenacity httpx tqdm "
            "python-dotenv huggingface_hub sentencepiece protobuf tiktoken paramiko")
PIP_NEMO = "pip install -q 'nemo_toolkit[asr]' cython packaging"

BOOTSTRAP = """set -e
export HF_HUB_DISABLE_XET=1 HF_HOME=/tmp/hf
apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq libsndfile1 >/dev/null 2>&1 || true
python - <<'PY'
import os, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ["TAHQIQ_RESULTS_REPO"], "code/tahqiq.tar.gz", repo_type="dataset", token=os.environ["HF_TOKEN"])
tarfile.open(p).extractall("/work", filter="data")
PY
cd /work/tahqiq
{pip}
pip install -q -e . --no-deps
python scripts/eval_job.py {args}
"""


def _api() -> HfApi:
    return HfApi(token=os.environ["HF_TOKEN"])


def _audio_secrets() -> dict:
    """SFTP fallback for the audio (only when TAHQIQ_JOBS_USE_SSH=1); default is the private HF dataset."""
    key_path = os.environ.get("TAHQIQ_AUDIO_KEY_PATH")
    ssh = os.environ.get("TAHQIQ_AUDIO_SSH")
    if os.environ.get("TAHQIQ_JOBS_USE_SSH") != "1" or not (key_path and ssh):
        return {}
    return {"TAHQIQ_AUDIO_SSH": ssh, "TAHQIQ_AUDIO_KEY": Path(key_path).read_text(encoding="utf-8")}


def cmd_bundle() -> None:
    buf = io.BytesIO()
    files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in files:
            if f.startswith(("data/manifests", "data/elm", "results/", "space/")):
                continue
            tf.add(ROOT / f, arcname=f"tahqiq/{f}")
    buf.seek(0)
    _api().upload_file(path_or_fileobj=buf.getvalue(), path_in_repo="code/tahqiq.tar.gz", repo_id=hub.RESULTS_REPO,
                       repo_type="dataset", commit_message="code bundle")
    print(f"bundle uploaded ({len(buf.getvalue())/1e6:.1f} MB) to {hub.RESULTS_REPO}/code/tahqiq.tar.gz")


def launch(slug: str, hf_id: str | None, display_name: str | None, *, nemo: bool = False, extra_args: str = "",
           flavor: str | None = None) -> str:
    args = f"--slug {slug}"
    if hf_id:
        args += f" --hf-id {hf_id}"
    if display_name:
        args += f" --display-name '{display_name}'"
    args += f" {extra_args}"
    script = BOOTSTRAP.format(pip=(PIP_MAIN + " && " + PIP_NEMO) if nemo else PIP_MAIN, args=args)
    job = _api().run_job(
        image=IMAGE, command=["bash", "-lc", script], flavor=flavor or FLAVOR, timeout=TIMEOUT_S,
        env={"TAHQIQ_AUDIO_REPO": AUDIO_REPO, "TAHQIQ_RESULTS_REPO": hub.RESULTS_REPO, "TAHQIQ_REQUESTS_REPO": hub.REQUESTS_REPO,
             "TOKENIZERS_PARALLELISM": "false"},
        secrets={"HF_TOKEN": os.environ["HF_TOKEN"], **_audio_secrets()},
        name=f"tahqiq-{slug}"[:60],
    )
    print(f"launched {job.id} for {slug} on {flavor or FLAVOR}: https://huggingface.co/jobs/{job.id}")
    return job.id


def cmd_burst(flavor: str | None) -> None:
    for slug in registry.list_models():
        nemo = slug.startswith("nvidia_")
        launch(slug, None, None, nemo=nemo, flavor=flavor)


def cmd_poll() -> None:
    new = hub.poll_discussions()
    for r in new:
        print("queued:", r["slug"])
    for req in hub.list_requests(force=True):
        if req.get("status") not in ("PENDING", "APPROVED"):
            continue
        dec = detect(req["hf_id"], revision=req.get("revision"))
        if dec.needs_review and req["status"] != "APPROVED":
            r = hub.update_status(req["slug"], "NEEDS_REVIEW", error="; ".join(dec.reasons), adapter=dec.adapter, worker="hf-jobs")
            hub.notify(r, "Parked for manual review: " + "; ".join(dec.reasons) + ". A maintainer will approve or decline.")
            continue
        extra = f"--adapter {dec.adapter}" if dec.adapter and dec.adapter != "nemo" else ""
        jid = launch(req["slug"], req["hf_id"], req.get("display_name"), nemo=(dec.env == "nemo"), extra_args=extra)
        r = hub.update_status(req["slug"], "RUNNING", adapter=dec.adapter, worker=f"hf-jobs:{jid}", started_at=hub.now(), error="")
        hub.notify(r, f"Running on Hugging Face Jobs ({FLAVOR}). Results land here when done (typically 1 to 3 hours).")
    hub.rebuild_board()


def cmd_schedule() -> None:
    script = BOOTSTRAP.replace("python scripts/eval_job.py {args}", "python scripts/hf_jobs.py poll").format(pip="pip install -q huggingface_hub python-dotenv pyyaml", args="")
    sj = _api().create_scheduled_job(
        image="python:3.12-slim", command=["bash", "-lc", script], schedule="*/10 * * * *", flavor="cpu-basic", timeout=600,
        env={"TAHQIQ_AUDIO_REPO": AUDIO_REPO, "TAHQIQ_RESULTS_REPO": hub.RESULTS_REPO, "TAHQIQ_REQUESTS_REPO": hub.REQUESTS_REPO,
             "TAHQIQ_JOB_FLAVOR": FLAVOR},
        secrets={"HF_TOKEN": os.environ["HF_TOKEN"]},
    )
    print("scheduled poller:", sj.id)


def cmd_status() -> None:
    for j in _api().list_jobs():
        st = getattr(j, "status", None)
        print(j.id, getattr(j, "flavor", ""), getattr(st, "stage", st), getattr(j, "created_at", ""), getattr(j, "command", [""])[-1][:40] if getattr(j, "command", None) else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bundle")
    l = sub.add_parser("launch"); l.add_argument("--slug", required=True); l.add_argument("--hf-id"); l.add_argument("--display-name"); l.add_argument("--nemo", action="store_true"); l.add_argument("--flavor")
    b = sub.add_parser("burst"); b.add_argument("--flavor")
    sub.add_parser("poll"); sub.add_parser("schedule"); sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "bundle":
        cmd_bundle()
    elif a.cmd == "launch":
        launch(a.slug, a.hf_id, a.display_name, nemo=a.nemo, flavor=a.flavor)
    elif a.cmd == "burst":
        cmd_burst(a.flavor)
    elif a.cmd == "poll":
        cmd_poll()
    elif a.cmd == "schedule":
        cmd_schedule()
    else:
        cmd_status()


if __name__ == "__main__":
    main()

"""Tahqiq worker Space: evaluates community submissions on ZeroGPU (included with PRO, no pay-per-use).

Flow: a background thread polls the request queue every few minutes; for each PENDING model it calls
this Space's own `evaluate` endpoint through gradio_client (ZeroGPU functions need a Gradio request
context), which runs the model on the COMMON SLICE (about 1,000 utterances per test set, ~9 h of
audio) in one ZeroGPU call per dataset, then scores, measures timing and publishes to the results
dataset. Full-set rows stay maintainer-only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from pathlib import Path

import spaces  # noqa: F401  (must be imported before torch)

ROOT = Path("/home/user/app") if Path("/home/user/app").exists() else Path(__file__).resolve().parent
CODE = ROOT / "code" / "tahqiq"
AUDIO_REPO = os.environ.get("TAHQIQ_AUDIO_REPO", "HeshamHaroon/tahqiq-audio-private")
POLL_S = int(os.environ.get("TAHQIQ_POLL_S", "300"))
LOG: list[str] = []
STATE = {"current": None, "last_poll": None, "ready": False, "error": None}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)
    del LOG[:-400]


# ---------- bootstrap: code bundle + common-slice audio ----------

def bootstrap() -> None:
    from huggingface_hub import hf_hub_download

    token = os.environ["HF_TOKEN"]
    results_repo = os.environ.get("TAHQIQ_RESULTS_REPO", "HeshamHaroon/tahqiq-results")
    bundle = hf_hub_download(results_repo, "code/tahqiq.tar.gz", repo_type="dataset", token=token, force_download=True)
    (ROOT / "code").mkdir(exist_ok=True)
    with tarfile.open(bundle) as tf:
        tf.extractall(ROOT / "code", filter="data")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(CODE), "--no-deps"])
    sys.path.insert(0, str(CODE))
    log("code bundle installed")
    from tahqiq.manifest import load_manifest

    mdir = CODE / "data" / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snap = snapshot_download(AUDIO_REPO, repo_type="dataset", allow_patterns=["manifests/*"], token=token)
    for p in Path(snap, "manifests").iterdir():
        (mdir / p.name).write_bytes(p.read_bytes())
    audio = CODE / "audio"
    audio.mkdir(exist_ok=True)
    for mf in sorted(mdir.glob("*.jsonl")):
        ds = mf.stem
        wanted = {u.audio_rel for u in load_manifest(mf) if u.in_api_slice}
        if all((audio / rel).exists() for rel in wanted):
            continue
        t0 = time.time()
        tar_path = hf_hub_download(AUDIO_REPO, f"audio_tars/{ds}.tar", repo_type="dataset", token=token)
        with tarfile.open(tar_path) as tf:
            members = [m for m in tf.getmembers() if m.name in wanted or m.name.lstrip("./") in wanted]
            tf.extractall(audio, members=members, filter="data")
        log(f"audio/{ds}: {len(wanted)} common-slice files in {time.time() - t0:.0f}s")
    STATE["ready"] = True
    log("ready")


# ---------- evaluation (ZeroGPU) ----------

def _gpu_run(slug: str, hf_id: str, display_name: str, adapter: str | None, ds: str) -> str:
    """Runs inside a ZeroGPU allocation: one dataset of the common slice."""
    from tahqiq.runner import run_model

    run = run_model(slug, [ds], slice_name="api", manifests_dir=CODE / "data" / "manifests", audio_dir=CODE / "audio",
                    results_dir=CODE / "results", hf_id=hf_id, display_name=display_name, adapter=adapter, write_run_json=False)
    return json.dumps(run["totals"])


_gpu_run = spaces.GPU(duration=int(os.environ.get("TAHQIQ_GPU_SECONDS", "480")))(_gpu_run)


def _gpu_timing(slug: str, hf_id: str, adapter: str | None) -> str:
    cmd = [sys.executable, str(CODE / "scripts" / "timing.py"), slug, "--hf-id", hf_id] + (["--adapter", adapter] if adapter else [])
    rc = subprocess.call(cmd, cwd=CODE)
    return f"timing rc={rc}"


_gpu_timing = spaces.GPU(duration=300)(_gpu_timing)


def evaluate(slug: str) -> str:
    """Gradio endpoint: evaluate one queued request end to end. Called by the poller through gradio_client."""
    from tahqiq import hub
    from tahqiq.manifest import manifest_paths
    from tahqiq.models.auto import detect

    req = hub.get_request(slug)
    if not req:
        return f"no request {slug}"
    STATE["current"] = slug
    try:
        dec = detect(req["hf_id"], revision=req.get("revision"))
        if dec.env == "nemo":
            dec.needs_review = True
            dec.reasons.append("NeMo checkpoints are not supported on the ZeroGPU worker")
        if dec.needs_review and req["status"] != "APPROVED":
            r = hub.update_status(slug, "NEEDS_REVIEW", error="; ".join(dec.reasons), adapter=dec.adapter, worker="zerogpu")
            hub.notify(r, "Parked for manual review: " + "; ".join(dec.reasons))
            return "needs review"
        hub.update_status(slug, "RUNNING", adapter=dec.adapter, worker="zerogpu", started_at=hub.now(), error="")
        hub.rebuild_board()
        datasets = [p.stem for p in manifest_paths(CODE / "data" / "manifests")]
        adapter = dec.adapter if dec.adapter and dec.adapter != "nemo" else None
        for ds in datasets:
            log(f"{slug}: {ds}")
            totals = _gpu_run(slug, req["hf_id"], req.get("display_name") or req["hf_id"], adapter, ds)
            log(f"{slug}: {ds} {totals}")
        # one run.json for the whole common slice
        runs = sorted((CODE / "results" / slug / "runs").glob("*.json"))
        base = json.loads(runs[-1].read_text(encoding="utf-8"))
        base.update({"datasets": datasets, "slice": "api", "worker": "zerogpu"})
        (CODE / "results" / slug / "run.json").write_text(json.dumps(base, indent=2, ensure_ascii=False), encoding="utf-8")
        log(_gpu_timing(slug, req["hf_id"], adapter))
        rc = subprocess.call([sys.executable, str(CODE / "scripts" / "publish.py"), slug, "--hf-id", req["hf_id"],
                              "--display-name", req.get("display_name") or req["hf_id"], "--slices", "api"], cwd=CODE)
        if rc != 0:
            raise RuntimeError(f"publish rc={rc}")
        doc = json.loads((CODE / "results" / slug / "scores.json").read_text(encoding="utf-8"))
        errors = sum(v["n"] - v["scored"] for v in doc["scores"]["api"]["coverage"].values())
        r = hub.update_status(slug, "DONE", finished_at=hub.now(), worker="zerogpu", results=hub.results_url(slug),
                              error=(f"{errors} utterances errored" if errors else ""), scope="common slice")
        summary = ", ".join(f"{d} {100*v['elm']['wer']['point']:.1f}" for d, v in doc["scores"]["api"]["datasets"].items() if "elm" in v)
        hub.notify(r, f"Done on the common slice. WER % (elm normalizer): {summary}. Outputs: {hub.results_url(slug)}", close=True)
        hub.rebuild_board()
        return "done"
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()[-1200:]
        log(f"{slug}: FAILED {e}")
        try:
            r = hub.update_status(slug, "FAILED", error=f"{type(e).__name__}: {e}", worker="zerogpu")
            hub.notify(r, f"Run failed: {type(e).__name__}: {e}\n```\n{tb}\n```", close=True)
            hub.rebuild_board()
        except Exception:
            pass
        return f"failed: {e}"
    finally:
        STATE["current"] = None


# ---------- poller ----------

def poll_once() -> str:
    from tahqiq import hub

    new = hub.poll_discussions()
    for r in new:
        log(f"queued from discussion: {r['slug']}")
    if new:
        hub.rebuild_board()
    todo = [r for r in hub.list_requests(force=True) if r.get("status") in ("PENDING", "APPROVED")]
    todo.sort(key=lambda r: (r["status"] != "APPROVED", r.get("submitted_at", "")))
    STATE["last_poll"] = time.strftime("%H:%M:%S")
    if not todo or STATE["current"]:
        return f"{len(todo)} waiting; current={STATE['current']}"
    slug = todo[0]["slug"]
    from gradio_client import Client

    client = Client(os.environ.get("SPACE_ID", "HeshamHaroon/tahqiq-worker"), token=os.environ["HF_TOKEN"], verbose=False)
    log(f"launching {slug}")
    out = client.predict(slug, api_name="/evaluate")
    return str(out)


def poller() -> None:
    while not STATE["ready"]:
        time.sleep(5)
    time.sleep(20)
    while True:
        try:
            log("poll: " + poll_once())
        except Exception as e:  # noqa: BLE001
            log(f"poll error: {e}")
        time.sleep(POLL_S)


def _boot() -> None:
    try:
        bootstrap()
    except Exception as e:  # noqa: BLE001
        STATE["error"] = f"{type(e).__name__}: {e}"
        log("bootstrap failed: " + STATE["error"])


threading.Thread(target=_boot, daemon=True).start()
threading.Thread(target=poller, daemon=True).start()

# ---------- UI ----------
import gradio as gr  # noqa: E402

with gr.Blocks(title="Tahqiq worker") as demo:
    gr.Markdown("# Tahqiq worker\nEvaluates queued submissions on the common slice using ZeroGPU. Submit models at "
                "[the leaderboard Space](https://huggingface.co/spaces/HeshamHaroon/tahqiq).")
    status = gr.Textbox(label="status", interactive=False)
    logbox = gr.Textbox(label="log", lines=18, interactive=False)
    with gr.Row():
        refresh = gr.Button("Refresh")
        poll_btn = gr.Button("Poll queue now")
    slug_in = gr.Textbox(label="slug (maintainer use)", visible=True)
    eval_btn = gr.Button("Evaluate slug", visible=True)
    eval_out = gr.Textbox(label="result", interactive=False)

    def _status():
        return json.dumps(STATE), "\n".join(LOG[-60:])

    refresh.click(_status, outputs=[status, logbox])
    poll_btn.click(lambda: poll_once(), outputs=[eval_out]).then(_status, outputs=[status, logbox])
    eval_btn.click(evaluate, inputs=[slug_in], outputs=[eval_out], api_name="evaluate")
    demo.load(_status, outputs=[status, logbox])

demo.queue(default_concurrency_limit=1).launch()

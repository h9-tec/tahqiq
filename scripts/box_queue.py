"""Queue runner for the H20 box: one model per GPU at a time, accuracy run + latency pass on the same GPU.

    python scripts/box_queue.py plan   [--priority 1,2,3]
    python scripts/box_queue.py run    --gpus 0,1,2,3,4,5,6,7 [--priority 1,2,3] [--retime-board] [--retry]
    python scripts/box_queue.py status

Rules (so the Production numbers are comparable):
  * one worker thread per GPU, and a worker runs exactly one model at a time; nothing else from Tahqiq
    touches that GPU (TAHQIQ_EXCLUSIVE_GPU=1 is recorded in timing.json);
  * every eval unit = runner (accuracy, resumable cache) -> scripts/timing.py on the same GPU
    (waits for other tenants to go idle, samples their SM% during the passes) -> scripts/publish.py;
  * --retime-board adds a timing-only unit for every model already on the board whose timing.json was
    not measured on this box's GPU type, so every model on the board is timed on the same hardware.

Units come from data/candidates.yaml. Slow LLM-style models deferred by the user (Gemma, Voxtral Realtime,
omniASR) and models without an adapter are SKIPPED with a reason. A unit is claimed only by a GPU whose free
memory fits it (1.3 x weights + 10 GB), so large models wait for a roomy GPU instead of crashing.
State: logs/box_queue.json (file-locked; resumable). Logs: logs/box/<slug>.log.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

STATE = ROOT / "logs" / "box_queue.json"
LOGS = ROOT / "logs" / "box"
UV = os.environ.get("TAHQIQ_UV", "/root/.local/bin/uv")
DEFERRED = ("google_gemma-4", "Voxtral-Mini-4B-Realtime", "omniASR")   # user: slow LLM-type models later
WAIT_QUIET_S = 1200
# extra packages run in a uv overlay on the project env; pin torch to the project's build or uv pulls a newer
# one built for a CUDA the box driver does not support
TORCH_PIN = os.environ.get("TAHQIQ_TORCH_PIN", "torch==2.13.0+cu129")
TORCH_INDEX = os.environ.get("TAHQIQ_TORCH_INDEX", "https://download.pytorch.org/whl/cu129")


# ---------------------------------------------------------------- state (shared, file-locked)
def _load() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def _save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE)


class _FileLock:
    def __enter__(self):
        import fcntl
        STATE.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(STATE.with_suffix(".lock"), "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *a):
        import fcntl
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def _update(key: str, **fields) -> None:
    with _FileLock():
        st = _load()
        st.setdefault(key, {}).update(fields)
        _save(st)


def gpu_free_gb(gpu: int) -> float:
    out = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return float(out or 0) / 1024


def need_gb(u: dict) -> float:
    w = float(u.get("weights_gb") or 0) or 6.0   # registry slugs carry no size; assume a mid-size model
    return 1.3 * w + 10


def _claim(queue: list[dict], gpu: int) -> tuple[dict | None, bool]:
    """Take the first PENDING unit that fits this GPU's free memory. Returns (unit, any_pending_left)."""
    free = gpu_free_gb(gpu)
    with _FileLock():
        st = _load()
        pending = [u for u in queue if st.get(u["key"], {}).get("status") == "PENDING"]
        for u in pending:
            if need_gb(u) <= free:
                st[u["key"]].update({"status": "RUNNING", "gpu": gpu, "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
                _save(st)
                return u, True
        return None, bool(pending)


# ---------------------------------------------------------------- units
def _cands() -> dict:
    import yaml
    c = (yaml.safe_load((ROOT / "data" / "candidates.yaml").read_text(encoding="utf-8")) or {}).get("models") or []
    return {x["hf_id"]: x for x in c if x.get("hf_id")}


def board_models() -> list[dict]:
    from huggingface_hub import hf_hub_download

    from tahqiq import hub
    p = hf_hub_download(hub.RESULTS_REPO, "board.json", repo_type="dataset", token=os.environ.get("HF_TOKEN"), force_download=True)
    return json.loads(Path(p).read_text(encoding="utf-8")).get("models") or []


def build_units(priorities: set[int], max_weight_gb: float, on_board: set[str]) -> tuple[list[dict], list[tuple]]:
    from runpod_burst import units_from_candidates

    meta = _cands()
    seen, units, skipped = set(), [], []
    for u in units_from_candidates(ROOT / "data" / "candidates.yaml", max_priority=max(priorities), split_over_gb=10_000):
        slug = u["slug"]
        if slug in seen:
            continue
        seen.add(slug)
        hf = u.get("hf_id")
        if not hf:   # registry slugs carry no hf_id in the unit; look it up for size/priority
            try:
                from tahqiq.models import registry
                hf = registry.build(slug).hf_id
            except Exception:
                hf = None
        c = meta.get(hf, {})
        pri = int(c.get("priority", 1)) if c else 1
        if pri not in priorities or slug in on_board:
            continue
        w = float(c.get("weights_gb") or 0)
        if c.get("duplicate_of"):
            skipped.append((slug, f"same weights as {c['duplicate_of']}, already on the board")); continue
        if any(d in slug for d in DEFERRED):
            skipped.append((slug, "deferred by user (slow LLM-type model)")); continue
        if u.get("omni"):
            skipped.append((slug, "omnilingual-asr env not installed on the box")); continue
        if u.get("remote_code") and not u.get("adapter"):
            skipped.append((slug, "custom code without a dedicated adapter")); continue
        if w > max_weight_gb:
            skipped.append((slug, f"weights {w} GB > {max_weight_gb} GB")); continue
        units.append({**u, "key": slug, "kind": "eval", "weights_gb": w, "priority": pri, "api_first": bool(c.get("api_first"))})
    # long jobs first (priority-1 leftovers are the big/slow ones), then small to large
    units.sort(key=lambda u: (u["priority"] != 1, u["weights_gb"]))
    return units, skipped


def retime_units(models: list[dict], gpu_name: str) -> list[dict]:
    meta = _cands()
    out = []
    for m in models:
        t = m.get("timing") or {}
        if gpu_name and gpu_name in str(t.get("gpu") or "") and t.get("exclusive_gpu"):
            continue
        slug = m["model_id"]
        c = meta.get(m.get("hf_id"), {})
        u = {"slug": slug, "key": f"timing:{slug}", "kind": "timing", "hf_id": m.get("hf_id"), "display_name": m.get("display_name"),
             "weights_gb": float(c.get("weights_gb") or 0), "nemo": bool(c.get("nemo")) or slug.startswith("nvidia_"),
             "adapter": c.get("adapter"), "transformers_spec": c.get("transformers_spec"), "pip": c.get("pip")}
        out.append(u)
    return out


def _py(u: dict) -> str:
    if u.get("nemo"):
        return ".venv-nemo/bin/python"
    extra = ([f"transformers{u['transformers_spec']}"] if u.get("transformers_spec") else []) + list(u.get("pip") or [])
    if extra:
        withs = " ".join(f"--with {shlex.quote(x)}" for x in [*extra, "librosa", TORCH_PIN])
        return f"{UV} run --no-sync {withs} --extra-index-url {TORCH_INDEX} --index-strategy unsafe-best-match python"
    return ".venv/bin/python"


def unit_cmds(u: dict) -> list[tuple[str, str, bool]]:
    """[(label, command, required)]; a non-required step may fail without failing the unit."""
    slug = u["slug"]
    ident = ""
    if u.get("hf_id"):
        ident += f" --hf-id {u['hf_id']}"
    if u.get("display_name"):
        ident += f" --display-name {shlex.quote(u['display_name'])}"
    adapter = f" --adapter {u['adapter']}" if u.get("adapter") else ""
    py = _py(u)
    steps = []
    if u["kind"] == "eval" and u.get("api_first"):
        # slow models: score the common slice first so the row appears within hours; the full run resumes from its cache
        steps.append(("run-api", f"{py} -m tahqiq.runner --model {slug}{ident}{adapter} --slice api", True))
        steps.append(("publish-api", f".venv/bin/python scripts/publish.py {slug}{ident} --slices api", True))
    if u["kind"] == "eval":
        steps.append(("run", f"{py} -m tahqiq.runner --model {slug}{ident}{adapter}" + (" --batch 1" if u.get("nemo") else ""), True))
    steps.append(("timing", f"{py} scripts/timing.py {slug}{ident}{adapter} --wait-quiet {WAIT_QUIET_S}", u["kind"] == "timing"))
    steps.append(("publish", f".venv/bin/python scripts/publish.py {slug}{ident}", True))
    return steps


# ---------------------------------------------------------------- worker
def worker(gpu: int, queue: list[dict]) -> None:
    while True:
        unit, left = _claim(queue, gpu)
        if unit is None:
            if not left:
                return
            time.sleep(60)   # pending units exist but none fits this GPU right now
            continue
        log = LOGS / f"{unit['slug']}.log"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TAHQIQ_PHYS_GPU=str(gpu), TAHQIQ_EXCLUSIVE_GPU="1",
                   HF_HUB_DISABLE_XET="1", TOKENIZERS_PARALLELISM="false", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", PYTHONPATH=str(ROOT))
        env.pop("TAHQIQ_TRUST_REMOTE_CODE", None)
        print(f"[gpu{gpu}] start {unit['kind']} {unit['slug']} ({unit['weights_gb']} GB)", flush=True)
        t0, rc, failed_step = time.time(), 0, None
        with open(log, "ab") as f:
            for label, cmd, required in unit_cmds(unit):
                f.write(f"\n==== {time.strftime('%F %T')} gpu{gpu} {label}\n{cmd}\n".encode()); f.flush()
                r = subprocess.call(cmd, shell=True, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
                if r != 0 and required:
                    rc, failed_step = r, label
                    break
                if r != 0:
                    f.write(f"\n==== {label} failed rc={r}; continuing without it\n".encode())
        status = "DONE" if rc == 0 else f"FAILED {failed_step} rc={rc}"
        _update(unit["key"], status=status, ended=time.strftime("%Y-%m-%dT%H:%M:%S"), minutes=round((time.time() - t0) / 60, 1))
        print(f"[gpu{gpu}] {unit['slug']}: {status} in {round((time.time() - t0) / 60)} min", flush=True)


# ---------------------------------------------------------------- cli
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "status"])
    ap.add_argument("--priority", default="1,2,3")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--max-weight-gb", type=float, default=60.0)
    ap.add_argument("--only", default=None)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--retry", action="store_true", help="re-run FAILED units too")
    ap.add_argument("--retime-board", action="store_true", help="timing-only units for board models not timed on this GPU type")
    ap.add_argument("--gpu-name", default="H20")
    a = ap.parse_args()
    if a.cmd == "status":
        from collections import Counter
        st = _load()
        print(Counter(v["status"].split(" ")[0] for v in st.values()))
        for k, v in st.items():
            if v["status"] not in ("PENDING", "SKIPPED"):
                print(f"  {v['status']:<22} gpu{v.get('gpu', '-')}  {k}  {v.get('minutes', '')}")
        return
    models = board_models()
    on_board = {m["model_id"] for m in models}
    units, skipped = build_units({int(p) for p in a.priority.split(",")}, a.max_weight_gb, on_board)
    if a.retime_board:
        units = retime_units(models, a.gpu_name) + units
    if a.only:
        units = [u for u in units if u["slug"] in a.only.split(",")]
    if a.exclude:
        units = [u for u in units if u["slug"] not in a.exclude.split(",")]
    st = _load()
    queue = [u for u in units if not (st.get(u["key"], {}).get("status") == "DONE"
                                     or (st.get(u["key"], {}).get("status", "").startswith("FAILED") and not a.retry))]
    if a.cmd == "plan":
        print(f"{len(queue)} to run ({sum(u['kind'] == 'timing' for u in queue)} timing-only), "
              f"{len(units) - len(queue)} done/failed, {len(skipped)} skipped")
        for u in queue:
            print(f"  {u['kind']:<6} {u['weights_gb']:>5} GB  need {need_gb(u):>5.1f}  {u['slug']}")
        for s, why in skipped:
            print(f"  skip {s}: {why}")
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    with _FileLock():
        st = _load()
        for u in queue:
            st[u["key"]] = {"status": "PENDING", "kind": u["kind"], "hf_id": u.get("hf_id"), "weights_gb": u["weights_gb"]}
        for s, why in skipped:
            st.setdefault(s, {"status": "SKIPPED", "why": why})
        _save(st)
    gpus = [int(g) for g in a.gpus.split(",")]
    print(f"{len(queue)} units on GPUs {gpus}, one model per GPU", flush=True)
    threads = [threading.Thread(target=worker, args=(g, queue), daemon=True) for g in gpus]
    for t in threads:
        t.start(); time.sleep(3)
    for t in threads:
        t.join()
    from collections import Counter
    print("BOX_QUEUE_DONE", dict(Counter(v["status"].split(" ")[0] for v in _load().values())), flush=True)


if __name__ == "__main__":
    main()

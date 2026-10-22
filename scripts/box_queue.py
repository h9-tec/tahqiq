"""Queue runner for the shared H20 box: evaluates many Hub models, one worker per GPU, publishing each when done.

    python scripts/box_queue.py plan  [--priority 2,3]              # list what would run
    python scripts/box_queue.py run   --gpus 1,2,4,5 [--priority 2,3] [--only a,b] [--exclude c]
    python scripts/box_queue.py status

Units come from data/candidates.yaml (same builder as the RunPod burst, but one unit per slug and no
timing pass: the box is shared, so RTFx/latency are measured elsewhere). State in logs/box_queue.json
(resumable: finished slugs are skipped; FAILED ones are retried only with --retry). Models that need
custom code, an unsupported architecture, or the omnilingual-asr env are listed as SKIPPED with the reason.
Each unit: `python -m tahqiq.runner --model <slug> ...` then `scripts/publish.py <slug> ...`, log in logs/box/<slug>.log.
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
LOCK = threading.Lock()


def _load() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def _save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE)


def build_units(priorities: set[int], max_weight_gb: float) -> tuple[list[dict], list[tuple]]:
    import yaml
    from runpod_burst import units_from_candidates

    cands = (yaml.safe_load((ROOT / "data" / "candidates.yaml").read_text(encoding="utf-8")) or {}).get("models") or []
    meta = {c["hf_id"]: c for c in cands if c.get("hf_id")}
    hi = max(priorities)
    units_hi = units_from_candidates(ROOT / "data" / "candidates.yaml", max_priority=hi, split_over_gb=10_000)
    lo_slugs = set()
    for p in range(1, min(priorities)):
        lo_slugs |= {u["slug"] for u in units_from_candidates(ROOT / "data" / "candidates.yaml", max_priority=p, split_over_gb=10_000)}
    units, skipped = [], []
    for u in units_hi:
        if u["slug"] in lo_slugs:
            continue
        c = meta.get(u.get("hf_id"), {})
        w = float(c.get("weights_gb") or 0)
        if u.get("omni"):
            skipped.append((u["slug"], "omnilingual-asr env not installed on the box")); continue
        if u.get("remote_code") and not u.get("adapter"):
            skipped.append((u["slug"], "custom code without a dedicated adapter")); continue
        if w > max_weight_gb:
            skipped.append((u["slug"], f"weights {w} GB > {max_weight_gb} GB")); continue
        u["weights_gb"] = w
        u["timing"] = False
        units.append(u)
    units.sort(key=lambda u: u["weights_gb"])
    return units, skipped


def unit_cmds(u: dict) -> tuple[str, str]:
    slug = u["slug"]
    extra = ""
    if u.get("hf_id"):
        extra += f" --hf-id {u['hf_id']}"
    if u.get("display_name"):
        extra += f" --display-name {shlex.quote(u['display_name'])}"
    run_extra = extra + (f" --adapter {u['adapter']}" if u.get("adapter") else "") + (" --batch 1" if u.get("nemo") else "")
    if u.get("nemo"):
        py = ".venv-nemo/bin/python"
    elif u.get("transformers_spec"):
        py = f"{UV} run --no-sync --with 'transformers{u['transformers_spec']}' --with librosa python"
    else:
        py = ".venv/bin/python"
    run = f"{py} -m tahqiq.runner --model {slug}{run_extra}"
    pub = f".venv/bin/python scripts/publish.py {slug}{extra}"
    return run, pub


def gpu_free_gb(gpu: int) -> float:
    out = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return float(out or 0) / 1024


def worker(gpu: int, queue: list[dict], state: dict) -> None:
    while True:
        with LOCK:
            unit = next((u for u in queue if state.get(u["slug"], {}).get("status") == "PENDING"), None)
            if unit is None:
                return
            state[unit["slug"]].update({"status": "WAITING_GPU", "gpu": gpu}); _save(state)
        need = 2.5 * unit["weights_gb"] + 6
        waited = 0
        while gpu_free_gb(gpu) < need and waited < 3600:
            time.sleep(60); waited += 60
        run, pub = unit_cmds(unit)
        log = LOGS / f"{unit['slug']}.log"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_DISABLE_XET="1", TOKENIZERS_PARALLELISM="false",
                   TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", PYTHONPATH=str(ROOT))
        env.pop("TAHQIQ_TRUST_REMOTE_CODE", None)
        with LOCK:
            state[unit["slug"]].update({"status": "RUNNING", "started": time.strftime("%Y-%m-%dT%H:%M:%S")}); _save(state)
        print(f"[gpu{gpu}] start {unit['slug']} ({unit['weights_gb']} GB)", flush=True)
        t0 = time.time()
        with open(log, "ab") as f:
            f.write(f"\n==== {time.strftime('%F %T')} gpu{gpu}\n{run}\n".encode())
            f.flush()
            rc = subprocess.call(run, shell=True, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            if rc == 0:
                f.write(f"\n==== publish\n{pub}\n".encode()); f.flush()
                rc = subprocess.call(pub, shell=True, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
                if rc != 0:
                    rc = 1000 + rc  # distinguishes a publish failure from a runner failure
        with LOCK:
            state[unit["slug"]].update({"status": "DONE" if rc == 0 else f"FAILED rc={rc}", "rc": rc,
                                        "ended": time.strftime("%Y-%m-%dT%H:%M:%S"), "minutes": round((time.time() - t0) / 60, 1)})
            _save(state)
        print(f"[gpu{gpu}] {unit['slug']}: {'done' if rc == 0 else 'FAILED rc=' + str(rc)} in {round((time.time() - t0) / 60)} min", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "status"])
    ap.add_argument("--priority", default="2,3")
    ap.add_argument("--gpus", default="1,2,4,5")
    ap.add_argument("--max-weight-gb", type=float, default=60.0)
    ap.add_argument("--only", default=None)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--retry", action="store_true", help="re-run FAILED units too")
    a = ap.parse_args()
    state = _load()
    if a.cmd == "status":
        from collections import Counter
        print(Counter(v["status"].split(" ")[0] for v in state.values()))
        for k, v in state.items():
            if v["status"] not in ("PENDING",):
                print(f"  {v['status']:<16} gpu{v.get('gpu', '-')}  {k}  {v.get('minutes', '')}")
        return
    pri = {int(p) for p in a.priority.split(",")}
    units, skipped = build_units(pri, a.max_weight_gb)
    if a.only:
        units = [u for u in units if u["slug"] in a.only.split(",")]
    if a.exclude:
        units = [u for u in units if u["slug"] not in a.exclude.split(",")]
    done = {k for k, v in state.items() if v.get("status") == "DONE"} | {p.parent.name for p in (ROOT / "results").glob("*/.published")}
    queue = []
    for u in units:
        s = state.get(u["slug"], {}).get("status", "")
        if u["slug"] in done or (s.startswith("FAILED") and not a.retry):
            continue
        queue.append(u)
    if a.cmd == "plan":
        print(f"{len(queue)} to run, {len(units) - len(queue)} already done/failed, {len(skipped)} skipped")
        for u in queue:
            print(f"  {u['weights_gb']:>5} GB  {'nemo ' if u.get('nemo') else ''}{u.get('adapter') or ''} {u['slug']}  <- {u.get('hf_id', 'registry')}")
        for s, why in skipped:
            print(f"  skip {s}: {why}")
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    for u in queue:
        state[u["slug"]] = {"status": "PENDING", "hf_id": u.get("hf_id"), "weights_gb": u["weights_gb"]}
    for s, why in skipped:
        state.setdefault(s, {"status": "SKIPPED", "why": why})
    _save(state)
    gpus = [int(g) for g in a.gpus.split(",")]
    print(f"{len(queue)} units on GPUs {gpus}", flush=True)
    threads = [threading.Thread(target=worker, args=(g, queue, state), daemon=True) for g in gpus]
    for t in threads:
        t.start(); time.sleep(5)
    for t in threads:
        t.join()
    from collections import Counter
    print("BOX_QUEUE_DONE", dict(Counter(v["status"].split(" ")[0] for v in state.values())), flush=True)


if __name__ == "__main__":
    main()

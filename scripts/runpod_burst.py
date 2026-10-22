"""Burst evaluation on RunPod: one pod per work unit, everything in parallel, then pull results back.

    uv run python scripts/runpod_burst.py plan                  # show units and pod count
    uv run python scripts/runpod_burst.py run  [--max-pods 16]  # create pods, run, collect, terminate
    uv run python scripts/runpod_burst.py collect               # re-pull results from pods that are still up
    uv run python scripts/runpod_burst.py kill                  # terminate every tahqiq-* pod

A unit = (model, datasets[, timing]). Each pod: image runpod/pytorch, bootstrap = pip deps + code bundle
(from the public results dataset) + audio pulled from the box over the restricted SFTP key, then
scripts/eval_job.py --helper for its datasets; the unit flagged `timing` also runs scripts/timing.py
at the end on the then-idle GPU. Results come back by rsync; scripts/assemble_run.py + publish.py finish it.

Needs in .env: HF_TOKEN, TAHQIQ_AUDIO_SSH, TAHQIQ_AUDIO_KEY_PATH; RunPod key in ~/.runpod/config.toml.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from tahqiq import hub  # noqa: E402

API = "https://api.runpod.io/graphql"
POD_KEY = Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go"
IMAGE = os.environ.get("TAHQIQ_POD_IMAGE", "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04")
GPU_PREFS = [g for g in os.environ.get("TAHQIQ_POD_GPUS", "NVIDIA H200,NVIDIA H100 NVL,NVIDIA H100 80GB HBM3,NVIDIA B200").split(",")]
MAX_SPEND_HR = float(os.environ.get("TAHQIQ_MAX_SPEND_HR", "45"))
STATE = ROOT / "logs" / "runpod_burst.json"
VOLUME_ID = os.environ.get("TAHQIQ_RUNPOD_VOLUME", "")
DATACENTER = os.environ.get("TAHQIQ_RUNPOD_DC", "")

# model → (datasets per pod, timing on the first pod). Slow models are split in two.
UNITS: list[dict] = []
_ALL = ["casablanca", "commonvoice", "masc_clean", "masc_noisy", "mgb2", "sada"]
_SPLIT = (["casablanca", "masc_clean", "sada"], ["commonvoice", "masc_noisy", "mgb2"])
for slug in ["whisper_large_v3", "qwen3_asr_1.7b", "qwen3_asr_0.6b", "seamless_m4t_v2_large", "cohere_transcribe_arabic_07_2026"]:
    UNITS.append({"slug": slug, "datasets": _SPLIT[0], "timing": True})
    UNITS.append({"slug": slug, "datasets": _SPLIT[1], "timing": False})
for slug in ["whisper_large_v3_turbo", "mms_1b_all"]:
    UNITS.append({"slug": slug, "datasets": _ALL, "timing": True})
UNITS.append({"slug": "nvidia_fastconformer_ar_pc", "datasets": _ALL, "timing": True, "nemo": True, "then": "nvidia_fastconformer_ar_pcd"})


def _key() -> str:
    m = re.search(r"apikey = '([^']+)'", (Path.home() / ".runpod" / "config.toml").read_text())
    if not m:
        raise SystemExit("no RunPod api key in ~/.runpod/config.toml")
    return m.group(1)


import threading

_GQL_LOCK = threading.Lock()
_GQL_LAST = [0.0]


def gql(query: str, variables: dict | None = None) -> dict:
    """One request at a time, >= 0.7 s apart, with backoff on 429 (RunPod rate limits aggressively)."""
    for attempt in range(8):
        with _GQL_LOCK:
            wait = 0.7 - (time.time() - _GQL_LAST[0])
            if wait > 0:
                time.sleep(wait)
            r = httpx.post(API, headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
                           json={"query": query, "variables": variables or {}}, timeout=60)
            _GQL_LAST[0] = time.time()
        if r.status_code == 429:
            time.sleep(min(60, 5 * (attempt + 1)))
            continue
        r.raise_for_status()
        d = r.json()
        if d.get("errors"):
            raise RuntimeError(d["errors"])
        return d["data"]
    raise RuntimeError("RunPod API: rate limited 8 times in a row")


def spend_per_hr() -> float:
    try:
        return float(gql("{ myself { currentSpendPerHr } }")["myself"]["currentSpendPerHr"])
    except Exception:
        return 0.0


def create_pod(name: str, gpu: str, cloud: str) -> str | None:
    pub = (POD_KEY.with_suffix(".pub")).read_text().strip()
    q = """mutation($in: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: $in) { id } }"""
    inp = {"cloudType": cloud, "gpuCount": 1, "volumeInGb": 0, "containerDiskInGb": 120, "minVcpuCount": 8, "minMemoryInGb": 48,
           "gpuTypeId": gpu, "name": name, "imageName": IMAGE, "ports": "22/tcp", "volumeMountPath": "/vol",
           "env": [{"key": "PUBLIC_KEY", "value": pub}]}
    if VOLUME_ID:
        inp["networkVolumeId"] = VOLUME_ID
    if DATACENTER:
        inp["dataCenterId"] = DATACENTER
    try:
        d = gql(q, {"in": inp})
        return d["podFindAndDeployOnDemand"]["id"]
    except Exception as e:  # noqa: BLE001
        print(f"  {name}: {gpu}/{cloud} unavailable ({str(e)[:80]})")
        return None


def pod_ssh(pod_id: str) -> tuple[str, int] | None:
    d = gql("""query($id: String!) { pod(input: {podId: $id}) { desiredStatus runtime { ports { ip isIpPublic privatePort publicPort type } } } }""", {"id": pod_id})
    rt = (d.get("pod") or {}).get("runtime")
    for p in (rt or {}).get("ports") or []:
        if p["privatePort"] == 22 and p["isIpPublic"]:
            return p["ip"], int(p["publicPort"])
    return None


def terminate(pod_id: str) -> None:
    try:
        gql("""mutation($id: String!) { podTerminate(input: {podId: $id}) }""", {"id": pod_id})
    except Exception as e:  # noqa: BLE001
        print("terminate failed", pod_id, e)


def ssh_opts(ip: str, port: int) -> list[str]:
    return ["-i", str(POD_KEY), "-p", str(port), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=20", f"root@{ip}"]


ENSURE_DATA = """
export HF_HUB_DISABLE_XET=1
if mkdir /vol/.seed.lock 2>/dev/null; then
  echo "seeding/verifying the volume (this pod holds the lock)"
  pip install -q huggingface_hub
python - <<'PY'
import os, tarfile, time
from huggingface_hub import hf_hub_download, list_repo_files
repo = "HeshamHaroon/tahqiq-audio-private"; tok = os.environ["HF_TOKEN"]
files = list_repo_files(repo, repo_type="dataset", token=tok)
for f in files:
    if f.startswith("manifests/"):
        p = hf_hub_download(repo, f, repo_type="dataset", token=tok)
        open("/vol/manifests/" + os.path.basename(f), "wb").write(open(p, "rb").read())
import json
def expected(ds):
    p = f"/vol/manifests/{ds}.jsonl"
    return sum(1 for l in open(p, encoding="utf-8") if l.strip()) if os.path.exists(p) else -1
for f in sorted(files):
    if not f.startswith("audio_tars/"): continue
    ds = os.path.basename(f)[:-4]
    have = len(os.listdir(f"/vol/audio/{ds}")) if os.path.isdir(f"/vol/audio/{ds}") else 0
    if have and have == expected(ds): print("have", ds, have, flush=True); continue
    if have: print("incomplete", ds, have, "expected", expected(ds), "-> re-pull", flush=True); import shutil; shutil.rmtree(f"/vol/audio/{ds}")
    t0 = time.time(); p = hf_hub_download(repo, f, repo_type="dataset", token=tok)
    tarfile.open(p).extractall("/vol/audio", filter="data"); os.remove(p)
    print(ds, "done", round(time.time()-t0), "s", "files", len(os.listdir(f"/vol/audio/{ds}")), "expected", expected(ds), flush=True)
PY

  touch /vol/.seed.done; rmdir /vol/.seed.lock
else
  echo "waiting for another pod to finish seeding"; while [ ! -f /vol/.seed.done ] || [ -d /vol/.seed.lock ]; do sleep 15; done
fi
"""


def bootstrap_script(unit: dict) -> str:
    key = Path(os.environ["TAHQIQ_AUDIO_KEY_PATH"]).read_text()
    pip = ("pip install -q -U 'transformers>=5.13' accelerate soundfile scipy numpy rapidfuzz jiwer pyyaml tenacity httpx tqdm "
           "python-dotenv huggingface_hub sentencepiece protobuf tiktoken paramiko 'mistral-common[audio]'")
    if unit.get("nemo"):
        pip += " && pip install -q 'nemo_toolkit[asr]' cython packaging"
    if unit.get("omni"):
        pip += " && pip install -q omnilingual-asr"
    def eval_cmd(slug: str, datasets: list[str], timing: bool, hf_id: str | None, display: str | None) -> str:
        sl = f" --slice {unit['slice']}" if unit.get("slice") else ""
        c = f"python scripts/eval_job.py --slug {slug} --datasets {','.join(datasets)} --helper{sl}"
        if hf_id:
            c += f" --hf-id {hf_id}"
        if display:
            c += f" --display-name '{display}'"
        if timing:
            c += f" && python scripts/timing.py {slug}" + (f" --hf-id {hf_id}" if hf_id else "")
        return c
    cmds = eval_cmd(unit["slug"], unit["datasets"], unit.get("timing", False), unit.get("hf_id"), unit.get("display_name"))
    if unit.get("then"):
        cmds += " && " + eval_cmd(unit["then"], unit["datasets"], True, None, None)
    secrets = "" if unit.get("remote_code") else f"""export HF_TOKEN='{os.environ["HF_TOKEN"]}' TAHQIQ_AUDIO_SSH='{os.environ["TAHQIQ_AUDIO_SSH"]}'
export TAHQIQ_AUDIO_KEY=$(cat <<'KEYEOF'
{key}KEYEOF
)"""
    trc = "export TAHQIQ_TRUST_REMOTE_CODE=1" if unit.get("remote_code") else ""
    return f"""set -e
export HF_HUB_DISABLE_XET=1 HF_HOME=/workspace/hf TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TOKENIZERS_PARALLELISM=false TAHQIQ_VOLUME=/vol
export TAHQIQ_RESULTS_REPO='{hub.RESULTS_REPO}' TAHQIQ_REQUESTS_REPO='{hub.REQUESTS_REPO}'
{secrets}
{trc}
apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq libsndfile1 rsync >/dev/null 2>&1 || true
mkdir -p /workspace && cd /workspace
{ENSURE_DATA if not unit.get("remote_code") else "while [ ! -f /vol/.seed.done ]; do sleep 15; done"}
{pip} > /workspace/pip.log 2>&1
python - <<'PY'
import os, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ["TAHQIQ_RESULTS_REPO"], "code/tahqiq.tar.gz", repo_type="dataset", token=os.environ.get("HF_TOKEN"))
tarfile.open(p).extractall("/workspace", filter="data")
PY
cd /workspace/tahqiq && export PYTHONPATH=/workspace/tahqiq && (pip install -q -e . --no-deps || true)
python -c "import tahqiq, sys; print('tahqiq import ok', sys.version.split()[0])"
{cmds}
echo UNIT_DONE
"""


def run_unit(unit: dict, idx: int, state: dict, *, retry_minutes: int = 120) -> None:
    name = f"tahqiq-{idx}-{unit['slug'][:18]}"
    prev = state.get(name) or {}
    if prev.get("status", "").startswith("FINISHED"):
        return
    pod_id = prev.get("pod") if prev.get("status") in ("CREATED", "RUNNING") else None
    if pod_id and not pod_ssh(pod_id) and not gql("""query($id: String!) { pod(input: {podId: $id}) { desiredStatus } }""", {"id": pod_id}).get("pod"):
        pod_id = None  # pod is gone
    prefs = [g for g in GPU_PREFS if not (unit.get("nemo") and "B200" in g)]
    deadline = time.time() + retry_minutes * 60
    while not pod_id:
        if spend_per_hr() >= MAX_SPEND_HR:
            print(f"{name}: spend cap {MAX_SPEND_HR}/h reached, waiting", flush=True)
        else:
            for gpu in prefs:
                for cloud in ("COMMUNITY", "SECURE"):
                    pod_id = create_pod(name, gpu, cloud)
                    if pod_id:
                        print(f"{name}: pod {pod_id} on {gpu} ({cloud})", flush=True)
                        break
                if pod_id:
                    break
        if pod_id or time.time() > deadline:
            break
        state[name] = {"status": "NO_GPU", "unit": unit}; _save(state)
        time.sleep(120)
    if not pod_id:
        state[name] = {"status": "NO_GPU", "unit": unit}; _save(state)
        return
    state[name] = {"pod": pod_id, "status": "CREATED", "unit": unit}
    _save(state)
    for _ in range(45):  # up to 15 min for the pod to come up
        time.sleep(20)
        addr = pod_ssh(pod_id)
        if addr and subprocess.call(["ssh", *ssh_opts(*addr), "true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            break
    else:
        state[name]["status"] = "NO_SSH"; _save(state); terminate(pod_id); return
    ip, port = addr
    state[name].update({"ip": ip, "port": port, "status": "RUNNING"}); _save(state)
    script = bootstrap_script(unit)
    log = ROOT / "logs" / "runpod" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "wb") as f:
        rc = subprocess.run(["ssh", *ssh_opts(ip, port), "bash -s"], input=script.encode(), stdout=f, stderr=subprocess.STDOUT).returncode
    state[name]["status"] = "FINISHED" if rc == 0 else f"FAILED rc={rc}"
    _save(state)
    collect_pod(ip, port)
    terminate(pod_id)
    state[name]["status"] += " (terminated)"; _save(state)
    print(f"{name}: {state[name]['status']}", flush=True)


def collect_pod(ip: str, port: int) -> None:
    dst = ROOT / "results"
    dst.mkdir(exist_ok=True)
    subprocess.call(["rsync", "-az", "-e", f"ssh -i {POD_KEY} -p {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
                     f"root@{ip}:/workspace/tahqiq/results/", str(dst) + "/"])


def _save(state: dict) -> None:
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")


SEED_SCRIPT = """set -e
export HF_TOKEN='{token}' HF_HUB_DISABLE_XET=1 HF_HOME=/workspace/hf
pip install -q huggingface_hub
mkdir -p /vol/audio /vol/manifests && cd /vol && rm -f /vol/*.tar
python - <<'PY'
import os, tarfile, time
from huggingface_hub import hf_hub_download, list_repo_files
repo = "HeshamHaroon/tahqiq-audio-private"; tok = os.environ["HF_TOKEN"]
files = list_repo_files(repo, repo_type="dataset", token=tok)
for f in files:
    if f.startswith("manifests/"):
        p = hf_hub_download(repo, f, repo_type="dataset", token=tok)
        open("/vol/manifests/" + os.path.basename(f), "wb").write(open(p, "rb").read())
import json
def expected(ds):
    p = f"/vol/manifests/{ds}.jsonl"
    return sum(1 for l in open(p, encoding="utf-8") if l.strip()) if os.path.exists(p) else -1
for f in sorted(files):
    if not f.startswith("audio_tars/"): continue
    ds = os.path.basename(f)[:-4]
    have = len(os.listdir(f"/vol/audio/{ds}")) if os.path.isdir(f"/vol/audio/{ds}") else 0
    if have and have == expected(ds): print("have", ds, have, flush=True); continue
    if have: print("incomplete", ds, have, "expected", expected(ds), "-> re-pull", flush=True); import shutil; shutil.rmtree(f"/vol/audio/{ds}")
    t0 = time.time(); p = hf_hub_download(repo, f, repo_type="dataset", token=tok)
    tarfile.open(p).extractall("/vol/audio", filter="data"); os.remove(p)
    print(ds, "done", round(time.time()-t0), "s", "files", len(os.listdir(f"/vol/audio/{ds}")), "expected", expected(ds), flush=True)
PY
du -sh /vol/audio/*; ls /vol/manifests; echo SEED_DONE
"""


def cmd_seed(gpu: str, pod_id: str | None = None) -> None:
    for g in ([gpu] + GPU_PREFS if not pod_id else []):
        for cloud in ("COMMUNITY", "SECURE"):
            pod_id = create_pod("tahqiq-seed", g, cloud)
            if pod_id:
                break
        if pod_id:
            break
    if not pod_id:
        raise SystemExit("no pod for seeding")
    print("seed pod", pod_id)
    for _ in range(60):
        time.sleep(10)
        addr = pod_ssh(pod_id)
        if addr and subprocess.call(["ssh", *ssh_opts(*addr), "true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            break
    else:
        terminate(pod_id); raise SystemExit("seed pod never came up")
    script = SEED_SCRIPT.replace("{token}", os.environ["HF_TOKEN"])
    rc = subprocess.run(["ssh", *ssh_opts(*addr), "bash -s"], input=script.encode()).returncode
    terminate(pod_id)
    print("seed rc", rc)


SLOW_TYPES = {"gemma4", "gemma3n", "voxtral_realtime", "omniasr", "qwen2_5_omni", "qwen3_omni_moe"}
SLOW_PREFIXES = ("facebook/omniASR-LLM-",)


def units_from_candidates(path: Path, *, max_priority: int = 1, split_over_gb: float = 8.0) -> list[dict]:
    """Build work units from data/candidates.yaml: registry slugs where known, auto-detected adapters otherwise.

    Models that need remote code or an unknown architecture are skipped (listed on stderr) so they
    can get a hand-written adapter later. Models with weights above ``split_over_gb`` get two pods.
    """
    import yaml

    from tahqiq.models import registry
    from tahqiq.models.auto import ADAPTERS

    cands = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("models") or []
    by_hf = {}
    for slug in registry.list_models():
        try:
            by_hf[registry.build(slug).hf_id] = slug
        except Exception:
            pass
    units, skipped = [], []
    for c in cands:
        if int(c.get("priority", 3)) > max_priority or not c.get("hf_id"):
            continue
        hf_id = c["hf_id"]
        slug = by_hf.get(hf_id) or hub.slug_for(hf_id)
        nemo = bool(c.get("nemo"))
        omni = hf_id.startswith("facebook/omniASR-")
        remote = bool(c.get("needs_remote_code"))
        if slug not in registry.REGISTRY and not (nemo or omni):
            if str(c.get("model_type", "")) not in ADAPTERS:
                skipped.append((hf_id, c.get("model_type"), "no adapter"))
                continue
        big = float(c.get("weights_gb") or 0) > split_over_gb
        base = {"slug": slug, "nemo": nemo, "omni": omni, "remote_code": remote}
        if slug not in registry.REGISTRY:
            base.update({"hf_id": hf_id, "display_name": c.get("name") or hf_id})
        if big:
            units.append({**base, "datasets": _SPLIT[0], "timing": True})
            units.append({**base, "datasets": _SPLIT[1], "timing": False})
        else:
            units.append({**base, "datasets": _ALL, "timing": True})
    for s in skipped:
        print("skip (needs adapter):", s, file=sys.stderr)
    return units


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan"); pl.add_argument("--from-candidates", action="store_true"); pl.add_argument("--max-priority", type=int, default=1)
    sd = sub.add_parser("seed"); sd.add_argument("--gpu", default="NVIDIA RTX A5000"); sd.add_argument("--pod-id", default=None, help="reuse an existing pod")
    r = sub.add_parser("run"); r.add_argument("--max-pods", type=int, default=16); r.add_argument("--only", default=None, help="comma list of slugs"); r.add_argument("--from-candidates", action="store_true"); r.add_argument("--skip-done", action="store_true", help="skip slugs that already have results/<slug>/run.json"); r.add_argument("--max-priority", type=int, default=1); r.add_argument("--exclude-slow", action="store_true", help="skip LLM-style models (Gemma, Voxtral Realtime, omniASR)")
    sub.add_parser("collect"); sub.add_parser("kill")
    a = ap.parse_args()
    if a.cmd == "seed":
        cmd_seed(a.gpu, a.pod_id); return
    units_all = units_from_candidates(ROOT / "data" / "candidates.yaml", max_priority=getattr(a, "max_priority", 1)) if getattr(a, "from_candidates", False) else UNITS
    if a.cmd == "plan":
        for i, u in enumerate(units_all):
            print(i, u["slug"], u["datasets"], "timing" if u.get("timing") else "", "+then " + u["then"] if u.get("then") else "")
        print(len(units_all), "pods")
        return
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if a.cmd == "kill":
        d = gql("{ myself { pods { id name } } }")
        for p in d["myself"]["pods"]:
            if p["name"].startswith("tahqiq-"):
                terminate(p["id"]); print("terminated", p["name"])
        return
    if a.cmd == "collect":
        for name, s in state.items():
            if s.get("ip"):
                collect_pod(s["ip"], s["port"]); print("collected", name)
        return
    units = [u for u in units_all if not a.only or u["slug"] in a.only.split(",")]
    if a.skip_done:
        units = [u for u in units if not (ROOT / "results" / u["slug"] / "run.json").exists()]
    if getattr(a, "exclude_slow", False):
        units = [u for u in units if not u.get("slice") == "api"]
    print(f"{len(units)} units, {a.max_pods} pods at a time")
    with ThreadPoolExecutor(max_workers=max(1, a.max_pods)) as ex:
        list(ex.map(lambda iu: run_unit(iu[1], iu[0], state), enumerate(units)))
    print(json.dumps({k: v["status"] for k, v in state.items()}, indent=1))


if __name__ == "__main__":
    main()

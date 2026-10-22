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
GPU_PREFS = [g for g in os.environ.get("TAHQIQ_POD_GPUS", "NVIDIA H200,NVIDIA H100 80GB HBM3,NVIDIA H100 NVL,NVIDIA B200").split(",")]
STATE = ROOT / "logs" / "runpod_burst.json"
VOLUME_ID = os.environ.get("TAHQIQ_RUNPOD_VOLUME", "")
DATACENTER = os.environ.get("TAHQIQ_RUNPOD_DC", "")

# model → (datasets per pod, timing on the first pod). Slow models are split in two.
UNITS: list[dict] = []
_ALL = ["casablanca", "commonvoice", "masc_clean", "masc_noisy", "sada"]
_SPLIT = (["casablanca", "masc_clean", "sada"], ["commonvoice", "masc_noisy"])
for slug in ["whisper_large_v3", "qwen3_asr_1.7b", "qwen3_asr_0.6b", "seamless_m4t_v2_large", "cohere_transcribe_arabic_07_2026"]:
    UNITS.append({"slug": slug, "datasets": _SPLIT[0], "timing": True})
    UNITS.append({"slug": slug, "datasets": _SPLIT[1], "timing": False})
for slug in ["whisper_large_v3_turbo", "mms_1b_all"]:
    UNITS.append({"slug": slug, "datasets": _ALL, "timing": True})
UNITS.append({"slug": "nvidia_fastconformer_ar_pc", "datasets": _ALL, "timing": True, "nemo": True, "then": "nvidia_fastconformer_ar_pcd"})
UNITS.append({"slug": "openai_whisper-tiny", "datasets": _ALL, "timing": True, "hf_id": "openai/whisper-tiny", "display_name": "whisper-tiny"})


def _key() -> str:
    m = re.search(r"apikey = '([^']+)'", (Path.home() / ".runpod" / "config.toml").read_text())
    if not m:
        raise SystemExit("no RunPod api key in ~/.runpod/config.toml")
    return m.group(1)


def gql(query: str, variables: dict | None = None) -> dict:
    r = httpx.post(API, headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
                   json={"query": query, "variables": variables or {}}, timeout=60)
    r.raise_for_status()
    d = r.json()
    if d.get("errors"):
        raise RuntimeError(d["errors"])
    return d["data"]


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


def bootstrap_script(unit: dict) -> str:
    key = Path(os.environ["TAHQIQ_AUDIO_KEY_PATH"]).read_text()
    pip = ("pip install -q -U 'transformers>=5.13' accelerate soundfile scipy numpy rapidfuzz jiwer pyyaml tenacity httpx tqdm "
           "python-dotenv huggingface_hub sentencepiece protobuf tiktoken paramiko")
    if unit.get("nemo"):
        pip += " && pip install -q 'nemo_toolkit[asr]' cython packaging"
    def eval_cmd(slug: str, datasets: list[str], timing: bool, hf_id: str | None, display: str | None) -> str:
        c = f"python scripts/eval_job.py --slug {slug} --datasets {','.join(datasets)} --helper"
        if hf_id:
            c += f" --hf-id {hf_id}"
        if display:
            c += f" --display-name '{display}'"
        c += " && python -m tahqiq.runner --model " + slug + "".join(f" --dataset {d}" for d in datasets) + (f" --hf-id {hf_id}" if hf_id else "") + " --helper"
        if timing:
            c += f" && python scripts/timing.py {slug}" + (f" --hf-id {hf_id}" if hf_id else "")
        return c
    cmds = eval_cmd(unit["slug"], unit["datasets"], unit.get("timing", False), unit.get("hf_id"), unit.get("display_name"))
    if unit.get("then"):
        cmds += " && " + eval_cmd(unit["then"], unit["datasets"], True, None, None)
    return f"""set -e
export HF_HUB_DISABLE_XET=1 HF_HOME=/workspace/hf TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TOKENIZERS_PARALLELISM=false TAHQIQ_VOLUME=/vol
export HF_TOKEN='{os.environ["HF_TOKEN"]}' TAHQIQ_AUDIO_SSH='{os.environ["TAHQIQ_AUDIO_SSH"]}' TAHQIQ_RESULTS_REPO='{hub.RESULTS_REPO}' TAHQIQ_REQUESTS_REPO='{hub.REQUESTS_REPO}'
export TAHQIQ_AUDIO_KEY=$(cat <<'KEYEOF'
{key}KEYEOF
)
apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq libsndfile1 rsync >/dev/null 2>&1 || true
mkdir -p /workspace && cd /workspace
{pip} > /workspace/pip.log 2>&1
python - <<'PY'
import os, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ["TAHQIQ_RESULTS_REPO"], "code/tahqiq.tar.gz", repo_type="dataset", token=os.environ["HF_TOKEN"])
tarfile.open(p).extractall("/workspace", filter="data")
PY
cd /workspace/tahqiq && pip install -q -e . --no-deps
{cmds}
echo UNIT_DONE
"""


def run_unit(unit: dict, idx: int, state: dict) -> None:
    name = f"tahqiq-{idx}-{unit['slug'][:18]}"
    pod_id = None
    for gpu in GPU_PREFS:
        for cloud in ("COMMUNITY", "SECURE"):
            pod_id = create_pod(name, gpu, cloud)
            if pod_id:
                print(f"{name}: pod {pod_id} on {gpu} ({cloud})", flush=True)
                break
        if pod_id:
            break
    if not pod_id:
        state[name] = {"status": "NO_GPU", "unit": unit}
        return
    state[name] = {"pod": pod_id, "status": "CREATED", "unit": unit}
    _save(state)
    for _ in range(90):  # up to 15 min for the pod to come up
        time.sleep(10)
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
        rc = subprocess.call(["ssh", *ssh_opts(ip, port), "bash -s"], input=script.encode(), stdout=f, stderr=subprocess.STDOUT)
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
export TAHQIQ_AUDIO_SSH='{ssh}'
export TAHQIQ_AUDIO_KEY=$(cat <<'KEYEOF'
{key}KEYEOF
)
pip install -q paramiko
mkdir -p /vol/audio /vol/manifests /vol/hf && cd /vol
python - <<'PY'
import io, os, tarfile, time, paramiko
user, host = os.environ["TAHQIQ_AUDIO_SSH"].split("@", 1)
key = paramiko.Ed25519Key.from_private_key(io.StringIO(os.environ["TAHQIQ_AUDIO_KEY"]))
t = paramiko.Transport((host, 22)); t.connect(username=user, pkey=key); s = paramiko.SFTPClient.from_transport(t)
for name in s.listdir("manifests"):
    with s.open(f"manifests/{name}", "rb") as f, open(f"/vol/manifests/{name}", "wb") as o: o.write(f.read())
for name in s.listdir("audio_tars"):
    ds = name[:-4]
    if os.path.isdir(f"/vol/audio/{ds}") and os.listdir(f"/vol/audio/{ds}"): print("have", ds, flush=True); continue
    t0 = time.time()
    with s.open(f"audio_tars/{name}", "rb") as f, open(f"/vol/{name}", "wb") as o:
        f.prefetch()
        while True:
            b = f.read(8_000_000)
            if not b: break
            o.write(b)
    tarfile.open(f"/vol/{name}").extractall("/vol/audio", filter="data"); os.remove(f"/vol/{name}")
    print(ds, "done", round(time.time()-t0), "s", flush=True)
t.close()
PY
du -sh /vol/audio/*; echo SEED_DONE
"""


def cmd_seed(gpu: str) -> None:
    pub = POD_KEY.with_suffix(".pub").read_text().strip()
    pod_id = None
    for cloud in ("COMMUNITY", "SECURE"):
        pod_id = create_pod("tahqiq-seed", gpu, cloud)
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
    script = SEED_SCRIPT.format(ssh=os.environ["TAHQIQ_AUDIO_SSH"], key=Path(os.environ["TAHQIQ_AUDIO_KEY_PATH"]).read_text())
    rc = subprocess.call(["ssh", *ssh_opts(*addr), "bash -s"], input=script.encode())
    terminate(pod_id)
    print("seed rc", rc)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    sd = sub.add_parser("seed"); sd.add_argument("--gpu", default="NVIDIA RTX A5000")
    r = sub.add_parser("run"); r.add_argument("--max-pods", type=int, default=16); r.add_argument("--only", default=None, help="comma list of slugs")
    sub.add_parser("collect"); sub.add_parser("kill")
    a = ap.parse_args()
    if a.cmd == "seed":
        cmd_seed(a.gpu); return
    if a.cmd == "plan":
        for i, u in enumerate(UNITS):
            print(i, u["slug"], u["datasets"], "timing" if u.get("timing") else "", "+then " + u["then"] if u.get("then") else "")
        print(len(UNITS), "pods")
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
    units = [u for u in UNITS if not a.only or u["slug"] in a.only.split(",")][: a.max_pods]
    with ThreadPoolExecutor(max_workers=len(units)) as ex:
        list(ex.map(lambda iu: run_unit(iu[1], iu[0], state), enumerate(units)))
    print(json.dumps({k: v["status"] for k, v in state.items()}, indent=1))


if __name__ == "__main__":
    main()

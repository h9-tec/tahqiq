"""Controlled timing pass for one system: RTFx, per-utterance latency, peak VRAM, load time.

    CUDA_VISIBLE_DEVICES=5 .venv/bin/python scripts/timing.py whisper_large_v3_turbo
    CUDA_VISIBLE_DEVICES=5 .venv/bin/python scripts/timing.py <slug> --hf-id owner/model --adapter whisper

Protocol (published in docs/methodology.md): the fixed timing slice (in_timing_slice in the
manifests, ~60 utterances of 3-15 s across all sets), one warm-up pass, then 3 timed passes,
batch size 1, concurrency 1. RTFx = total audio seconds / total wall seconds, median over
passes. Latency = wall seconds per utterance, p50 and p95 over all timed utterances. Peak VRAM
from torch.cuda.max_memory_allocated. The number of other processes on the GPU at the time is
recorded, because the box is shared; a measurement with other jobs present is flagged.
Writes results/<slug>/timing.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import numpy as np  # noqa: E402

from tahqiq.manifest import load_manifest, manifest_paths  # noqa: E402
from tahqiq.models import registry  # noqa: E402
from tahqiq.runner import AUDIO, MANIFESTS, RESULTS  # noqa: E402


def gpu_neighbours() -> int:
    """Other compute processes on the visible GPU (0 when we are alone)."""
    try:
        import torch

        uuid = torch.cuda.get_device_properties(0).uuid if hasattr(torch.cuda.get_device_properties(0), "uuid") else None
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
        rows = [l.split(",") for l in out.strip().splitlines() if l.strip()]
        if uuid is None:
            return -1
        mine = str(subprocess.os.getpid())
        return sum(1 for g, p in rows if g.strip() == f"GPU-{uuid}" and p.strip() != mine)
    except Exception:
        return -1


def measure(slug: str, *, hf_id: str | None = None, display_name: str | None = None, adapter: str | None = None, passes: int = 3) -> dict:
    """Timing pass; returns the timing dict and writes results/<slug>/timing.json."""
    utts = [u for p in manifest_paths(MANIFESTS) for u in load_manifest(p) if u.in_timing_slice]
    utts.sort(key=lambda u: u.id)
    audio_s = sum(u.duration_s for u in utts)
    if hf_id is None and slug not in registry.REGISTRY:
        rj = RESULTS / slug / "run.json"
        if rj.exists():
            hf_id = json.loads(rj.read_text(encoding="utf-8")).get("hf_id")
    model = registry.build(slug, hf_id=hf_id, display_name=display_name, adapter=adapter)
    torch = None
    try:
        import torch

        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    t0 = time.perf_counter()
    model.load()
    load_s = time.perf_counter() - t0
    for u in utts[:5]:
        model.transcribe_path(AUDIO / u.audio_rel)
    passes_wall, lat = [], []
    for _ in range(passes):
        t1 = time.perf_counter()
        for u in utts:
            lat.append(model.transcribe_path(AUDIO / u.audio_rel).wall_s)
        passes_wall.append(time.perf_counter() - t1)
    peak = gpu = None
    try:
        peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    neighbours = gpu_neighbours()
    model.unload()
    lat_a = np.array(lat)
    doc = {
        "slug": slug, "n_utts": len(utts), "audio_s": round(audio_s, 1),
        "passes_wall_s": [round(x, 2) for x in passes_wall],
        "rtfx": round(audio_s / float(np.median(passes_wall)), 1),
        "latency_p50_s": round(float(np.percentile(lat_a, 50)), 3),
        "latency_p95_s": round(float(np.percentile(lat_a, 95)), 3),
        "load_s": round(load_s, 1), "peak_vram_gb": peak, "gpu": gpu, "gpu_other_processes": neighbours,
        "batch_size": 1, "concurrency": 1, "machine": platform.node(),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = RESULTS / slug / "timing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--hf-id", default=None)
    ap.add_argument("--display-name", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--passes", type=int, default=3)
    args = ap.parse_args()
    print(json.dumps(measure(args.slug, hf_id=args.hf_id, display_name=args.display_name, adapter=args.adapter, passes=args.passes)))


if __name__ == "__main__":
    main()

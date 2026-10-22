"""Load each model, transcribe one wav twice (cold, warm), print text + timings, unload.

    uv run python scripts/smoke_models.py path/to.wav                  # all registry models
    uv run python scripts/smoke_models.py path/to.wav --model whisper_large_v3_turbo --model mms_1b_all
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tahqiq.models import registry  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--model", action="append", default=None)
    args = ap.parse_args()
    wav = Path(args.wav)
    models = args.model or registry.list_models()
    results = {}
    for mid in models:
        print(f"\n=== {mid}", flush=True)
        try:
            m = registry.build(mid)
            t0 = time.perf_counter()
            m.load()
            load_s = time.perf_counter() - t0
            t1 = m.transcribe_path(wav)
            t2 = m.transcribe_path(wav)
            print(f"load {load_s:.1f}s | cold {t1.wall_s:.2f}s | warm {t2.wall_s:.2f}s", flush=True)
            print(f"text: {t2.text[:200]}", flush=True)
            results[mid] = "OK"
            m.unload()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            results[mid] = f"FAIL {type(e).__name__}: {str(e)[:120]}"
    print("\n=== SUMMARY")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

"""Apply an adapter's deterministic text post-processing to an existing per-utterance cache, so a fix to the
output stage does not need a GPU re-run. Today: Buckwalter -> Arabic for CTC models whose vocabulary is
Buckwalter (same function the CTC adapter now applies at decode time). Records the step in run.json.

    .venv/bin/python scripts/postprocess_cache.py --detect            # list board CTC models with a Buckwalter vocab
    .venv/bin/python scripts/postprocess_cache.py SLUG [SLUG ...]     # rewrite results/SLUG/*.jsonl in place
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from tahqiq.normalize.buckwalter import looks_like_buckwalter_vocab, to_arabic  # noqa: E402
from tahqiq.runner import RESULTS  # noqa: E402

STEP = "buckwalter -> arabic ('/' = word delimiter)"


def vocab_is_buckwalter(hf_id: str) -> bool:
    from transformers import AutoTokenizer

    return looks_like_buckwalter_vocab(AutoTokenizer.from_pretrained(hf_id).get_vocab())


def rewrite(slug: str) -> None:
    d = RESULTS / slug
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    if run.get("decode_params", {}).get("postprocess") == STEP:
        print(f"{slug}: already post-processed"); return
    if not vocab_is_buckwalter(run["hf_id"]):
        raise SystemExit(f"{slug}: vocabulary is not Buckwalter; refusing")
    n = 0
    for f in sorted(d.glob("*.jsonl")):
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            if r.get("hyp"):
                r["hyp"] = to_arabic(r["hyp"]); n += 1
        f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    run.setdefault("decode_params", {})["postprocess"] = STEP
    run["postprocessed_from_cache"] = "hypotheses converted from the cached greedy CTC output (deterministic; no re-decode)"
    (d / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{slug}: {n} hypotheses converted")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="*")
    ap.add_argument("--detect", action="store_true")
    a = ap.parse_args()
    if a.detect:
        for rj in sorted(RESULTS.glob("*/run.json")):
            run = json.loads(rj.read_text(encoding="utf-8"))
            if run.get("hf_id") and "ctc" in json.dumps(run.get("decode_params", {})).lower() + str(run.get("adapter", "")) or \
               run.get("decode_params") == {"decoding": "greedy"}:
                try:
                    if vocab_is_buckwalter(run["hf_id"]):
                        print("BUCKWALTER", rj.parent.name, run["hf_id"])
                except Exception as e:  # noqa: BLE001
                    print("skip", rj.parent.name, type(e).__name__)
        return
    for s in a.slugs:
        rewrite(s)


if __name__ == "__main__":
    main()

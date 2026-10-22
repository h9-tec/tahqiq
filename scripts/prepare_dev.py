"""Validation ("dev") subsamples of the six corpora for the dev-vs-test contamination gap.

    uv run python scripts/prepare_dev.py --sada-src /data/ziad/data/sada2022/raw --masc-src /data/arabic_pretrain/MASC-Arabic/data
    uv run python scripts/prepare_dev.py --limit 20     # smoke

About 500 utterances per corpus, chosen deterministically by id hash (same choice on any machine).
Manifests go to data/dev_manifests/dev_<set>.jsonl (own hash), audio to audio/dev_<set>/.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from tahqiq.manifest import write_manifest_hash  # noqa: E402

DEV = ROOT / "data" / "dev_manifests"
# every = validation rows / ~500
PLAN = {
    "casablanca": {"module": "tahqiq.data.casablanca", "kw": {"split": "validation", "every": 13, "name": "dev_casablanca"}},
    "commonvoice": {"module": "tahqiq.data.commonvoice", "kw": {"split": "validation", "every": 20, "name": "dev_commonvoice"}},
    "mgb2": {"module": "tahqiq.data.mgb2", "kw": {"split": "validation", "every": 10, "name": "dev_mgb2"}},
    "sada": {"module": "tahqiq.data.sada", "kw": {"split": "validation", "every": 10, "name": "dev_sada"}},
    "masc": {"module": "tahqiq.data.masc", "kw": {"split": "validation", "every": 38, "name_prefix": "dev_"}},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sada-src", default="/data/ziad/data/sada2022/raw")
    ap.add_argument("--masc-src", default="/data/arabic_pretrain/MASC-Arabic/data")
    a = ap.parse_args()
    DEV.mkdir(parents=True, exist_ok=True)
    for ds, spec in PLAN.items():
        if a.only and ds not in a.only.split(","):
            continue
        mod = importlib.import_module(spec["module"])
        kw = dict(spec["kw"])
        kw.update({"manifests_dir": DEV, "n_api": 10_000, "n_timing": 0, "per_config_limit": a.limit})
        if ds == "sada":
            kw["src"] = Path(a.sada_src)
        if ds == "masc":
            local = Path(a.masc_src)
            kw["src"] = local if local.exists() and any(local.glob("validation-*.parquet")) else "MohamedRashad/MASC-Arabic"
        print("==", ds, kw.get("split"), flush=True)
        mod.build(**kw)
    print("DEV_MANIFEST_SHA256", write_manifest_hash(DEV))


if __name__ == "__main__":
    main()

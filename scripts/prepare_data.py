"""Build audio/ and data/manifests/ for one dataset, then refresh the manifest hash.

    uv run python scripts/prepare_data.py casablanca
    uv run python scripts/prepare_data.py sada --src /data/ziad/data/sada2022/raw
    uv run python scripts/prepare_data.py commonvoice            # HF mirror
    uv run python scripts/prepare_data.py masc --src /data/arabic_pretrain/MASC-Arabic/data
    uv run python scripts/prepare_data.py casablanca --limit 20   # smoke

Set HARFIYYAN_AUDIO to put audio somewhere other than <repo>/audio.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from harfiyyan.manifest import write_manifest_hash  # noqa: E402

BUILDERS = {
    "casablanca": "harfiyyan.data.casablanca",
    "sada": "harfiyyan.data.sada",
    "commonvoice": "harfiyyan.data.commonvoice",
    "masc": "harfiyyan.data.masc",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(BUILDERS))
    ap.add_argument("--src", default=None, help="source dir / HF id (dataset specific)")
    ap.add_argument("--limit", type=int, default=None, help="utterances per config/dataset (smoke test)")
    ap.add_argument("--n-api", type=int, default=1000)
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")

    mod = importlib.import_module(BUILDERS[args.dataset])
    kwargs = {"per_config_limit": args.limit, "n_api": args.n_api}
    if args.src:
        kwargs["src"] = Path(args.src) if Path(args.src).exists() else args.src
    mod.build(**kwargs)
    print("MANIFEST_SHA256", write_manifest_hash(ROOT / "data" / "manifests"))


if __name__ == "__main__":
    main()

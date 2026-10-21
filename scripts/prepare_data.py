"""Build audio/ and data/manifests/ for one dataset, then refresh the manifest hash.

    uv run python scripts/prepare_data.py casablanca
    uv run python scripts/prepare_data.py casablanca --per-config-limit 20   # smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from harfiyyan.manifest import write_manifest_hash  # noqa: E402

BUILDERS = {
    "casablanca": "harfiyyan.data.casablanca",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(BUILDERS))
    ap.add_argument("--per-config-limit", type=int, default=None)
    ap.add_argument("--n-api", type=int, default=1000)
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")

    import importlib

    mod = importlib.import_module(BUILDERS[args.dataset])
    mod.build(per_config_limit=args.per_config_limit, n_api=args.n_api)
    digest = write_manifest_hash(ROOT / "data" / "manifests")
    print("MANIFEST_SHA256", digest)


if __name__ == "__main__":
    main()

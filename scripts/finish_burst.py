"""After a burst: assemble each slug's run.json from its unit runs, publish, rebuild the board.

    uv run python scripts/finish_burst.py            # every results/<slug>/runs/*.json without a fresh run.json
    uv run python scripts/finish_burst.py --slugs a,b
    uv run python scripts/finish_burst.py --no-upload
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from assemble_run import assemble  # noqa: E402

from tahqiq import hub  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs", default=None)
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    results = ROOT / "results"
    slugs = a.slugs.split(",") if a.slugs else sorted(p.parent.name for p in results.glob("*/runs"))
    ok, bad = [], []
    for slug in slugs:
        try:
            run = assemble(slug)
        except SystemExit as e:
            print(f"{slug}: {e}"); bad.append(slug); continue
        cmd = [sys.executable, str(ROOT / "scripts" / "publish.py"), slug]
        if run.get("hf_id"):
            cmd += ["--hf-id", run["hf_id"]]
        if run.get("display_name"):
            cmd += ["--display-name", run["display_name"]]
        if a.no_upload:
            cmd.append("--no-upload")
        rc = subprocess.call(cmd, cwd=ROOT)
        (ok if rc == 0 else bad).append(slug)
    if not a.no_upload and ok:
        hub.rebuild_board()
    print("published:", ok)
    print("failed:", bad)


if __name__ == "__main__":
    main()

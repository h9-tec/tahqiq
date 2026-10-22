"""Re-score and re-upload every completed local run (e.g. after the baseline finishes or a scorer change).

    uv run python scripts/republish_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    slugs = sorted(p.parent.name for p in (ROOT / "results").glob("*/run.json"))
    for slug in slugs:
        print("==", slug, flush=True)
        subprocess.call([sys.executable, str(ROOT / "scripts" / "publish.py"), slug], cwd=ROOT)
    subprocess.call([sys.executable, "-c", "from tahqiq import hub; hub.rebuild_board()"], cwd=ROOT)

"""Dataset builders: original source → audio/<dataset>/<id>.wav + data/manifests/<dataset>.jsonl.

Each builder reproduces Elm's test membership from ``data/elm/<name>.json`` so
our utterance set equals theirs, and reports any mismatch instead of hiding it.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ELM_DIR = ROOT / "data" / "elm"
MANIFESTS = ROOT / "data" / "manifests"
AUDIO = ROOT / "audio"


def elm_members(name: str) -> dict[str, dict]:
    """Elm manifest rows keyed by audio basename without extension."""
    out: dict[str, dict] = {}
    with open(ELM_DIR / f"{name}.json", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = Path(row["audio_filepath"]).stem
            out[key] = row
    return out

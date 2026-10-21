"""Dataset builders: original source → audio/<dataset>/<id>.wav + data/manifests/<dataset>.jsonl.

Each builder reproduces Elm's test membership from ``data/elm/<name>.json`` so
our utterance set equals theirs, and reports any mismatch instead of hiding it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..manifest import Utterance, make_id, map_dialect, sha256_file, write_manifest
from ..normalize.numbers import has_written_number
from ..slices import assign_slices

ROOT = Path(__file__).resolve().parents[2]
ELM_DIR = ROOT / "data" / "elm"
MANIFESTS = ROOT / "data" / "manifests"
AUDIO = Path(os.environ.get("HARFIYYAN_AUDIO", ROOT / "audio"))


def elm_members(name: str) -> dict[str, dict]:
    """Elm manifest rows keyed by audio basename without extension."""
    out: dict[str, dict] = {}
    with open(ELM_DIR / f"{name}.json", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[Path(row["audio_filepath"]).stem] = row
    return out


def make_utt(
    dataset: str,
    source_id: str,
    wav: Path,
    audio_rel: str,
    reference: str,
    *,
    duration_s: float,
    dialect_raw: str = "",
    in_elm: bool,
) -> Utterance:
    return Utterance(
        id=make_id(dataset, source_id),
        dataset=dataset,
        split="test",
        source_id=source_id,
        audio_rel=audio_rel,
        audio_sha256=sha256_file(wav),
        duration_s=float(duration_s),
        reference=reference,
        dialect_raw=dialect_raw,
        dialect=map_dialect(dialect_raw),
        has_number=has_written_number(reference),
        elm_member=in_elm,
    )


def finalize(
    dataset: str,
    utts: list[Utterance],
    elm: dict[str, dict],
    seen_elm: set[str],
    *,
    manifests_dir: Path,
    n_api: int,
    n_timing: int,
    seed: int,
) -> list[Utterance]:
    missing = len(set(elm) - seen_elm)
    extra = sum(1 for u in utts if not u.elm_member)
    print(
        f"{dataset}: {len(utts)} utts, {sum(u.duration_s for u in utts)/3600:.2f} h; "
        f"elm members matched {len(seen_elm)}/{len(elm)}; not in elm: {extra}; elm rows we lack: {missing}"
    )
    assign_slices(utts, n_api=n_api, n_timing=n_timing, seed=seed)
    utts.sort(key=lambda u: u.id)
    write_manifest(manifests_dir / f"{dataset}.jsonl", utts)
    return utts

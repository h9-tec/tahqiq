"""MASC (Massive Arabic Speech Corpus, CC-BY-4.0), test split: clean and noisy.

Source: parquet shards of the ``MohamedRashad/MASC-Arabic`` mirror
(columns video_id, duration, text, type 'c'|'n', audio@16k). Pass ``--src`` as
the directory holding ``test-*.parquet`` or the HF id.

Elm's ids are ``<video_id>_<segment start seconds>``; the mirror has no start
offset, so membership is matched on (video_id, duration rounded to 2 dp) and,
when that is ambiguous, on text after Elm normalization. Unmatched rows are
kept with ``elm_member=False`` and reported.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .. import io as hio
from ..manifest import Utterance, make_id
from ..normalize.elm import normalize_elm
from . import AUDIO, MANIFESTS, elm_members, finalize, make_utt

DEFAULT_SRC = "MohamedRashad/MASC-Arabic"
NAMES = {"c": "masc_clean", "n": "masc_noisy"}


def _elm_index(name: str) -> dict[tuple[str, float], list[tuple[str, dict]]]:
    idx: dict[tuple[str, float], list[tuple[str, dict]]] = defaultdict(list)
    for key, row in elm_members(name).items():
        video_id, _, _start = key.rpartition("_")
        idx[(video_id, round(float(row["duration"]), 2))].append((key, row))
    return idx


def build(
    *,
    src: str | Path = DEFAULT_SRC,
    per_config_limit: int | None = None,
    audio_dir: Path = AUDIO,
    manifests_dir: Path = MANIFESTS,
    n_api: int = 1000,
    n_timing: int = 12,
    seed: int = 0,
    split: str = "test",
    every: int = 1,
    name_prefix: str = "",
) -> list[Utterance]:
    from datasets import Audio, load_dataset

    from . import keep

    src = str(src)
    if Path(src).is_dir():
        ds = load_dataset("parquet", data_files=str(Path(src) / f"{split}-*.parquet"), split="train")
    else:
        ds = load_dataset(src, data_files={split: f"data/{split}-*.parquet"}, split=split)
    names = {t: name_prefix + n for t, n in NAMES.items()}
    ds = ds.cast_column("audio", Audio(decode=False))

    elm_files = {"c": "masc_test_clean", "n": "masc_test_noisy"}
    elm_idx = {t: (_elm_index(elm_files[t]) if split == "test" else {}) for t in NAMES}
    elm_all = {t: (elm_members(elm_files[t]) if split == "test" else {}) for t in NAMES}
    utts: dict[str, list[Utterance]] = {n: [] for n in names.values()}
    seen: dict[str, set[str]] = {n: set() for n in names.values()}
    counts = {"c": 0, "n": 0}
    for i, row in enumerate(tqdm(ds, desc="masc", unit="utt")):
        t = (row.get("type") or "").strip().lower()[:1]
        if t not in NAMES:
            continue
        if per_config_limit and counts[t] >= per_config_limit:
            if all(counts[k] >= per_config_limit for k in NAMES):
                break
            continue
        name = names[t]
        video_id = str(row["video_id"])
        if not keep(name, f"{video_id}_row{i}", every):
            continue
        arr, sr = hio.decode_audio_bytes(row["audio"]["bytes"])
        dur = float(row.get("duration") or 0.0) or len(arr) / sr
        text = row.get("text") or ""
        cands = [c for c in elm_idx[t].get((video_id, round(dur, 2)), []) if c[0] not in seen[name]]
        match = None
        if len(cands) == 1:
            match = cands[0]
        elif len(cands) > 1:
            nt = normalize_elm(text)
            exact = [c for c in cands if normalize_elm(c[1]["text"]) == nt]
            match = exact[0] if exact else cands[0]
        if match:
            key, erow = match
            seen[name].add(key)
            source_id, ref, in_elm = key, erow["text"], True
        else:
            source_id, ref, in_elm = f"{video_id}_row{i}", text, False
        rel = f"{name}/{make_id(name, source_id)}.wav"
        wav = audio_dir / rel
        if not wav.exists():
            hio.write_wav_16k(wav, arr, sr)
        utts[name].append(make_utt(name, source_id, wav, rel, ref, duration_s=dur, dialect_raw="", in_elm=in_elm))
        counts[t] += 1

    out: list[Utterance] = []
    for t, name in names.items():
        out += finalize(name, utts[name], elm_all[t], seen[name], manifests_dir=manifests_dir,
                        n_api=n_api, n_timing=n_timing, seed=seed)
    return out

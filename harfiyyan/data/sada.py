"""SADA 2022 (SDAIA), test split, re-cut from the full episode wavs.

Source layout (Kaggle release): ``raw/test.csv`` with FileName, SegmentID,
SegmentStart, SegmentEnd, SpeakerDialect, Environment, ProcessedText, and
``raw/batch_*/*.wav``. Elm's ids are the SegmentIDs; Elm's text equals
ProcessedText. We cut [SegmentStart, SegmentEnd] from the episode file.

Point ``--src`` at the directory holding ``test.csv`` and ``batch_*``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from .. import io as hio
from ..manifest import Utterance, make_id
from . import AUDIO, MANIFESTS, elm_members, finalize, make_utt

NAME = "sada"


def build(
    *,
    src: Path,
    per_config_limit: int | None = None,
    audio_dir: Path = AUDIO,
    manifests_dir: Path = MANIFESTS,
    n_api: int = 1000,
    n_timing: int = 12,
    seed: int = 0,
) -> list[Utterance]:
    elm = elm_members("sada_test")
    rows = list(csv.DictReader(open(src / "test.csv", encoding="utf-8")))
    if per_config_limit:
        rows = rows[:per_config_limit]
    utts: list[Utterance] = []
    seen_elm: set[str] = set()
    cache: dict[str, tuple[np.ndarray, int]] = {}
    for r in tqdm(rows, desc=NAME, unit="utt"):
        seg = r["SegmentID"]
        rel = f"{NAME}/{make_id(NAME, seg)}.wav"
        wav = audio_dir / rel
        start, end = float(r["SegmentStart"]), float(r["SegmentEnd"])
        if not wav.exists():
            fn = r["FileName"]
            if fn not in cache:
                cache.clear()  # episodes are ~10 min; keep one at a time
                audio, sr = sf.read(str(src / fn), dtype="float32", always_2d=False)
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                cache[fn] = (audio, sr)
            audio, sr = cache[fn]
            seg_audio = audio[int(start * sr): int(end * sr)]
            hio.write_wav_16k(wav, seg_audio, sr)
        in_elm = seg in elm
        if in_elm:
            seen_elm.add(seg)
        ref = elm[seg]["text"] if in_elm else (r.get("ProcessedText") or r.get("GroundTruthText") or "")
        dialect_raw = (r.get("SpeakerDialect") or "").strip()
        utts.append(make_utt(NAME, seg, wav, rel, ref, duration_s=end - start, dialect_raw=dialect_raw, in_elm=in_elm))
    return finalize(NAME, utts, elm, seen_elm, manifests_dir=manifests_dir, n_api=n_api, n_timing=n_timing, seed=seed)

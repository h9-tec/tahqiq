"""Casablanca (UBC-NLP/Casablanca, test split, 8 country configs, CC-BY-NC-ND 4.0).

Audio is never redistributed; only our per-utterance outputs are.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from .. import io as hio
from ..manifest import Utterance, make_id
from . import AUDIO, MANIFESTS, elm_members, finalize, make_utt

HF_ID = "UBC-NLP/Casablanca"
CONFIGS = ["Algeria", "Egypt", "Jordan", "Mauritania", "Morocco", "Palestine", "UAE", "Yemen"]
NAME = "casablanca"


def build(
    *,
    configs: list[str] | None = None,
    per_config_limit: int | None = None,
    audio_dir: Path = AUDIO,
    manifests_dir: Path = MANIFESTS,
    n_api: int = 1000,
    n_timing: int = 12,
    seed: int = 0,
    split: str = "test",
    every: int = 1,
    name: str = NAME,
) -> list[Utterance]:
    from datasets import Audio, load_dataset

    from . import keep

    elm = elm_members("casablanca_test") if split == "test" else {}
    utts: list[Utterance] = []
    seen_elm: set[str] = set()
    for cfg in configs or CONFIGS:
        ds = load_dataset(HF_ID, cfg, split=split)
        ds = ds.cast_column("audio", Audio(decode=False))
        n = 0
        for row in tqdm(ds, desc=f"{NAME}/{cfg}", unit="utt"):
            seg_id = str(row["seg_id"])
            if not keep(name, seg_id, every):
                continue
            key = Path(seg_id).stem
            rel = f"{name}/{make_id(name, seg_id)}.wav"
            wav = audio_dir / rel
            arr, sr = hio.decode_audio_bytes(row["audio"]["bytes"])
            if not wav.exists():
                hio.write_wav_16k(wav, arr, sr)
            in_elm = key in elm
            if in_elm:
                seen_elm.add(key)
            ref = elm[key]["text"] if in_elm else (row["transcription"] or "")
            dur = float(row.get("duration") or 0.0) or len(arr) / sr
            utts.append(make_utt(name, seg_id, wav, rel, ref, duration_s=dur, dialect_raw=cfg, in_elm=in_elm))
            n += 1
            if per_config_limit and n >= per_config_limit:
                break
    return finalize(name, utts, elm, seen_elm, manifests_dir=manifests_dir, n_api=n_api, n_timing=n_timing, seed=seed)

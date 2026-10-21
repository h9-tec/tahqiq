"""Casablanca (UBC-NLP/Casablanca, test split, 8 country configs, CC-BY-NC-ND 4.0).

Audio is never redistributed; only our per-utterance outputs are.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from .. import io as hio
from ..manifest import Utterance
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
) -> list[Utterance]:
    from datasets import Audio, load_dataset

    elm = elm_members("casablanca_test")
    utts: list[Utterance] = []
    seen_elm: set[str] = set()
    for cfg in configs or CONFIGS:
        ds = load_dataset(HF_ID, cfg, split="test")
        ds = ds.cast_column("audio", Audio(sampling_rate=16_000, decode=True))
        n = 0
        for row in tqdm(ds, desc=f"{NAME}/{cfg}", unit="utt"):
            seg_id = str(row["seg_id"])
            key = Path(seg_id).stem
            uid_rel = None
            from ..manifest import make_id

            rel = f"{NAME}/{make_id(NAME, seg_id)}.wav"
            wav = audio_dir / rel
            if not wav.exists():
                arr = np.asarray(row["audio"]["array"], dtype=np.float32)
                hio.write_wav_16k(wav, arr, int(row["audio"]["sampling_rate"]))
            in_elm = key in elm
            if in_elm:
                seen_elm.add(key)
            ref = elm[key]["text"] if in_elm else (row["transcription"] or "")
            dur = float(row.get("duration") or 0.0) or len(row["audio"]["array"]) / 16_000
            utts.append(make_utt(NAME, seg_id, wav, rel, ref, duration_s=dur, dialect_raw=cfg, in_elm=in_elm))
            n += 1
            if per_config_limit and n >= per_config_limit:
                break
    return finalize(NAME, utts, elm, seen_elm, manifests_dir=manifests_dir, n_api=n_api, n_timing=n_timing, seed=seed)

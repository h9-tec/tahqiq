"""Common Voice 18.0 Arabic, test split (CC0).

Source: the parquet mirror ``MohamedRashad/common-voice-18-arabic`` (columns
client_id, path, audio@48k, sentence, accent, ...). ``path`` stems are Elm's ids
(``common_voice_ar_<n>``). Pass ``--src`` as the HF dataset id or a local
directory of parquet files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from .. import io as hio
from ..manifest import Utterance, make_id
from . import AUDIO, MANIFESTS, elm_members, finalize, make_utt

NAME = "commonvoice"
DEFAULT_SRC = "MohamedRashad/common-voice-18-arabic"


def build(
    *,
    src: str | Path = DEFAULT_SRC,
    per_config_limit: int | None = None,
    audio_dir: Path = AUDIO,
    manifests_dir: Path = MANIFESTS,
    n_api: int = 1000,
    n_timing: int = 12,
    seed: int = 0,
) -> list[Utterance]:
    from datasets import Audio, load_dataset

    elm = elm_members("commonvoice_test")
    src = str(src)
    if Path(src).is_dir():
        ds = load_dataset("parquet", data_files=str(Path(src) / "test-*.parquet"), split="train")
    else:
        ds = load_dataset(src, split="test")
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000, decode=True))
    utts: list[Utterance] = []
    seen_elm: set[str] = set()
    n = 0
    for row in tqdm(ds, desc=NAME, unit="utt"):
        key = Path(str(row["path"])).stem
        rel = f"{NAME}/{make_id(NAME, key)}.wav"
        wav = audio_dir / rel
        arr = np.asarray(row["audio"]["array"], dtype=np.float32)
        if not wav.exists():
            hio.write_wav_16k(wav, arr, int(row["audio"]["sampling_rate"]))
        in_elm = key in elm
        if in_elm:
            seen_elm.add(key)
        ref = elm[key]["text"] if in_elm else (row["sentence"] or "")
        dur = len(arr) / 16_000
        utts.append(make_utt(NAME, key, wav, rel, ref, duration_s=dur, dialect_raw=(row.get("accent") or ""), in_elm=in_elm))
        n += 1
        if per_config_limit and n >= per_config_limit:
            break
    return finalize(NAME, utts, elm, seen_elm, manifests_dir=manifests_dir, n_api=n_api, n_timing=n_timing, seed=seed)

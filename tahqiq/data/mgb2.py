"""MGB-2 (Aljazeera broadcast, QCRI), test split, from the parquet mirror MohamedRashad/mgb2-arabic.

Columns: audio@16k, transcript, duration, quality, segment_id, recording_id. ``segment_id`` is
exactly Elm's id (``<recording>_spk-NNNN_seg-start:end``), so membership is an exact match.
"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from .. import io as hio
from ..manifest import Utterance, make_id
from . import AUDIO, MANIFESTS, elm_members, finalize, make_utt

NAME = "mgb2"
DEFAULT_SRC = "MohamedRashad/mgb2-arabic"


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

    elm = elm_members("mgb2_test")
    src = str(src)
    if Path(src).is_dir():
        ds = load_dataset("parquet", data_files=str(Path(src) / "test-*.parquet"), split="train")
    else:
        ds = load_dataset(src, split="test")
    ds = ds.cast_column("audio", Audio(decode=False))
    utts: list[Utterance] = []
    seen_elm: set[str] = set()
    n = 0
    for row in tqdm(ds, desc=NAME, unit="utt"):
        seg = str(row["segment_id"])
        rel = f"{NAME}/{make_id(NAME, seg)}.wav"
        wav = audio_dir / rel
        arr, sr = hio.decode_audio_bytes(row["audio"]["bytes"])
        if not wav.exists():
            hio.write_wav_16k(wav, arr, sr)
        in_elm = seg in elm
        if in_elm:
            seen_elm.add(seg)
        ref = elm[seg]["text"] if in_elm else (row["transcript"] or "")
        dur = float(row.get("duration") or 0.0) or len(arr) / sr
        utts.append(make_utt(NAME, seg, wav, rel, ref, duration_s=dur, dialect_raw="", in_elm=in_elm))
        n += 1
        if per_config_limit and n >= per_config_limit:
            break
    return finalize(NAME, utts, elm, seen_elm, manifests_dir=manifests_dir, n_api=n_api, n_timing=n_timing, seed=seed)

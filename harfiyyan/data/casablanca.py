"""Casablanca (UBC-NLP/Casablanca, test split, 8 country configs).

Gated on the Hub (CC-BY-NC-ND 4.0): accept the terms once with the HF
account, then ``HF_TOKEN`` in ``.env``. Audio is never redistributed; only
our per-utterance outputs are.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from .. import io as hio
from ..manifest import Utterance, make_id, map_dialect, sha256_file, write_manifest
from ..normalize.numbers import has_written_number
from ..slices import assign_slices
from . import AUDIO, MANIFESTS, elm_members

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
        for row in tqdm(ds, desc=f"casablanca/{cfg}", unit="utt"):
            seg_id = str(row["seg_id"])
            key = Path(seg_id).stem
            uid = make_id(NAME, seg_id)
            rel = f"{NAME}/{uid}.wav"
            wav = audio_dir / rel
            if not wav.exists():
                arr = np.asarray(row["audio"]["array"], dtype=np.float32)
                hio.write_wav_16k(wav, arr, int(row["audio"]["sampling_rate"]))
            ref = row["transcription"] or ""
            in_elm = key in elm
            if in_elm:
                seen_elm.add(key)
            utts.append(
                Utterance(
                    id=uid,
                    dataset=NAME,
                    split="test",
                    source_id=seg_id,
                    audio_rel=rel,
                    audio_sha256=sha256_file(wav),
                    duration_s=float(row.get("duration") or 0.0),
                    reference=ref,
                    dialect_raw=cfg,
                    dialect=map_dialect(cfg),
                    has_number=has_written_number(ref),
                    elm_member=in_elm,
                )
            )
            n += 1
            if per_config_limit and n >= per_config_limit:
                break
    missing = sorted(set(elm) - seen_elm)
    extra = sum(1 for u in utts if not u.elm_member)
    print(f"casablanca: {len(utts)} utts; elm members matched {len(seen_elm)}/{len(elm)}; "
          f"not in elm: {extra}; elm rows we lack: {len(missing)}")
    assign_slices(utts, n_api=n_api, n_timing=n_timing, seed=seed)
    utts.sort(key=lambda u: u.id)
    write_manifest(manifests_dir / f"{NAME}.jsonl", utts)
    return utts

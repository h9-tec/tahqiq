"""Audio loading and the per-model prediction cache.

Results live at ``results/<model_id>/<dataset>.jsonl``; one line per
utterance, append-only, resumable. The cache is the published replication
artifact, so the schema is small and stable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000


@dataclass
class Prediction:
    id: str
    hyp: str
    wall_s: float
    attempts: int = 1
    error: str | None = None
    run_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def results_path(results_dir: Path, model_id: str, dataset: str) -> Path:
    return results_dir / model_id / f"{dataset}.jsonl"


def load_predictions(results_dir: Path, model_id: str, dataset: str) -> dict[str, Prediction]:
    p = results_path(results_dir, model_id, dataset)
    if not p.exists():
        return {}
    out: dict[str, Prediction] = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["id"]] = Prediction(**d)
    return out


def append_prediction(results_dir: Path, model_id: str, dataset: str, pred: Prediction) -> None:
    p = results_path(results_dir, model_id, dataset)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")


def load_audio_16k_mono(path: str | Path) -> tuple[np.ndarray, float]:
    """Float32 mono at 16 kHz plus duration in seconds. Resamples only if needed."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)
    return audio, duration


def write_wav_16k(path: Path, audio: np.ndarray, sr: int) -> None:
    """Write mono 16 kHz PCM16 wav, resampling from ``sr`` when needed."""
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")

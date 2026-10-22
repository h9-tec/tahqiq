"""Adapter interface shared by open models and vendor APIs.

    model = registry.build("whisper_large_v3_turbo")
    model.load()
    t = model.transcribe_path(Path("audio/x.wav"))            # Transcript(text, wall_s, attempts)
    ts = model.transcribe_paths([Path(...), Path(...)])        # batched; falls back to one by one
    model.unload()

Open models implement ``transcribe(audio)`` on a float32 16 kHz array and, when the
architecture batches well, ``transcribe_batch(audios)``. API adapters override
``transcribe_path`` so the timed region includes upload and network.
"""

from __future__ import annotations

import gc
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

SAMPLE_RATE = 16_000


@dataclass
class Pricing:
    usd_per_min: float
    source_url: str
    as_of: str  # YYYY-MM-DD


@dataclass
class Transcript:
    text: str
    wall_s: float
    attempts: int = 1
    raw: dict | None = None


class ASRModel(ABC):
    model_id: str
    display_name: str = ""
    kind: Literal["open", "api"] = "open"
    hf_id: str | None = None
    api_version: str | None = None
    language_hint: str | None = "ar"
    pricing: Pricing | None = None
    decode_params: dict = {}
    batch_size: int = 1            # >1 only when transcribe_batch is implemented
    max_batch_audio_s: float = 40.0  # utterances longer than this are never batched

    def load(self) -> None:  # noqa: B027
        pass

    def unload(self) -> None:
        for attr in ("model", "processor", "pipe", "asr_model", "client"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def transcribe(self, audio: np.ndarray) -> str:
        raise NotImplementedError(f"{self.model_id} does not transcribe arrays")

    def transcribe_batch(self, audios: list[np.ndarray]) -> list[str]:
        """Default: one at a time. Override for architectures that batch well."""
        return [self.transcribe(a) for a in audios]

    def transcribe_path(self, wav_path: Path) -> Transcript:
        from ..io import load_audio_16k_mono

        audio, _ = load_audio_16k_mono(wav_path)
        t0 = time.perf_counter()
        text = self.transcribe(audio)
        return Transcript(text=text, wall_s=time.perf_counter() - t0)

    def transcribe_paths(self, wav_paths: list[Path]) -> list[Transcript]:
        """Batched transcription; wall time is the batch time divided evenly.

        Long clips (> max_batch_audio_s) go one by one. If the batched call fails,
        every clip in the batch is retried individually so one bad file cannot
        take the others down.
        """
        from ..io import load_audio_16k_mono

        if self.batch_size <= 1 or len(wav_paths) == 1:
            return [self.transcribe_path(p) for p in wav_paths]
        audios = [load_audio_16k_mono(p)[0] for p in wav_paths]
        short_idx = [i for i, a in enumerate(audios) if len(a) <= self.max_batch_audio_s * SAMPLE_RATE]
        out: list[Transcript | None] = [None] * len(wav_paths)
        if short_idx:
            t0 = time.perf_counter()
            try:
                texts = self.transcribe_batch([audios[i] for i in short_idx])
                per = (time.perf_counter() - t0) / len(short_idx)
                for i, text in zip(short_idx, texts):
                    out[i] = Transcript(text=text, wall_s=per)
            except Exception:
                for i in short_idx:
                    out[i] = None
        for i in range(len(wav_paths)):
            if out[i] is None:
                t0 = time.perf_counter()
                text = self.transcribe(audios[i])
                out[i] = Transcript(text=text, wall_s=time.perf_counter() - t0)
        return out  # type: ignore[return-value]

    def reference_nll(self, audio: np.ndarray, text: str) -> float | None:
        """Teacher-forced negative log-likelihood (nats, summed over tokens) of ``text`` given ``audio``.

        Only decoders that can score a given transcript implement it (Whisper, Cohere Transcribe).
        Returns None when unsupported."""
        return None

    def env_info(self) -> dict:
        """Versions worth recording in run.json. Overridden per adapter."""
        info: dict = {}
        try:
            import torch

            info["torch"] = torch.__version__
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                info["cuda"] = torch.version.cuda
        except Exception:
            pass
        try:
            import transformers

            info["transformers"] = transformers.__version__
        except Exception:
            pass
        return info

    @abstractmethod
    def _marker(self) -> None:
        """Subclasses exist; keeps ASRModel abstract."""

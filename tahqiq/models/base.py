"""Adapter interface shared by open models and vendor APIs.

    model = registry.build("whisper_large_v3_turbo")
    model.load()
    t = model.transcribe_path(Path("audio/x.wav"))   # Transcript(text, wall_s, attempts)
    model.unload()

Open models implement ``transcribe(audio)`` on a float32 16 kHz array; the
default ``transcribe_path`` loads the file and times the call. API adapters
override ``transcribe_path`` so the timed region includes upload and network.
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

    def transcribe_path(self, wav_path: Path) -> Transcript:
        from ..io import load_audio_16k_mono

        audio, _ = load_audio_16k_mono(wav_path)
        t0 = time.perf_counter()
        text = self.transcribe(audio)
        return Transcript(text=text, wall_s=time.perf_counter() - t0)

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

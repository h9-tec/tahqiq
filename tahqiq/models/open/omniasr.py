"""Meta Omnilingual ASR (omniASR CTC / LLM families) through the ``omnilingual-asr`` package (fairseq2).

Runs in its own environment (the package pins torch/fairseq2): see scripts/setup_omni_env.sh.
Language is forced to Modern Standard Arabic, Arabic script (``arb_Arab``), the closest
single code the model offers for mixed-dialect test sets.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ..base import SAMPLE_RATE, ASRModel


class OmniASR(ASRModel):
    kind = "open"
    decode_params = {"lang": "arb_Arab"}
    batch_size = 8

    def __init__(self, model_id: str, hf_id: str, display_name: str):
        self.model_id = model_id
        self.hf_id = hf_id
        self.display_name = display_name
        self.card = hf_id.split("/")[-1].replace("-", "_")  # facebook/omniASR-LLM-1B -> omniASR_LLM_1B
        self.pipe = None

    def _marker(self) -> None:
        pass

    def load(self) -> None:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        self.pipe = ASRInferencePipeline(model_card=self.card)

    def transcribe_batch(self, audios: list[np.ndarray]) -> list[str]:
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i, a in enumerate(audios):
                p = Path(td) / f"{i}.wav"
                sf.write(str(p), a.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
                paths.append(str(p))
            out = self.pipe.transcribe(paths, lang=[self.decode_params["lang"]] * len(paths), batch_size=len(paths))
        return [str(t).strip() for t in out]

    def transcribe(self, audio: np.ndarray) -> str:
        return self.transcribe_batch([audio])[0]

    def env_info(self) -> dict:
        info = super().env_info()
        try:
            import omnilingual_asr

            info["omnilingual_asr"] = getattr(omnilingual_asr, "__version__", "?")
        except Exception:
            pass
        return info

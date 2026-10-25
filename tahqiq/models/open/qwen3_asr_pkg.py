"""Qwen3-ASR checkpoints in the original release format (``thinker.*`` weights, transformers 4.57), run through
Qwen's own ``qwen-asr`` package. Covers Qwen/Qwen3-ASR-0.6B/1.7B and the community fine-tunes saved from them;
the Transformers-native ``-hf`` conversions use qwen3_asr.py instead. Language forced to Arabic, which makes the
prompt end in ``language Arabic<asr_text>`` so the output is text only. The package chunks long audio itself.

Needs ``qwen-asr`` (pins transformers==4.57.6): run with
``uv run --no-sync --with qwen-asr --with 'torch==<project torch>'`` (see box_queue ``pip``).
"""

from __future__ import annotations

import numpy as np

from ..base import SAMPLE_RATE, ASRModel


class Qwen3ASRPkg(ASRModel):
    kind = "open"
    decode_params = {"language": "Arabic", "max_new_tokens": 512, "do_sample": False, "runtime": "qwen-asr"}
    batch_size = 16

    def __init__(self, model_id: str, hf_id: str, display_name: str):
        self.model_id = model_id
        self.hf_id = hf_id
        self.display_name = display_name
        self.model = None

    def _marker(self) -> None:
        pass

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        self.model = Qwen3ASRModel.from_pretrained(
            self.hf_id, dtype=torch.bfloat16, device_map="cuda:0",
            max_inference_batch_size=self.batch_size, max_new_tokens=self.decode_params["max_new_tokens"],
        )

    def transcribe_batch(self, audios: list[np.ndarray]) -> list[str]:
        res = self.model.transcribe(audio=[(a.astype(np.float32), SAMPLE_RATE) for a in audios],
                                    language=self.decode_params["language"])
        if len(res) != len(audios):
            raise RuntimeError("qwen-asr returned a different count")
        return [r.text.strip() for r in res]

    def transcribe(self, audio: np.ndarray) -> str:
        return self.transcribe_batch([audio])[0]

    def env_info(self) -> dict:
        info = super().env_info()
        try:
            from importlib.metadata import version

            info["qwen_asr"] = version("qwen-asr")
        except Exception:
            pass
        return info

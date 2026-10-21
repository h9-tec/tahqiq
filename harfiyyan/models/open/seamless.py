"""SeamlessM4T v2 large, speech-to-text task, tgt_lang=arb (Elm reports the best of arb/ary/arz; we fix arb)."""

from __future__ import annotations

import numpy as np

from ..base import SAMPLE_RATE, ASRModel


class Seamless(ASRModel):
    kind = "open"
    decode_params = {"tgt_lang": "arb", "num_beams": 1}

    def __init__(self) -> None:
        self.model_id = "seamless_m4t_v2_large"
        self.hf_id = "facebook/seamless-m4t-v2-large"
        self.display_name = "SeamlessM4T v2 large (arb)"
        self.model = None
        self.processor = None

    def _marker(self) -> None:
        pass

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(self.hf_id, dtype=torch.float16).to("cuda")
        self.model.eval()

    def transcribe(self, audio: np.ndarray) -> str:
        import torch

        inputs = self.processor(audios=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").to("cuda")
        inputs = {k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, tgt_lang="arb", num_beams=1, do_sample=False)
        return self.processor.decode(out[0].tolist(), skip_special_tokens=True).strip()


def seamless_m4t_v2_large() -> Seamless:
    return Seamless()

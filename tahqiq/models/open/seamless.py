"""SeamlessM4T v2 large, speech-to-text, tgt_lang=arb (Elm reports the best of arb/ary/arz; we fix arb). Batched."""

from __future__ import annotations

import numpy as np

from ..base import SAMPLE_RATE, ASRModel


class Seamless(ASRModel):
    kind = "open"
    decode_params = {"tgt_lang": "arb", "num_beams": 1}
    batch_size = 16
    max_batch_audio_s = 30.0

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

    def transcribe_batch(self, audios: list[np.ndarray]) -> list[str]:
        import torch

        inputs = self.processor(audio=list(audios), sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True).to("cuda")
        inputs = {k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, tgt_lang="arb", num_beams=1, do_sample=False)
        return [self.processor.decode(o.tolist(), skip_special_tokens=True).strip() for o in out]

    def transcribe(self, audio: np.ndarray) -> str:
        return self.transcribe_batch([audio])[0]


def seamless_m4t_v2_large() -> Seamless:
    return Seamless()

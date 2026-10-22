"""Voxtral Mini 4B Realtime (and its Arabic fine-tune) via Transformers (>= 5.2).

The model is streaming by design; here it is run offline on whole utterances, greedy,
which is the configuration the model card's benchmarks use at a fixed delay.
"""

from __future__ import annotations

import numpy as np

from ..base import SAMPLE_RATE, ASRModel


class VoxtralRealtime(ASRModel):
    kind = "open"
    decode_params = {"do_sample": False, "max_new_tokens": 512}
    batch_size = 1  # processor is single-audio; the model is cheap enough

    def __init__(self, model_id: str, hf_id: str, display_name: str):
        self.model_id = model_id
        self.hf_id = hf_id
        self.display_name = display_name
        self.model = None
        self.processor = None

    def _marker(self) -> None:
        pass

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(self.hf_id, dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()

    def transcribe(self, audio: np.ndarray) -> str:
        import torch

        inputs = self.processor(audio.astype(np.float32), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        with torch.no_grad():
            out = self.model.generate(**inputs, **self.decode_params)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

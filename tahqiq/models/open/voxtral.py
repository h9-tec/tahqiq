"""Voxtral Mini 3B via Transformers, dedicated transcription mode, language=ar, temperature 0."""

from __future__ import annotations

import numpy as np

from ..base import SAMPLE_RATE, ASRModel


class Voxtral(ASRModel):
    kind = "open"
    decode_params = {"language": "ar", "max_new_tokens": 512, "do_sample": False}

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
        from transformers import AutoProcessor, VoxtralForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.model = VoxtralForConditionalGeneration.from_pretrained(self.hf_id, dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()

    def transcribe(self, audio: np.ndarray) -> str:
        import torch

        inputs = self.processor.apply_transcription_request(
            language="ar", audio=audio, sampling_rate=SAMPLE_RATE, model_id=self.hf_id, format=["WAV"]
        )
        inputs = inputs.to("cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.decode_params["max_new_tokens"], do_sample=False)
        gen = out[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def voxtral_mini_3b() -> Voxtral:
    return Voxtral("voxtral_mini_3b", "mistralai/Voxtral-Mini-3B-2507", "Voxtral Mini 3B")

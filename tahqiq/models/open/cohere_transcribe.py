"""Cohere Transcribe Arabic (open weights, Apache-2.0; transformers >= 5.4). Language forced to ar."""

from __future__ import annotations

import numpy as np

from ..base import SAMPLE_RATE, ASRModel


class CohereTranscribe(ASRModel):
    kind = "open"
    decode_params = {"language": "ar", "max_new_tokens": 256}

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
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(self.hf_id, device_map="cuda", dtype=torch.bfloat16)
        self.model.eval()

    def transcribe(self, audio: np.ndarray) -> str:
        import torch

        inputs = self.processor(audio=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", language="ar")
        chunk_index = inputs.get("audio_chunk_index")
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.decode_params["max_new_tokens"])
        if chunk_index is not None:
            text = self.processor.decode(out, skip_special_tokens=True, audio_chunk_index=chunk_index, language="ar")
        else:
            text = self.processor.decode(out, skip_special_tokens=True)
        if isinstance(text, list):
            text = text[0]
        return str(text).strip()


def cohere_transcribe_arabic() -> CohereTranscribe:
    return CohereTranscribe(
        "cohere_transcribe_arabic_07_2026", "CohereLabs/cohere-transcribe-arabic-07-2026", "Cohere Transcribe Arabic 07-2026"
    )

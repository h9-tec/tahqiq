"""Cohere Transcribe Arabic (open weights, Apache-2.0; transformers >= 5.4). Language forced to ar. Batched for short clips."""

from __future__ import annotations

import numpy as np

from ..base import SAMPLE_RATE, ASRModel


class CohereTranscribe(ASRModel):
    kind = "open"
    decode_params = {"language": "ar", "max_new_tokens": 256}
    batch_size = 16
    max_batch_audio_s = 30.0

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

    def reference_nll(self, audio: np.ndarray, text: str) -> float | None:
        import torch

        if len(audio) > 30 * SAMPLE_RATE or not text.strip():
            return None
        inputs = self.processor(audio=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", language="ar")
        if inputs.get("audio_chunk_index") is not None and len(set(inputs["audio_chunk_index"].tolist())) > 1:
            return None
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        labels = self.processor.tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids.to(self.model.device)
        try:
            with torch.no_grad():
                out = self.model(**{k: v for k, v in inputs.items() if k != "audio_chunk_index"}, labels=labels)
            return float(out.loss) * (labels.shape[1] - 1)
        except Exception:
            return None

    def transcribe_batch(self, audios: list[np.ndarray]) -> list[str]:
        import torch

        inputs = self.processor(audio=list(audios), sampling_rate=SAMPLE_RATE, return_tensors="pt", language="ar")
        if inputs.get("audio_chunk_index") is not None and len(set(inputs["audio_chunk_index"].tolist())) != len(audios):
            raise RuntimeError("chunked input in batch; falling back")
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.decode_params["max_new_tokens"])
        texts = self.processor.decode(out, skip_special_tokens=True)
        if isinstance(texts, str):
            texts = [texts]
        if len(texts) != len(audios):
            raise RuntimeError("batched decode count mismatch; falling back")
        return [str(t).strip() for t in texts]


def cohere_transcribe_arabic() -> CohereTranscribe:
    return CohereTranscribe(
        "cohere_transcribe_arabic_07_2026", "CohereLabs/cohere-transcribe-arabic-07-2026", "Cohere Transcribe Arabic 07-2026"
    )

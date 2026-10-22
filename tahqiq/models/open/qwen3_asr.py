"""Qwen3-ASR (Transformers-native checkpoints, transformers >= 5.13). Language forced to Arabic. Batched."""

from __future__ import annotations

import numpy as np

from ..base import SAMPLE_RATE, ASRModel


class Qwen3ASR(ASRModel):
    kind = "open"
    decode_params = {"language": "Arabic", "max_new_tokens": 512, "do_sample": False}
    batch_size = 16

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
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(self.hf_id, device_map="cuda", dtype=torch.bfloat16)
        self.model.eval()

    def _run(self, audio) -> list[str]:
        import torch

        inputs = self.processor.apply_transcription_request(
            audio=audio, sampling_rate=SAMPLE_RATE, language=self.decode_params["language"]
        ).to(self.model.device, self.model.dtype)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.decode_params["max_new_tokens"], do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        texts = self.processor.decode(gen, return_format="transcription_only")
        if isinstance(texts, str):
            texts = [texts]
        return [t.strip() for t in texts]

    def transcribe(self, audio: np.ndarray) -> str:
        return self._run(audio)[0]

    def transcribe_batch(self, audios: list[np.ndarray]) -> list[str]:
        texts = self._run(list(audios))
        if len(texts) != len(audios):
            raise RuntimeError("batched decode returned a different count; falling back")
        return texts


def qwen3_asr_0_6b() -> Qwen3ASR:
    return Qwen3ASR("qwen3_asr_0.6b", "Qwen/Qwen3-ASR-0.6B-hf", "Qwen3-ASR 0.6B")


def qwen3_asr_1_7b() -> Qwen3ASR:
    return Qwen3ASR("qwen3_asr_1.7b", "Qwen/Qwen3-ASR-1.7B-hf", "Qwen3-ASR 1.7B")

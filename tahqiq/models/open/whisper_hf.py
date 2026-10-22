"""Whisper via Transformers. Forces language=ar, task=transcribe, greedy decoding. Batched for clips <= 30 s."""

from __future__ import annotations

import numpy as np

from ..base import SAMPLE_RATE, ASRModel


class WhisperHF(ASRModel):
    kind = "open"
    decode_params = {"language": "ar", "task": "transcribe", "num_beams": 1, "max_new_tokens": 440}
    batch_size = 32
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
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.hf_id, dtype=self._dtype, low_cpu_mem_usage=True
        ).to(self._device)
        self.model.eval()

    def _generate(self, audios: list[np.ndarray]) -> list[str]:
        import torch

        feats = self.processor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        feats = feats.to(self._device, dtype=self._dtype)
        with torch.no_grad():
            ids = self.model.generate(feats, do_sample=False, **self.decode_params)
        return [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]

    def transcribe(self, audio: np.ndarray) -> str:
        if len(audio) > 30 * SAMPLE_RATE:
            return self._transcribe_long(audio)
        return self._generate([audio])[0]

    def transcribe_batch(self, audios: list[np.ndarray]) -> list[str]:
        return self._generate(audios)

    def _transcribe_long(self, audio: np.ndarray) -> str:
        import torch
        from transformers import pipeline

        if not hasattr(self, "pipe") or self.pipe is None:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                chunk_length_s=30,
                dtype=self._dtype,
                device=0 if self._device == "cuda" else -1,
            )
        with torch.no_grad():
            out = self.pipe(
                {"raw": audio, "sampling_rate": SAMPLE_RATE},
                generate_kwargs={"language": "ar", "task": "transcribe", "num_beams": 1},
            )
        return out["text"].strip()


def whisper_large_v3() -> WhisperHF:
    return WhisperHF("whisper_large_v3", "openai/whisper-large-v3", "Whisper large-v3")


def whisper_large_v3_turbo() -> WhisperHF:
    return WhisperHF("whisper_large_v3_turbo", "openai/whisper-large-v3-turbo", "Whisper large-v3-turbo")

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
        self._fix_generation_config()

    # (d_model, decoder_layers, num_mel_bins) -> the OpenAI checkpoint whose generation config matches
    _BASES = {(384, 4, 80): "openai/whisper-tiny", (512, 6, 80): "openai/whisper-base", (768, 12, 80): "openai/whisper-small",
              (1024, 24, 80): "openai/whisper-medium", (1280, 32, 80): "openai/whisper-large-v2",
              (1280, 32, 128): "openai/whisper-large-v3", (1280, 4, 128): "openai/whisper-large-v3-turbo"}

    def _fix_generation_config(self) -> None:
        """Older fine-tunes ship a generation config without lang_to_id/task_to_id, so generate(language=...)
        refuses it. Take the config of the OpenAI checkpoint with the same shape (special tokens are shared)."""
        gc = self.model.generation_config
        if getattr(gc, "lang_to_id", None) and getattr(gc, "task_to_id", None):
            return
        from transformers import GenerationConfig

        c = self.model.config
        base = self._BASES.get((c.d_model, c.decoder_layers, c.num_mel_bins))
        if base is None:
            raise RuntimeError(f"{self.hf_id}: generation config lacks lang_to_id and no OpenAI base matches its shape")
        self.model.generation_config = GenerationConfig.from_pretrained(base)
        self.generation_config_from = base
        print(f"[whisper] {self.hf_id}: generation config taken from {base}")

    def _generate(self, audios: list[np.ndarray]) -> list[str]:
        import torch

        feats = self.processor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        feats = feats.to(self._device, dtype=self._dtype)
        with torch.no_grad():
            ids = self.model.generate(feats, do_sample=False, **self.decode_params)
        return [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]

    def env_info(self) -> dict:
        info = super().env_info()
        if getattr(self, "generation_config_from", None):
            info["generation_config_from"] = self.generation_config_from
        return info

    def transcribe(self, audio: np.ndarray) -> str:
        if len(audio) > 30 * SAMPLE_RATE:
            return self._transcribe_long(audio)
        return self._generate([audio])[0]

    def transcribe_batch(self, audios: list[np.ndarray]) -> list[str]:
        return self._generate(audios)

    def reference_nll(self, audio: np.ndarray, text: str) -> float | None:
        import torch

        if len(audio) > 30 * SAMPLE_RATE or not text.strip():
            return None
        feats = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(self._device, dtype=self._dtype)
        tok = self.processor.tokenizer
        prefix = tok.prefix_tokens if hasattr(tok, "prefix_tokens") else []
        tok.set_prefix_tokens(language="ar", task="transcribe", predict_timestamps=False)
        ids = tok(text, add_special_tokens=True).input_ids   # <|startoftranscript|><|ar|><|transcribe|><|notimestamps|> ... <|endoftext|>
        labels = torch.tensor([ids], device=self._device)
        with torch.no_grad():
            out = self.model(input_features=feats, labels=labels)
        # loss is mean over label tokens; return the sum so lengths are comparable between ref and hyp
        return float(out.loss) * (labels.shape[1] - 1)

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

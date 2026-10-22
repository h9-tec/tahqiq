"""Audar-ASR V1 (Flash / Turbo), run exactly as the model card documents: custom modeling code under
transformers 4.57, Arabic system prompt, greedy decoding, `language <X><asr_text>` prefix stripped.

The repositories ship a Qwen3-ASR-derived config class that breaks under transformers >= 5, so these
models get their own pod with ``transformers==4.57.6`` (see data/candidates.yaml ``transformers_spec``).
"""

from __future__ import annotations

import re

import numpy as np

from ..base import SAMPLE_RATE, ASRModel

SYSTEM_PROMPT = "فرّغ الكلام العربي التالي."   # "Transcribe the following Arabic speech." (card's auto-dialect prompt)
_PREFIX = re.compile(r"^\s*language\s+[A-Za-z]+\s*(?:<asr_text>)?\s*")
CHUNK_S = 30.0


class AudarASR(ASRModel):
    kind = "open"
    decode_params = {"system_prompt": SYSTEM_PROMPT, "num_beams": 1, "max_new_tokens": 440}
    batch_size = 1

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
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.hf_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda:0"
        ).eval()
        conv = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": [{"type": "audio"}]}]
        self._prompt = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

    def _one(self, audio: np.ndarray) -> str:
        import torch

        inputs = self.processor(text=self._prompt, audio=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(self.model.device)
        inputs["input_features"] = inputs["input_features"].to(self.model.dtype)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.decode_params["max_new_tokens"], do_sample=False)
        ids = out.sequences if hasattr(out, "sequences") else out   # the custom generate may return a GenerateOutput
        hyp = self.processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return _PREFIX.sub("", hyp).strip()

    def transcribe(self, audio: np.ndarray) -> str:
        n = int(CHUNK_S * SAMPLE_RATE)
        if len(audio) <= n:
            return self._one(audio)
        parts = [self._one(audio[i:i + n]) for i in range(0, len(audio), n) if len(audio[i:i + n]) > SAMPLE_RATE // 2]
        return " ".join(p for p in parts if p).strip()


def audar_flash() -> AudarASR:
    return AudarASR("audarai_Audar-ASR-V1-Flash", "audarai/Audar-ASR-V1-Flash", "Audar-ASR V1 Flash")


def audar_turbo() -> AudarASR:
    return AudarASR("audarai_Audar-ASR-V1-Turbo", "audarai/Audar-ASR-V1-Turbo", "Audar-ASR V1 Turbo")

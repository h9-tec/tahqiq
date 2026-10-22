"""Gemma 4 (E2B/E4B/12B) and Gemma 3n audio input via the chat template, prompted to transcribe.

Prompt is fixed and published; thinking mode off; greedy. Output is taken verbatim.
"""

from __future__ import annotations

import numpy as np

from ..base import SAMPLE_RATE, ASRModel

PROMPT = "Transcribe this Arabic audio exactly as spoken. Output only the transcription, in Arabic script, with no translation and no commentary."


class GemmaAudio(ASRModel):
    kind = "open"
    decode_params = {"prompt": PROMPT, "do_sample": False, "max_new_tokens": 512}
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
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        try:
            from transformers import AutoModelForMultimodalLM as M
        except ImportError:  # older naming
            from transformers import AutoModelForImageTextToText as M
        self.model = M.from_pretrained(self.hf_id, dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()

    def transcribe(self, audio: np.ndarray) -> str:
        import torch

        messages = [{"role": "user", "content": [
            {"type": "audio", "audio": audio.astype(np.float32), "sampling_rate": SAMPLE_RATE},
            {"type": "text", "text": PROMPT},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        n = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.decode_params["max_new_tokens"])
        text = self.processor.batch_decode(out[:, n:], skip_special_tokens=True)[0]
        return text.strip().strip('"').strip()

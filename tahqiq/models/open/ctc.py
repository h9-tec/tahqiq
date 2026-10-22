"""Generic CTC adapter (wav2vec2 / HuBERT / WavLM / w2v-BERT fine-tunes), greedy decoding."""

from __future__ import annotations

import numpy as np

from ..base import SAMPLE_RATE, ASRModel


class CTCModel(ASRModel):
    kind = "open"
    decode_params = {"decoding": "greedy"}

    def __init__(self, model_id: str, hf_id: str, display_name: str, *, target_lang: str | None = None):
        self.model_id = model_id
        self.hf_id = hf_id
        self.display_name = display_name
        self.target_lang = target_lang
        self.model = None
        self.processor = None

    def _marker(self) -> None:
        pass

    def load(self) -> None:
        from transformers import AutoModelForCTC, AutoProcessor

        kw = {"target_lang": self.target_lang} if self.target_lang else {}
        self.processor = AutoProcessor.from_pretrained(self.hf_id, **kw)
        self.model = AutoModelForCTC.from_pretrained(self.hf_id, **({**kw, "ignore_mismatched_sizes": True} if kw else {})).to("cuda")
        self.model.eval()

    def transcribe(self, audio: np.ndarray) -> str:
        import torch

        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)[0]
        return self.processor.decode(ids).strip()

"""Meta MMS-1b-all with the Arabic adapter (CTC, greedy)."""

from __future__ import annotations

import numpy as np

from ..base import SAMPLE_RATE, ASRModel


class MMS(ASRModel):
    kind = "open"
    decode_params = {"target_lang": "ara", "decoding": "greedy"}

    def __init__(self) -> None:
        self.model_id = "mms_1b_all"
        self.hf_id = "facebook/mms-1b-all"
        self.display_name = "MMS-1b-all (ara)"
        self.model = None
        self.processor = None

    def _marker(self) -> None:
        pass

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self.processor = AutoProcessor.from_pretrained(self.hf_id, target_lang="ara")
        self.model = Wav2Vec2ForCTC.from_pretrained(self.hf_id, target_lang="ara", ignore_mismatched_sizes=True).to("cuda")
        self.model.eval()

    def transcribe(self, audio: np.ndarray) -> str:
        import torch

        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)[0]
        return self.processor.decode(ids).strip()


def mms_1b_all() -> MMS:
    return MMS()

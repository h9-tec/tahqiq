"""Jais-ASR (IFM/Jais-ASR): Qwen3-Omni audio encoder + Jais/K2 decoder, custom modeling code, run as the card's
"Batch Transcription" recipe: chat-template prompt, left padding, greedy decoding, ``language="ar"`` (Arabic incl.
all 19 dialects; MSA/dialect still chosen by the model), no diacritics, no code-switch spans, and the repo's own
``post_process_transcription`` (strips the language/dialect tag; ``[IFM-NoTranscript]`` becomes an empty hypothesis).

The repo's remote code is pinned to the revision that was reviewed (no I/O, no network; version shims only).
Card: transformers 5.16.1, soundfile, soxr.
"""

from __future__ import annotations

import numpy as np

from ..base import SAMPLE_RATE, ASRModel

# remote code reviewed at these revisions; a new revision needs a new review
REVISION = {"IFM/Jais-ASR": "1323995fa189c479369a3929be0e95039eba5cba"}


class JaisASR(ASRModel):
    kind = "open"
    decode_params = {"language": "ar", "diacritics": False, "code_switch": False, "do_sample": False, "max_new_tokens": 256}
    # batch 1: batched decoding (the card's left-padded batch recipe) changed 5 of 32 hypotheses vs one file at a time
    batch_size = 1

    def __init__(self, model_id: str, hf_id: str, display_name: str):
        self.model_id = model_id
        self.hf_id = hf_id
        self.display_name = display_name
        self.revision = REVISION.get(hf_id)
        self.model = None
        self.processor = None

    def _marker(self) -> None:
        pass

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        if self.revision is None:
            raise RuntimeError(f"{self.hf_id}: remote code not reviewed; add its revision to REVISION")
        kw = {"trust_remote_code": True, "revision": self.revision}
        self.processor = AutoProcessor.from_pretrained(self.hf_id, **kw)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_id, dtype=torch.bfloat16, device_map="cuda:0", **kw).eval()
        self._prompt = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "audio"}]}], tokenize=False, add_generation_prompt=True)
        self.stats = {"refused": 0, "truncated": 0}

    def transcribe_batch(self, audios: list[np.ndarray]) -> list[str]:
        import torch

        inputs = self.processor(text=[self._prompt] * len(audios), audio=[a.astype(np.float32) for a in audios],
                                sampling_rate=[SAMPLE_RATE] * len(audios), padding=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.decode_params["max_new_tokens"],
                                      language=self.decode_params["language"])
        res = self.processor.post_process_transcription(out, input_lengths=inputs["input_ids"].shape[1])
        if len(res) != len(audios):
            raise RuntimeError("batched decode count mismatch; falling back")
        for r in res:
            self.stats["refused"] += bool(getattr(r, "refused", False))
            self.stats["truncated"] += bool(getattr(r, "truncated", False))
        return [(r.text or "").strip() for r in res]

    def transcribe(self, audio: np.ndarray) -> str:
        return self.transcribe_batch([audio])[0]

    def env_info(self) -> dict:
        info = super().env_info()
        info["revision"] = self.revision
        info["postprocess_counts"] = getattr(self, "stats", None)
        return info

"""Nawah-ASR (oddadmix/Nawah-*): from-scratch FastConformer-TDT in plain torch, loaded with the repo's own
``parakeet_model.py`` / ``fastconformer.py`` / ``parakeet_data.py`` (no transformers or NeMo class exists for it).

Decoding is the repo's ``transcribe()``: greedy TDT, one utterance at a time, as on the card. The checkpoint is
read with ``weights_only=True`` (the repo's ``load()`` unpickles it with ``weights_only=False``); the model is
then built from the checkpoint's own dimensions, exactly as ``load()`` does.

Needs ``torchaudio`` (kaldi fbank) and ``tokenizers``: run with ``--with torchaudio`` (see box_queue ``pip``).
"""

from __future__ import annotations

import importlib
import json
import os
import sys

import numpy as np

from ..base import ASRModel

CODE_FILES = ("parakeet_model.py", "fastconformer.py", "parakeet_data.py")


class NawahTDT(ASRModel):
    kind = "open"
    decode_params = {"decoder": "tdt_greedy (repo transcribe(), max_sym_per_frame=10)", "runtime": "repo code parakeet_model.py"}

    def __init__(self, model_id: str, hf_id: str, display_name: str):
        self.model_id = model_id
        self.hf_id = hf_id
        self.display_name = display_name
        self.model = None
        self.tok = None
        self.P = None

    def _marker(self) -> None:
        pass

    def load(self) -> None:
        import torch
        from huggingface_hub import hf_hub_download, snapshot_download
        from tokenizers import Tokenizer

        tok_env = os.environ.get("HF_TOKEN") or None
        cfg = json.loads(open(hf_hub_download(self.hf_id, "config.json", token=tok_env), encoding="utf-8").read())
        weights = cfg["weights_file"]
        tok_file = cfg.get("tokenizer", {}).get("file", "tokenizer_bpe1024.json")
        snap = snapshot_download(self.hf_id, allow_patterns=[*CODE_FILES, weights, tok_file], token=tok_env)
        if snap not in sys.path:
            sys.path.insert(0, snap)
        self.P = importlib.import_module("parakeet_model")
        importlib.import_module("parakeet_data")   # transcribe() imports it lazily
        self.tok = Tokenizer.from_file(os.path.join(snap, tok_file))
        self.model = self._build(torch.load(os.path.join(snap, weights), map_location="cpu", weights_only=True))
        self.model = self.model.eval().to("cuda")
        self.weights_file = weights

    def _build(self, sd: dict):
        """parakeet_model.load() without the unpickling: dimensions read from the state dict."""
        sd = sd["model"] if "model" in sd else sd
        aux = any(k.startswith("ctc_head.") for k in sd)
        m = self.P.ParakeetTDT(
            vocab=sd["predictor.embed.weight"].shape[0],
            joint_dim=sd["transcriber.proj.weight"].shape[0],
            pred_dim=sd["predictor.embed.weight"].shape[1],
            aux_ctc=aux,
            d=(sd["ctc_head.weight"] if aux else sd["transcriber.proj.weight"]).shape[1],
            layers=1 + max(int(k.split(".")[3]) for k in sd if k.startswith("transcriber.enc.blocks.")),
            heads=sd["transcriber.enc.blocks.0.attn.rel"].shape[0],
            ffn=sd["transcriber.enc.blocks.0.ff1.net.1.weight"].shape[0],
            sub_channels=sd["transcriber.enc.embed.conv.0.0.weight"].shape[0],
        )
        m.load_state_dict(sd)
        return m

    def transcribe(self, audio: np.ndarray) -> str:
        return str(self.P.transcribe(self.model, audio.astype(np.float32), self.tok)).strip()

    def env_info(self) -> dict:
        info = super().env_info()
        info["code_files"] = list(CODE_FILES)
        info["weights_file"] = getattr(self, "weights_file", None)
        try:
            import torchaudio

            info["torchaudio"] = torchaudio.__version__
        except Exception:
            pass
        return info

"""Pick an adapter for an arbitrary Hub model id from its config.

``detect(hf_id)`` reads config.json and the file list (no weights) and returns
a Decision: which adapter, which interpreter (main env or NeMo env), the
estimated weight size, and whether a human must look first (custom code,
unknown architecture, too large).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from huggingface_hub import HfApi, hf_hub_download

from .base import ASRModel

MAX_WEIGHT_GB = 24.0

ADAPTERS = {
    # model_type → (adapter module, class name)
    "whisper": (".open.whisper_hf", "WhisperHF"),
    "qwen3_asr": (".open.qwen3_asr", "Qwen3ASR"),
    "cohere_asr": (".open.cohere_transcribe", "CohereTranscribe"),
    "wav2vec2": (".open.ctc", "CTCModel"),
    "hubert": (".open.ctc", "CTCModel"),
    "wavlm": (".open.ctc", "CTCModel"),
    "wav2vec2-bert": (".open.ctc", "CTCModel"),
    "voxtral": (".open.voxtral", "Voxtral"),
    "voxtral_realtime": (".open.voxtral_realtime", "VoxtralRealtime"),
    "gemma4": (".open.gemma_audio", "GemmaAudio"),
    "gemma3n": (".open.gemma_audio", "GemmaAudio"),
    "omni": (".open.omniasr", "OmniASR"),      # facebook/omniASR-* (fairseq2 package, own env)
}


@dataclass
class Decision:
    hf_id: str
    adapter: str = ""          # key in ADAPTERS, or "nemo"
    env: str = "main"          # "main" | "nemo"
    weight_gb: float = 0.0
    model_type: str = ""
    needs_review: bool = False
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def detect(hf_id: str, *, revision: str | None = None) -> Decision:
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    d = Decision(hf_id=hf_id)
    info = api.model_info(hf_id, revision=revision, files_metadata=True)
    files = [s.rfilename for s in info.siblings or []]
    d.weight_gb = round(sum((s.size or 0) for s in info.siblings or []
                            if s.rfilename.endswith((".safetensors", ".bin", ".pt", ".nemo", ".ckpt"))) / 1e9, 2)
    if hf_id.startswith("facebook/omniASR-"):
        d.adapter, d.env, d.model_type = "omni", "omni", "omniasr"
    elif any(f.endswith(".nemo") for f in files):
        d.adapter, d.env = "nemo", "nemo"
    elif "config.json" in files:
        p = hf_hub_download(hf_id, "config.json", revision=revision, token=os.environ.get("HF_TOKEN") or None)
        cfg = json.loads(open(p, encoding="utf-8").read())
        d.model_type = str(cfg.get("model_type", ""))
        if cfg.get("auto_map") or any(f.endswith(".py") for f in files):
            if os.environ.get("TAHQIQ_TRUST_REMOTE_CODE") == "1":
                d.reasons.append("custom code allowed by TAHQIQ_TRUST_REMOTE_CODE=1 (isolated pod, no secrets)")
            else:
                d.needs_review = True
                d.reasons.append("repository ships custom code (trust_remote_code); needs manual review")
        if d.model_type in ADAPTERS:
            d.adapter = d.model_type
        else:
            d.needs_review = True
            d.reasons.append(f"unsupported architecture model_type={d.model_type!r}")
    else:
        d.needs_review = True
        d.reasons.append("no config.json and no .nemo file")
    if d.weight_gb > MAX_WEIGHT_GB:
        d.needs_review = True
        d.reasons.append(f"weights {d.weight_gb} GB exceed the {MAX_WEIGHT_GB} GB automatic limit")
    if getattr(info, "gated", False):
        d.reasons.append("gated repo: the worker's token must have access")
    return d


def build_from_hub(model_id: str, hf_id: str, display_name: str | None = None, *, adapter: str | None = None) -> ASRModel:
    """Instantiate the adapter for a Hub model. ``model_id`` is the slug used in results/."""
    import importlib

    dec = detect(hf_id) if adapter is None else Decision(hf_id=hf_id, adapter=adapter)
    name = display_name or hf_id
    if dec.adapter == "nemo":
        from .open.nemo_fastconformer import NemoFastConformer

        return NemoFastConformer(model_id, hf_id, name)
    if dec.adapter not in ADAPTERS:
        raise ValueError(f"no adapter for {hf_id}: {dec.reasons}")
    mod, cls = ADAPTERS[dec.adapter]
    klass = getattr(importlib.import_module(mod, __package__), cls)
    if dec.adapter in ("wav2vec2", "hubert", "wavlm", "wav2vec2-bert"):
        target = "ara" if "mms" in hf_id.lower() else None
        return klass(model_id, hf_id, name, target_lang=target)
    return klass(model_id, hf_id, name)

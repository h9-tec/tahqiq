"""Production profile of a system: what the Hub says about it plus what we declare by hand.

``hub_profile(hf_id)``   license, release date, weight size, parameter count, architecture (HF API, no download)
``declared(model_id)``   streaming / timestamps / max audio / languages from data/models.yaml (hand-maintained,
                         each entry cites the model card; community submissions start as "unknown")
``profile(model_id, hf_id)`` merges both.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS_YAML = ROOT / "data" / "models.yaml"

DECLARED_FIELDS = ("streaming", "timestamps", "max_audio", "languages", "arabic_only", "declared_source", "basis", "streaming_note", "output_note")


_WEIGHT_EXTS = (".safetensors", ".nemo", ".bin", ".pt", ".ckpt")
# training leftovers that are not the model: optimizer/scheduler state, intermediate checkpoints, trainer logs
_TRAINING = re.compile(r"(^|/)(optimizer|scheduler|rng_state|scaler|training_args|trainer_state)[^/]*$"
                       r"|(^|/)(checkpoint-\d+|checkpoints?|runs|train_state|last-checkpoint)/")
_VARIANT = re.compile(r"\.(fp32|fp16|bf16)(-\d+-of-\d+)?\.[a-z]+$")   # transformers weight variants: extra copies
_CANONICAL = re.compile(r"^(model|pytorch_model)(-\d+-of-\d+)?\.(safetensors|bin)$")
_LANG_ADAPTER = re.compile(r"^adapter\.[^/]+\.(safetensors|bin)$")     # MMS per-language adapters, not the base model


def weight_bytes(files: list[tuple[str, int]]) -> int:
    """Size of ONE copy of the weights: repos often carry the same model as safetensors + bin, fp32 + fp16
    variants, consolidated + sharded files, or training checkpoints; summing everything over-counts (one repo
    showed 113 GB for a 1.5 GB model)."""
    w = [(n, b) for n, b in files if n.endswith(_WEIGHT_EXTS) and not _TRAINING.search(n) and not _LANG_ADAPTER.match(n)]
    top = [f for f in w if "/" not in f[0]] or w
    for ext in _WEIGHT_EXTS:
        g = [f for f in top if f[0].endswith(ext)]
        if not g:
            continue
        canonical = [f for f in g if _CANONICAL.match(f[0])]   # also drops Mistral's consolidated.* and stray copies
        g = canonical or [f for f in g if not _VARIANT.search(f[0])] or g
        return sum(b for _, b in g)
    return 0


def hub_profile(hf_id: str | None) -> dict:
    if not hf_id:
        return {}
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    try:
        info = api.model_info(hf_id, files_metadata=True)
    except Exception as e:  # noqa: BLE001
        return {"hub_error": f"{type(e).__name__}: {str(e)[:120]}"}
    weights = weight_bytes([(s.rfilename, s.size or 0) for s in info.siblings or []])
    params = None
    st = getattr(info, "safetensors", None)
    if st and getattr(st, "total", None):
        params = int(st.total)
    card = getattr(info, "card_data", None)
    license_ = None
    if card is not None:
        license_ = card.get("license") if hasattr(card, "get") else getattr(card, "license", None)
    if not license_:
        for t in info.tags or []:
            if t.startswith("license:"):
                license_ = t.split(":", 1)[1]
    return {
        "license": license_,
        "created_at": info.created_at.date().isoformat() if getattr(info, "created_at", None) else None,
        "last_modified": info.last_modified.date().isoformat() if getattr(info, "last_modified", None) else None,
        "weights_gb": round(weights / 1e9, 2),
        "params_m": round(params / 1e6) if params else None,
        "model_type": (info.config or {}).get("model_type") if getattr(info, "config", None) else None,
        "gated": bool(getattr(info, "gated", False)),
        "downloads": getattr(info, "downloads", None),
    }


def declared(model_id: str) -> dict:
    data = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8")) or {}
    m = (data.get("models") or {}).get(model_id) or {}
    out = {k: m.get(k, None if k in ("basis", "streaming_note", "output_note") else "unknown") for k in DECLARED_FIELDS}
    if m.get("params"):
        out["params_label"] = m["params"]
    return out


def profile(model_id: str, hf_id: str | None) -> dict:
    return {**hub_profile(hf_id), **declared(model_id)}

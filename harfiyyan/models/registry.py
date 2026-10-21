"""Single source of truth for which systems are on the board.

``REGISTRY`` maps model_id → zero-arg factory. Factories import heavy
dependencies lazily so listing the registry never needs torch or vendor SDKs.
"""

from __future__ import annotations

from typing import Callable

from .base import ASRModel


def _open(path: str, name: str) -> Callable[[], ASRModel]:
    def factory() -> ASRModel:
        import importlib

        return getattr(importlib.import_module(path, __package__), name)()

    return factory


REGISTRY: dict[str, Callable[[], ASRModel]] = {
    # Open models
    "whisper_large_v3": _open(".open.whisper_hf", "whisper_large_v3"),
    "whisper_large_v3_turbo": _open(".open.whisper_hf", "whisper_large_v3_turbo"),
}

# Vendor API adapters exist under models/api/ and are tested, but are not on the
# board for v0 (decision 2026-10-21: open models only). Opt in explicitly:
API_REGISTRY: dict[str, Callable[[], ASRModel]] = {
    "deepgram_nova3": _open(".api.deepgram", "deepgram_nova3"),
    "elevenlabs_scribe": _open(".api.elevenlabs", "elevenlabs_scribe"),
    "openai_gpt4o_transcribe": _open(".api.openai", "openai_gpt4o_transcribe"),
}


def list_models() -> list[str]:
    return list(REGISTRY)


def build(model_id: str) -> ASRModel:
    if model_id in REGISTRY:
        return REGISTRY[model_id]()
    if model_id in API_REGISTRY:
        return API_REGISTRY[model_id]()
    raise KeyError(f"unknown model_id {model_id!r}; known: {list_models()} + api: {list(API_REGISTRY)}")

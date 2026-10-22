"""ElevenLabs Scribe speech-to-text. Model id overridable via ELEVENLABS_STT_MODEL."""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from ._http import HTTPAdapter, require_env


class ElevenLabsScribe(HTTPAdapter):
    model_id = "elevenlabs_scribe"
    display_name = "ElevenLabs Scribe"
    language_hint = "ara"

    def __init__(self) -> None:
        self.api_version = os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v1")
        self.decode_params = {
            "model_id": self.api_version,
            "language_code": "ara",
            "tag_audio_events": "false",
            "diarize": "false",
        }

    def _request(self, client: httpx.Client, wav_bytes: bytes, wav_path: Path) -> httpx.Response:
        return client.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": require_env("ELEVENLABS_API_KEY")},
            data=self.decode_params,
            files={"file": (wav_path.name, wav_bytes, "audio/wav")},
        )

    def _parse(self, payload: dict) -> str:
        return payload.get("text", "")


def elevenlabs_scribe() -> ElevenLabsScribe:
    return ElevenLabsScribe()

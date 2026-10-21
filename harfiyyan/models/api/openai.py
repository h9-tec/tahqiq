"""OpenAI audio transcriptions endpoint. Model overridable via OPENAI_STT_MODEL."""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from ._http import HTTPAdapter, require_env


class OpenAITranscribe(HTTPAdapter):
    model_id = "openai_gpt4o_transcribe"
    display_name = "OpenAI gpt-4o-transcribe"
    language_hint = "ar"

    def __init__(self) -> None:
        self.api_version = os.environ.get("OPENAI_STT_MODEL", "gpt-4o-transcribe")
        self.decode_params = {"model": self.api_version, "language": "ar", "response_format": "json", "temperature": "0"}

    def _request(self, client: httpx.Client, wav_bytes: bytes, wav_path: Path) -> httpx.Response:
        return client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {require_env('OPENAI_API_KEY')}"},
            data=self.decode_params,
            files={"file": (wav_path.name, wav_bytes, "audio/wav")},
        )

    def _parse(self, payload: dict) -> str:
        return payload.get("text", "")


def openai_gpt4o_transcribe() -> OpenAITranscribe:
    return OpenAITranscribe()

"""Deepgram pre-recorded API, Nova-3, language=ar, no smart formatting."""

from __future__ import annotations

from pathlib import Path

import httpx

from ._http import HTTPAdapter, require_env


class Deepgram(HTTPAdapter):
    model_id = "deepgram_nova3"
    display_name = "Deepgram Nova-3"
    api_version = "nova-3"
    language_hint = "ar"
    decode_params = {"model": "nova-3", "language": "ar", "smart_format": "false", "punctuate": "false"}

    def _request(self, client: httpx.Client, wav_bytes: bytes, wav_path: Path) -> httpx.Response:
        return client.post(
            "https://api.deepgram.com/v1/listen",
            params=self.decode_params,
            headers={"Authorization": f"Token {require_env('DEEPGRAM_API_KEY')}", "Content-Type": "audio/wav"},
            content=wav_bytes,
        )

    def _parse(self, payload: dict) -> str:
        return payload["results"]["channels"][0]["alternatives"][0]["transcript"]


def deepgram_nova3() -> Deepgram:
    return Deepgram()

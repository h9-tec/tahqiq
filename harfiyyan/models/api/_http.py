"""Shared HTTP plumbing for vendor adapters: env keys, retries, timing."""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ..base import ASRModel, Transcript

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class RetryableHTTP(Exception):
    pass


def _is_retryable(e: BaseException) -> bool:
    return isinstance(e, (RetryableHTTP, httpx.TransportError, httpx.TimeoutException))


def require_env(name: str) -> str:
    v = os.environ.get(name, "")
    if not v:
        raise RuntimeError(f"{name} is not set (put it in .env; keys are never read from files in the repo)")
    return v


class HTTPAdapter(ASRModel):
    """Base for REST vendors. Subclasses implement ``_request`` and ``_parse``."""

    kind = "api"
    timeout_s = 120.0
    max_attempts = 5

    def _marker(self) -> None:
        pass

    def load(self) -> None:
        self.client = httpx.Client(timeout=self.timeout_s)

    def _request(self, client: httpx.Client, wav_bytes: bytes, wav_path: Path) -> httpx.Response:
        raise NotImplementedError

    def _parse(self, payload: dict) -> str:
        raise NotImplementedError

    def transcribe_path(self, wav_path: Path) -> Transcript:
        wav_bytes = wav_path.read_bytes()
        attempts = {"n": 0}

        @retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential_jitter(initial=1, max=30),
            reraise=True,
        )
        def _go() -> dict:
            attempts["n"] += 1
            r = self._request(self.client, wav_bytes, wav_path)
            if r.status_code in RETRY_STATUS:
                raise RetryableHTTP(f"HTTP {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            return r.json()

        t0 = time.perf_counter()
        payload = _go()
        wall = time.perf_counter() - t0
        return Transcript(text=self._parse(payload).strip(), wall_s=wall, attempts=attempts["n"], raw=None)

    def env_info(self) -> dict:
        return {"httpx": httpx.__version__, "api_version": self.api_version}

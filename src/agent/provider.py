"""Replaceable LLM provider interface and an OpenAI-compatible implementation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol
from urllib import error, request


class LLMProvider(Protocol):
    """Minimal interface required by the grounded reply generator."""

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""


class LLMProviderError(RuntimeError):
    """Raised when the configured LLM provider cannot generate a reply."""


@dataclass
class OpenAIProvider:
    """Small OpenAI Chat Completions client with environment-based settings."""

    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    timeout: int = 30

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        self.model = self.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = self.base_url or os.getenv(
            "OPENAI_BASE_URL",
            "https://api.openai.com/v1/chat/completions",
        )

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.2,
        }

        http_request = request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(
                http_request,
                timeout=self.timeout,
            ) as response:
                response_data = json.loads(
                    response.read().decode("utf-8")
                )

        except error.HTTPError as exc:
            error_body = exc.read().decode(
                "utf-8",
                errors="replace",
            )
            raise LLMProviderError(
                f"LLM provider request failed: HTTP {exc.code}: {error_body}"
            ) from exc

        except (
            error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            raise LLMProviderError(
                "LLM provider request failed"
            ) from exc

        try:
            content = response_data["choices"][0]["message"]["content"]

        except (
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            raise LLMProviderError(
                "LLM provider returned an unexpected response"
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError(
                "LLM provider returned an empty response"
            )

        return content.strip()
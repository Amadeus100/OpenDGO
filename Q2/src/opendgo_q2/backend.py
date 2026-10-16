from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .schema import Usage


@dataclass(frozen=True)
class Completion:
    content: str
    usage: Usage
    finish_reason: str
    request_attempts: int


class ChatBackend(Protocol):
    model: str

    async def complete(
        self, system: str, user: str, *, max_tokens: int, seed: int
    ) -> Completion: ...

    async def health(self) -> dict[str, Any]: ...


class OpenAICompatibleBackend:
    def __init__(self, settings: dict[str, Any]) -> None:
        self.base_url = str(settings["base_url"]).rstrip("/")
        self.api_key = str(settings.get("api_key", "EMPTY"))
        self.model = str(settings["model"])
        self.temperature = float(settings.get("temperature", 0.2))
        self.timeout = float(settings.get("timeout_seconds", 600))
        self.request_retries = int(settings.get("request_retries", 1))
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def health(self) -> dict[str, Any]:
        response = await self._client.get(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()
        return {"models": [item.get("id") for item in response.json().get("data", [])]}

    async def complete(
        self, system: str, user: str, *, max_tokens: int, seed: int
    ) -> Completion:
        last_error: Exception | None = None
        for attempt in range(self.request_retries + 1):
            try:
                response = await self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "temperature": self.temperature,
                        "max_tokens": int(max_tokens),
                        "seed": int(seed),
                    },
                )
                response.raise_for_status()
                payload = response.json()
                raw_usage = payload.get("usage") or {}
                return Completion(
                    content=str(payload["choices"][0]["message"].get("content") or "").strip(),
                    usage=Usage(
                        prompt_tokens=int(raw_usage.get("prompt_tokens", 0) or 0),
                        completion_tokens=int(raw_usage.get("completion_tokens", 0) or 0),
                        total_tokens=int(raw_usage.get("total_tokens", 0) or 0),
                        api_calls=1,
                    ),
                    finish_reason=str(payload["choices"][0].get("finish_reason") or ""),
                    request_attempts=attempt + 1,
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.request_retries:
                    await asyncio.sleep(min(30.0, 2.0**attempt))
        raise RuntimeError(
            f"LLM request failed after {self.request_retries + 1} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        )

    async def close(self) -> None:
        await self._client.aclose()


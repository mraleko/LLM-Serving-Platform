from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Protocol

import httpx

from common.schemas import RequestEnvelope
from services.worker.app.config import WorkerSettings
from services.worker.app.mock_backend import MockTokenBackend


class TokenBackend(Protocol):
    async def stream_batch_tokens(
        self,
        requests: list[RequestEnvelope],
    ) -> AsyncIterator[tuple[RequestEnvelope, int, str]]: ...

    async def aclose(self) -> None: ...


def parse_openai_sse_token(data_line: str) -> str | None:
    try:
        payload = json.loads(data_line)
    except json.JSONDecodeError:
        return None

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None

    delta = first_choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str) and content:
            return content

    text = first_choice.get("text")
    if isinstance(text, str) and text:
        return text
    return None


def parse_anthropic_sse_token(data_line: str) -> str | None:
    try:
        payload = json.loads(data_line)
    except json.JSONDecodeError:
        return None

    event_type = payload.get("type")
    if event_type == "content_block_delta":
        delta = payload.get("delta")
        if isinstance(delta, dict):
            text = delta.get("text")
            if isinstance(text, str) and text:
                return text
    elif event_type == "content_delta":
        text = payload.get("text")
        if isinstance(text, str) and text:
            return text
        delta = payload.get("delta")
        if isinstance(delta, dict):
            nested_text = delta.get("text")
            if isinstance(nested_text, str) and nested_text:
                return nested_text
    return None


def _effective_model(request_model: str, fallback_model: str) -> str:
    if request_model and request_model != "mock-v1":
        return request_model
    return fallback_model


@dataclass(frozen=True)
class _QueueTokenEvent:
    request: RequestEnvelope
    index: int
    token: str


@dataclass(frozen=True)
class _QueueErrorEvent:
    request: RequestEnvelope
    error: Exception


@dataclass(frozen=True)
class _QueueDoneEvent:
    request_id: str


class _HttpSseBackend(ABC):
    def __init__(self, *, timeout_seconds: float, logger: logging.Logger) -> None:
        self._logger = logger
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 10.0))
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream_batch_tokens(
        self,
        requests: list[RequestEnvelope],
    ) -> AsyncIterator[tuple[RequestEnvelope, int, str]]:
        queue: asyncio.Queue[_QueueTokenEvent | _QueueErrorEvent | _QueueDoneEvent] = asyncio.Queue()
        tasks = [asyncio.create_task(self._stream_request_into_queue(request, queue)) for request in requests]
        completed = 0

        try:
            while completed < len(tasks):
                event = await queue.get()
                if isinstance(event, _QueueDoneEvent):
                    completed += 1
                    continue
                if isinstance(event, _QueueErrorEvent):
                    raise RuntimeError(
                        "provider request failed for "
                        f"request_id={event.request.request_id}: {event.error}"
                    ) from event.error
                yield event.request, event.index, event.token
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _stream_request_into_queue(
        self,
        request: RequestEnvelope,
        queue: asyncio.Queue[_QueueTokenEvent | _QueueErrorEvent | _QueueDoneEvent],
    ) -> None:
        index = 0
        try:
            async for token in self._stream_request_tokens(request):
                await queue.put(_QueueTokenEvent(request=request, index=index, token=token))
                index += 1
        except Exception as exc:
            self._logger.exception(
                "provider request failed",
                extra={"request_id": request.request_id, "error": str(exc)},
            )
            await queue.put(_QueueErrorEvent(request=request, error=exc))
        finally:
            await queue.put(_QueueDoneEvent(request_id=request.request_id))

    @abstractmethod
    async def _stream_request_tokens(self, request: RequestEnvelope) -> AsyncIterator[str]:
        raise NotImplementedError


class OpenAITokenBackend(_HttpSseBackend):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        default_model: str,
        timeout_seconds: float,
        logger: logging.Logger,
    ) -> None:
        super().__init__(timeout_seconds=timeout_seconds, logger=logger)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model

    async def _stream_request_tokens(self, request: RequestEnvelope) -> AsyncIterator[str]:
        payload = {
            "model": _effective_model(request.model, self._default_model),
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with self._client.stream(
            "POST",
            f"{self._base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace").strip()
                detail = body[:800] if body else "<empty body>"
                raise RuntimeError(
                    f"openai provider HTTP {response.status_code}: {detail}"
                )
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                token = parse_openai_sse_token(data)
                if token:
                    yield token


class AnthropicTokenBackend(_HttpSseBackend):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        api_version: str,
        default_model: str,
        timeout_seconds: float,
        logger: logging.Logger,
    ) -> None:
        super().__init__(timeout_seconds=timeout_seconds, logger=logger)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._api_version = api_version
        self._default_model = default_model

    async def _stream_request_tokens(self, request: RequestEnvelope) -> AsyncIterator[str]:
        payload = {
            "model": _effective_model(request.model, self._default_model),
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": self._api_version,
            "content-type": "application/json",
        }

        async with self._client.stream(
            "POST",
            f"{self._base_url}/v1/messages",
            json=payload,
            headers=headers,
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace").strip()
                detail = body[:800] if body else "<empty body>"
                raise RuntimeError(
                    f"anthropic provider HTTP {response.status_code}: {detail}"
                )
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                token = parse_anthropic_sse_token(data)
                if token:
                    yield token


def build_backend(settings: WorkerSettings, logger: logging.Logger) -> TokenBackend:
    provider = settings.backend_provider.strip().lower()
    if provider == "mock":
        return MockTokenBackend(token_delay_seconds=settings.mock_token_delay_seconds)

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when WORKER_BACKEND_PROVIDER=openai")
        return OpenAITokenBackend(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            default_model=settings.openai_default_model,
            timeout_seconds=settings.provider_timeout_seconds,
            logger=logger,
        )

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when WORKER_BACKEND_PROVIDER=anthropic")
        return AnthropicTokenBackend(
            api_key=settings.anthropic_api_key,
            base_url=settings.anthropic_base_url,
            api_version=settings.anthropic_version,
            default_model=settings.anthropic_default_model,
            timeout_seconds=settings.provider_timeout_seconds,
            logger=logger,
        )

    raise ValueError(
        f"unsupported backend provider '{settings.backend_provider}' "
        "(supported: mock, openai, anthropic)"
    )

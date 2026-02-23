from __future__ import annotations

import asyncio
import logging

import pytest

from services.worker.app.backends import (
    AnthropicTokenBackend,
    OpenAITokenBackend,
    build_backend,
    parse_anthropic_sse_token,
    parse_openai_sse_token,
)
from services.worker.app.config import WorkerSettings
from services.worker.app.mock_backend import MockTokenBackend


def _settings(**overrides: object) -> WorkerSettings:
    base = WorkerSettings(
        service_name="worker",
        log_level="INFO",
        redis_url="redis://redis:6379/0",
        request_queue_key="inference:queue",
        response_channel_prefix="inference:response:",
        batch_window_ms=20,
        max_batch_size=16,
        max_pending_items=2000,
        max_concurrent_batches=4,
        mock_token_delay_seconds=0.02,
        backend_provider="mock",
        provider_timeout_seconds=60.0,
        openai_api_key=None,
        openai_base_url="https://api.openai.com",
        openai_default_model="gpt-4o-mini",
        anthropic_api_key=None,
        anthropic_base_url="https://api.anthropic.com",
        anthropic_version="2023-06-01",
        anthropic_default_model="claude-3-5-haiku-latest",
        retry_attempts=3,
        retry_base_delay_seconds=0.02,
        retry_max_delay_seconds=0.2,
        shutdown_timeout_seconds=10.0,
        metrics_port=9100,
    )
    values = base.__dict__.copy()
    values.update(overrides)
    return WorkerSettings(**values)


def test_parse_openai_sse_token() -> None:
    line = '{"choices":[{"delta":{"content":"hello "}}]}'
    assert parse_openai_sse_token(line) == "hello "


def test_parse_anthropic_sse_token() -> None:
    line = '{"type":"content_block_delta","delta":{"type":"text_delta","text":"world "}}'
    assert parse_anthropic_sse_token(line) == "world "


def test_build_backend_mock() -> None:
    backend = build_backend(_settings(backend_provider="mock"), logging.getLogger("test"))
    assert isinstance(backend, MockTokenBackend)


def test_build_backend_openai_requires_key() -> None:
    with pytest.raises(ValueError):
        build_backend(_settings(backend_provider="openai", openai_api_key=None), logging.getLogger("test"))


def test_build_backend_anthropic_requires_key() -> None:
    with pytest.raises(ValueError):
        build_backend(
            _settings(backend_provider="anthropic", anthropic_api_key=None), logging.getLogger("test")
        )


def test_build_backend_openai_and_anthropic() -> None:
    openai = build_backend(
        _settings(backend_provider="openai", openai_api_key="key"),
        logging.getLogger("test"),
    )
    anthropic = build_backend(
        _settings(backend_provider="anthropic", anthropic_api_key="key"),
        logging.getLogger("test"),
    )

    assert isinstance(openai, OpenAITokenBackend)
    assert isinstance(anthropic, AnthropicTokenBackend)
    asyncio.run(openai.aclose())
    asyncio.run(anthropic.aclose())

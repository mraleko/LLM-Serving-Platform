from __future__ import annotations

import os
from dataclasses import dataclass

from common.redis_utils import RetryConfig


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_optional(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


@dataclass(frozen=True)
class WorkerSettings:
    service_name: str
    log_level: str
    redis_url: str
    request_queue_key: str
    response_channel_prefix: str
    batch_window_ms: int
    max_batch_size: int
    max_pending_items: int
    max_concurrent_batches: int
    mock_token_delay_seconds: float
    backend_provider: str
    provider_timeout_seconds: float
    openai_api_key: str | None
    openai_base_url: str
    openai_default_model: str
    anthropic_api_key: str | None
    anthropic_base_url: str
    anthropic_version: str
    anthropic_default_model: str
    retry_attempts: int
    retry_base_delay_seconds: float
    retry_max_delay_seconds: float
    shutdown_timeout_seconds: float
    metrics_port: int

    @property
    def retry_config(self) -> RetryConfig:
        return RetryConfig(
            attempts=self.retry_attempts,
            base_delay_seconds=self.retry_base_delay_seconds,
            max_delay_seconds=self.retry_max_delay_seconds,
        )


def load_settings() -> WorkerSettings:
    return WorkerSettings(
        service_name=os.getenv("WORKER_SERVICE_NAME", "worker"),
        log_level=os.getenv("WORKER_LOG_LEVEL", "INFO"),
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        request_queue_key=os.getenv("REQUEST_QUEUE_KEY", "inference:queue"),
        response_channel_prefix=os.getenv("RESPONSE_CHANNEL_PREFIX", "inference:response:"),
        batch_window_ms=_env_int("BATCH_WINDOW_MS", 20),
        max_batch_size=_env_int("MAX_BATCH_SIZE", 16),
        max_pending_items=_env_int("MAX_PENDING_ITEMS", 2000),
        max_concurrent_batches=_env_int("MAX_CONCURRENT_BATCHES", 4),
        mock_token_delay_seconds=_env_float("MOCK_TOKEN_DELAY_SECONDS", 0.02),
        backend_provider=os.getenv("WORKER_BACKEND_PROVIDER", "mock"),
        provider_timeout_seconds=_env_float("PROVIDER_TIMEOUT_SECONDS", 60.0),
        openai_api_key=_env_optional("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com"),
        openai_default_model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
        anthropic_api_key=_env_optional("ANTHROPIC_API_KEY"),
        anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        anthropic_version=os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
        anthropic_default_model=os.getenv(
            "ANTHROPIC_DEFAULT_MODEL", "claude-3-5-haiku-latest"
        ),
        retry_attempts=_env_int("INTERNAL_RETRY_ATTEMPTS", 3),
        retry_base_delay_seconds=_env_float("INTERNAL_RETRY_BASE_DELAY_SECONDS", 0.02),
        retry_max_delay_seconds=_env_float("INTERNAL_RETRY_MAX_DELAY_SECONDS", 0.2),
        shutdown_timeout_seconds=_env_float("SHUTDOWN_TIMEOUT_SECONDS", 10.0),
        metrics_port=_env_int("WORKER_METRICS_PORT", 9100),
    )

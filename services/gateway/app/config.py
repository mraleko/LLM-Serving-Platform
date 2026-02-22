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


@dataclass(frozen=True)
class GatewaySettings:
    service_name: str
    log_level: str
    redis_url: str
    request_queue_key: str
    response_channel_prefix: str
    queue_max_depth: int
    request_timeout_seconds: float
    rate_limit_requests: int
    rate_limit_window_seconds: int
    api_key: str | None
    retry_attempts: int
    retry_base_delay_seconds: float
    retry_max_delay_seconds: float

    @property
    def retry_config(self) -> RetryConfig:
        return RetryConfig(
            attempts=self.retry_attempts,
            base_delay_seconds=self.retry_base_delay_seconds,
            max_delay_seconds=self.retry_max_delay_seconds,
        )


def load_settings() -> GatewaySettings:
    api_key = os.getenv("GATEWAY_API_KEY", "").strip() or None
    return GatewaySettings(
        service_name=os.getenv("GATEWAY_SERVICE_NAME", "gateway"),
        log_level=os.getenv("GATEWAY_LOG_LEVEL", "INFO"),
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        request_queue_key=os.getenv("REQUEST_QUEUE_KEY", "inference:queue"),
        response_channel_prefix=os.getenv("RESPONSE_CHANNEL_PREFIX", "inference:response:"),
        queue_max_depth=_env_int("QUEUE_MAX_DEPTH", 1000),
        request_timeout_seconds=_env_float("REQUEST_TIMEOUT_SECONDS", 30.0),
        rate_limit_requests=_env_int("RATE_LIMIT_REQUESTS", 60),
        rate_limit_window_seconds=_env_int("RATE_LIMIT_WINDOW_SECONDS", 60),
        api_key=api_key,
        retry_attempts=_env_int("INTERNAL_RETRY_ATTEMPTS", 3),
        retry_base_delay_seconds=_env_float("INTERNAL_RETRY_BASE_DELAY_SECONDS", 0.02),
        retry_max_delay_seconds=_env_float("INTERNAL_RETRY_MAX_DELAY_SECONDS", 0.2),
    )

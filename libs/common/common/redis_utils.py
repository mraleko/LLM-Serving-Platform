from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryConfig:
    attempts: int = 3
    base_delay_seconds: float = 0.02
    max_delay_seconds: float = 0.2


async def with_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    retry_config: RetryConfig,
    logger: logging.Logger | None = None,
    operation_name: str = "redis_operation",
) -> T:
    for attempt in range(1, retry_config.attempts + 1):
        try:
            return await operation()
        except (RedisConnectionError, RedisTimeoutError) as exc:
            if attempt >= retry_config.attempts:
                raise
            delay = min(
                retry_config.max_delay_seconds,
                retry_config.base_delay_seconds * (2 ** (attempt - 1)),
            ) + random.uniform(0, retry_config.base_delay_seconds)
            if logger:
                logger.warning(
                    "retrying redis operation",
                    extra={
                        "operation": operation_name,
                        "attempt": attempt,
                        "max_attempts": retry_config.attempts,
                        "delay_seconds": round(delay, 4),
                        "error": str(exc),
                    },
                )
            await asyncio.sleep(delay)

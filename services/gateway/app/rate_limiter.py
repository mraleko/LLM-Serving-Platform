from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import redis.asyncio as redis

from common.redis_utils import RetryConfig, with_retries

_RATE_LIMIT_SCRIPT = """
local current = redis.call("INCR", KEYS[1])
if current == 1 then
  redis.call("EXPIRE", KEYS[1], tonumber(ARGV[1]))
end
return current
"""


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    reset_after_seconds: int


class RedisRateLimiter:
    def __init__(
        self,
        redis_client: redis.Redis,
        *,
        limit: int,
        window_seconds: int,
        retry_config: RetryConfig,
        logger: logging.Logger,
    ) -> None:
        self._redis = redis_client
        self._limit = limit
        self._window_seconds = window_seconds
        self._retry_config = retry_config
        self._logger = logger

    async def allow(self, identity: str) -> RateLimitDecision:
        window_bucket = int(time.time() // self._window_seconds)
        key = f"ratelimit:{identity}:{window_bucket}"

        async def _call() -> int:
            raw = await self._redis.eval(_RATE_LIMIT_SCRIPT, 1, key, self._window_seconds)
            return int(raw)

        count = await with_retries(
            _call,
            retry_config=self._retry_config,
            logger=self._logger,
            operation_name="rate_limit_eval",
        )
        now = int(time.time())
        window_end = (window_bucket + 1) * self._window_seconds
        remaining = max(self._limit - count, 0)
        return RateLimitDecision(
            allowed=count <= self._limit,
            remaining=remaining,
            reset_after_seconds=max(window_end - now, 1),
        )

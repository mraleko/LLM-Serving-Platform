from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator

import redis.asyncio as redis
from pydantic import ValidationError

from common.redis_utils import RetryConfig, with_retries
from common.schemas import RequestEnvelope, WorkerEvent

_ENQUEUE_SCRIPT = """
local queue_size = redis.call("LLEN", KEYS[1])
if queue_size >= tonumber(ARGV[1]) then
  return 0
end
redis.call("RPUSH", KEYS[1], ARGV[2])
return 1
"""


class QueueOverloadedError(Exception):
    pass


class InternalTimeoutError(Exception):
    pass


class WorkerExecutionError(Exception):
    pass


@dataclass
class CompletionResult:
    text: str
    generated_tokens: int
    batch_size: int | None


class RedisGatewayClient:
    def __init__(
        self,
        *,
        redis_url: str,
        request_queue_key: str,
        response_channel_prefix: str,
        queue_max_depth: int,
        retry_config: RetryConfig,
        logger: logging.Logger,
    ) -> None:
        self._redis_url = redis_url
        self._request_queue_key = request_queue_key
        self._response_channel_prefix = response_channel_prefix
        self._queue_max_depth = queue_max_depth
        self._retry_config = retry_config
        self._logger = logger
        self._redis: redis.Redis | None = None

    async def connect(self) -> None:
        self._redis = redis.from_url(
            self._redis_url,
            decode_responses=True,
            encoding="utf-8",
            health_check_interval=30,
            socket_timeout=2,
            socket_connect_timeout=2,
        )
        await self.ping()

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()

    async def ping(self) -> None:
        redis_client = self._require_redis()
        await with_retries(
            redis_client.ping,
            retry_config=self._retry_config,
            logger=self._logger,
            operation_name="redis_ping",
        )

    async def enqueue_request(self, envelope: RequestEnvelope) -> None:
        redis_client = self._require_redis()
        payload = envelope.model_dump_json()

        async def _enqueue() -> int:
            result = await redis_client.eval(
                _ENQUEUE_SCRIPT,
                1,
                self._request_queue_key,
                self._queue_max_depth,
                payload,
            )
            return int(result)

        accepted = await with_retries(
            _enqueue,
            retry_config=self._retry_config,
            logger=self._logger,
            operation_name="enqueue_request",
        )
        if accepted == 0:
            raise QueueOverloadedError("queue is overloaded")

    async def iter_worker_events(
        self,
        request_id: str,
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[WorkerEvent]:
        redis_client = self._require_redis()
        channel = f"{self._response_channel_prefix}{request_id}"
        pubsub = redis_client.pubsub(ignore_subscribe_messages=True)
        await with_retries(
            lambda: pubsub.subscribe(channel),
            retry_config=self._retry_config,
            logger=self._logger,
            operation_name="pubsub_subscribe",
        )

        deadline = time.monotonic() + timeout_seconds
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise InternalTimeoutError("timed out waiting for worker event")
                message = await with_retries(
                    lambda: pubsub.get_message(timeout=min(1.0, remaining)),
                    retry_config=self._retry_config,
                    logger=self._logger,
                    operation_name="pubsub_get_message",
                )
                if message is None:
                    await asyncio.sleep(0.01)
                    continue
                raw_data = message.get("data")
                if raw_data is None:
                    continue
                try:
                    event = WorkerEvent.model_validate_json(raw_data)
                except ValidationError:
                    self._logger.warning(
                        "dropping invalid worker event",
                        extra={"request_id": request_id, "payload": raw_data},
                    )
                    continue
                yield event
                if event.type in {"end", "error"}:
                    return
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    async def collect_completion(
        self,
        request_id: str,
        *,
        timeout_seconds: float,
    ) -> CompletionResult:
        parts: list[str] = []
        generated_tokens = 0
        batch_size: int | None = None
        async for event in self.iter_worker_events(request_id, timeout_seconds=timeout_seconds):
            if event.type == "batch" and event.batch_size:
                batch_size = event.batch_size
            elif event.type == "token" and event.token is not None:
                parts.append(event.token)
                generated_tokens += 1
            elif event.type == "error":
                raise WorkerExecutionError(event.error or "worker execution failed")
        return CompletionResult(
            text="".join(parts).strip(),
            generated_tokens=generated_tokens,
            batch_size=batch_size,
        )

    def redis_client(self) -> redis.Redis:
        return self._require_redis()

    def _require_redis(self) -> redis.Redis:
        if self._redis is None:
            raise RuntimeError("Redis client is not connected")
        return self._redis

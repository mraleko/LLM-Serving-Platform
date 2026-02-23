from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Iterable

import redis.asyncio as redis
from prometheus_client import start_http_server
from pydantic import ValidationError

from common.logging import configure_logging
from common.metrics import WorkerMetrics
from common.redis_utils import with_retries
from common.schemas import RequestEnvelope, WorkerEvent
from services.worker.app.backends import TokenBackend, build_backend
from services.worker.app.config import WorkerSettings, load_settings
from services.worker.app.scheduler import BatchScheduler, SchedulerBackpressureError


class WorkerService:
    def __init__(self, settings: WorkerSettings) -> None:
        self._settings = settings
        configure_logging(service_name=settings.service_name, level=settings.log_level)
        self._logger = logging.getLogger("worker")
        self._redis: redis.Redis | None = None
        self._stop_event = asyncio.Event()
        self._scheduler = BatchScheduler(
            batch_window_ms=settings.batch_window_ms,
            max_batch_size=settings.max_batch_size,
            max_pending_items=settings.max_pending_items,
        )
        self._backend: TokenBackend = build_backend(settings, self._logger)
        self._metrics = WorkerMetrics()
        self._batch_semaphore = asyncio.Semaphore(settings.max_concurrent_batches)
        self._tasks: set[asyncio.Task[None]] = set()

    async def run(self) -> None:
        await self._connect()
        start_http_server(self._settings.metrics_port, registry=self._metrics.registry)
        self._logger.info(
            "worker started",
            extra={
                "batch_window_ms": self._settings.batch_window_ms,
                "max_batch_size": self._settings.max_batch_size,
                "backend_provider": self._settings.backend_provider,
                "metrics_port": self._settings.metrics_port,
            },
        )

        loop = asyncio.get_running_loop()
        self._register_signal_handlers(loop, (signal.SIGINT, signal.SIGTERM))

        try:
            while not self._stop_event.is_set():
                await self._pull_and_schedule()
                await self._dispatch_ready_batches()
        finally:
            await self._shutdown()

    def request_stop(self) -> None:
        self._stop_event.set()

    async def _connect(self) -> None:
        self._redis = redis.from_url(
            self._settings.redis_url,
            decode_responses=True,
            encoding="utf-8",
            health_check_interval=30,
            socket_timeout=2,
            socket_connect_timeout=2,
        )
        await with_retries(
            self._redis.ping,
            retry_config=self._settings.retry_config,
            logger=self._logger,
            operation_name="worker_redis_ping",
        )

    async def _pull_and_schedule(self) -> None:
        redis_client = self._require_redis()
        entry = await with_retries(
            lambda: redis_client.blpop(self._settings.request_queue_key, timeout=1),
            retry_config=self._settings.retry_config,
            logger=self._logger,
            operation_name="worker_blpop",
        )
        if entry is not None:
            _, payload = entry
            await self._schedule_payload(payload)
            await self._drain_queue(limit=self._settings.max_batch_size * 4)
        else:
            # Even if the queue is empty, flush requests that hit batching window.
            await asyncio.sleep(0.001)

    async def _drain_queue(self, *, limit: int) -> None:
        redis_client = self._require_redis()
        for _ in range(limit):
            payload = await with_retries(
                lambda: redis_client.lpop(self._settings.request_queue_key),
                retry_config=self._settings.retry_config,
                logger=self._logger,
                operation_name="worker_lpop",
            )
            if payload is None:
                return
            await self._schedule_payload(payload)

    async def _schedule_payload(self, payload: str) -> None:
        try:
            envelope = RequestEnvelope.model_validate_json(payload)
        except ValidationError:
            self._logger.warning("dropping invalid request payload", extra={"payload": payload})
            self._metrics.errors_total.labels(code="INVALID_PAYLOAD").inc()
            return

        try:
            self._scheduler.add(envelope)
        except SchedulerBackpressureError:
            self._metrics.errors_total.labels(code="SCHEDULER_OVERFLOW").inc()
            self._logger.warning(
                "scheduler overflow",
                extra={"request_id": envelope.request_id, "pending": self._scheduler.pending_items},
            )
            await self._publish_event(
                envelope.request_id,
                WorkerEvent(
                    type="error",
                    request_id=envelope.request_id,
                    error="worker scheduler overloaded",
                ),
            )

    async def _dispatch_ready_batches(self) -> None:
        batches = self._scheduler.pop_ready_batches()
        for batch in batches:
            task = asyncio.create_task(self._process_batch(batch))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _process_batch(self, batch: list[RequestEnvelope]) -> None:
        async with self._batch_semaphore:
            batch_size = len(batch)
            self._metrics.batch_size_histogram.observe(batch_size)
            self._logger.info("processing batch", extra={"batch_size": batch_size})
            token_counts = {request.request_id: 0 for request in batch}

            try:
                for request in batch:
                    await self._publish_event(
                        request.request_id,
                        WorkerEvent(
                            type="batch",
                            request_id=request.request_id,
                            batch_size=batch_size,
                        ),
                    )

                async for request, index, token in self._backend.stream_batch_tokens(batch):
                    token_counts[request.request_id] += 1
                    self._metrics.tokens_generated_total.inc()
                    await self._publish_event(
                        request.request_id,
                        WorkerEvent(
                            type="token",
                            request_id=request.request_id,
                            token=token,
                            index=index,
                        ),
                    )

                for request in batch:
                    await self._publish_event(
                        request.request_id,
                        WorkerEvent(
                            type="end",
                            request_id=request.request_id,
                            generated_tokens=token_counts[request.request_id],
                            finish_reason="length",
                        ),
                    )
            except Exception as exc:
                self._metrics.errors_total.labels(code="BATCH_FAILURE").inc()
                self._logger.exception(
                    "batch failed",
                    extra={"batch_size": batch_size, "error": str(exc)},
                )
                await self._publish_errors(batch, "worker batch processing failed")

    async def _publish_errors(self, batch: Iterable[RequestEnvelope], message: str) -> None:
        for request in batch:
            try:
                await self._publish_event(
                    request.request_id,
                    WorkerEvent(type="error", request_id=request.request_id, error=message),
                )
            except Exception:
                self._logger.exception(
                    "failed to publish error event",
                    extra={"request_id": request.request_id},
                )

    async def _publish_event(self, request_id: str, event: WorkerEvent) -> None:
        redis_client = self._require_redis()
        channel = f"{self._settings.response_channel_prefix}{request_id}"
        payload = event.model_dump_json(exclude_none=True)
        await with_retries(
            lambda: redis_client.publish(channel, payload),
            retry_config=self._settings.retry_config,
            logger=self._logger,
            operation_name="worker_publish",
        )

    async def _shutdown(self) -> None:
        self._logger.info("worker shutdown initiated")
        for batch in self._scheduler.pop_ready_batches(force=True):
            task = asyncio.create_task(self._process_batch(batch))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        if self._tasks:
            done, pending = await asyncio.wait(
                self._tasks,
                timeout=self._settings.shutdown_timeout_seconds,
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            self._logger.info(
                "worker tasks drained",
                extra={"completed": len(done), "cancelled": len(pending)},
            )

        redis_client = self._redis
        if redis_client is not None:
            await redis_client.aclose()
        await self._backend.aclose()
        self._logger.info("worker stopped")

    def _register_signal_handlers(
        self,
        loop: asyncio.AbstractEventLoop,
        signals: tuple[signal.Signals, ...],
    ) -> None:
        for sig in signals:
            try:
                loop.add_signal_handler(sig, self.request_stop)
            except NotImplementedError:  # pragma: no cover - windows compatibility
                pass

    def _require_redis(self) -> redis.Redis:
        if self._redis is None:
            raise RuntimeError("Redis client not initialized")
        return self._redis


async def _run() -> None:
    service = WorkerService(load_settings())
    await service.run()


if __name__ == "__main__":
    asyncio.run(_run())

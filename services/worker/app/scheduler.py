from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque

from common.schemas import RequestEnvelope

BatchKey = tuple[str, int]


class SchedulerBackpressureError(Exception):
    pass


@dataclass(frozen=True)
class ScheduledRequest:
    envelope: RequestEnvelope
    enqueued_at: float


class BatchScheduler:
    def __init__(
        self,
        *,
        batch_window_ms: int,
        max_batch_size: int,
        max_pending_items: int,
    ) -> None:
        self._batch_window_seconds = batch_window_ms / 1000
        self._max_batch_size = max_batch_size
        self._max_pending_items = max_pending_items
        self._pending_items = 0
        self._queues: dict[BatchKey, Deque[ScheduledRequest]] = defaultdict(deque)

    def add(self, envelope: RequestEnvelope, now: float | None = None) -> None:
        if self._pending_items >= self._max_pending_items:
            raise SchedulerBackpressureError("scheduler queue is full")
        key = self._make_key(envelope)
        self._queues[key].append(
            ScheduledRequest(envelope=envelope, enqueued_at=now if now is not None else time.monotonic())
        )
        self._pending_items += 1

    def pop_ready_batches(self, now: float | None = None, *, force: bool = False) -> list[list[RequestEnvelope]]:
        current = now if now is not None else time.monotonic()
        ready_batches: list[list[RequestEnvelope]] = []
        empty_keys: list[BatchKey] = []

        for key, queue in self._queues.items():
            while queue:
                oldest = queue[0]
                waited_long_enough = (current - oldest.enqueued_at) >= self._batch_window_seconds
                has_full_batch = len(queue) >= self._max_batch_size
                if not force and not waited_long_enough and not has_full_batch:
                    break

                batch: list[RequestEnvelope] = []
                while queue and len(batch) < self._max_batch_size:
                    batch.append(queue.popleft().envelope)
                    self._pending_items -= 1
                ready_batches.append(batch)
            if not queue:
                empty_keys.append(key)

        for key in empty_keys:
            del self._queues[key]

        return ready_batches

    @property
    def pending_items(self) -> int:
        return self._pending_items

    def _make_key(self, envelope: RequestEnvelope) -> BatchKey:
        return envelope.model, envelope.max_tokens

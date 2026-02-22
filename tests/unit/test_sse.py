from __future__ import annotations

import asyncio
from typing import AsyncIterator

from common.schemas import WorkerEvent
from services.gateway.app.sse import worker_events_to_sse


async def _emit(events: list[WorkerEvent]) -> AsyncIterator[WorkerEvent]:
    for event in events:
        yield event


def _collect(events: list[WorkerEvent]) -> str:
    async def _run() -> str:
        chunks = [chunk async for chunk in worker_events_to_sse(_emit(events))]
        return b"".join(chunks).decode("utf-8")

    return asyncio.run(_run())


def test_sse_stream_emits_multiple_token_events() -> None:
    events = [
        WorkerEvent(type="batch", request_id="req-1", batch_size=4),
        WorkerEvent(type="token", request_id="req-1", token="hello ", index=0),
        WorkerEvent(type="token", request_id="req-1", token="world ", index=1),
        WorkerEvent(type="end", request_id="req-1", generated_tokens=2, finish_reason="length"),
    ]

    body = _collect(events)

    assert body.count("event: token") == 2
    assert "event: done" in body
    assert '"generated_tokens":2' in body


def test_sse_stream_emits_error_event() -> None:
    events = [
        WorkerEvent(type="error", request_id="req-2", error="boom"),
    ]

    body = _collect(events)

    assert "event: error" in body
    assert '"code":"WORKER_ERROR"' in body

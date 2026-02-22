from __future__ import annotations

import json
from typing import AsyncIterator, Callable

from common.schemas import WorkerEvent


def encode_sse(event: str, data: dict[str, object]) -> bytes:
    payload = json.dumps(data, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


async def worker_events_to_sse(
    events: AsyncIterator[WorkerEvent],
    *,
    on_internal_event: Callable[[WorkerEvent], None] | None = None,
) -> AsyncIterator[bytes]:
    async for event in events:
        if on_internal_event is not None:
            on_internal_event(event)
        if event.type == "token":
            yield encode_sse(
                "token",
                {
                    "request_id": event.request_id,
                    "token": event.token,
                    "index": event.index,
                },
            )
        elif event.type == "end":
            yield encode_sse(
                "done",
                {
                    "request_id": event.request_id,
                    "generated_tokens": event.generated_tokens,
                    "finish_reason": event.finish_reason or "stop",
                },
            )
            return
        elif event.type == "error":
            yield encode_sse(
                "error",
                {
                    "request_id": event.request_id,
                    "code": "WORKER_ERROR",
                    "message": event.error or "worker execution failed",
                },
            )
            return

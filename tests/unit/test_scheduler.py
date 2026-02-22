from __future__ import annotations

from common.schemas import RequestEnvelope
from services.worker.app.scheduler import BatchScheduler


def _envelope(request_id: str, *, model: str = "mock-v1", max_tokens: int = 16) -> RequestEnvelope:
    return RequestEnvelope(
        request_id=request_id,
        created_at=0.0,
        prompt=f"prompt-{request_id}",
        max_tokens=max_tokens,
        temperature=0.7,
        model=model,
        stream=False,
    )


def test_batches_after_window_for_same_key() -> None:
    scheduler = BatchScheduler(batch_window_ms=20, max_batch_size=8, max_pending_items=100)
    scheduler.add(_envelope("r1"), now=0.0)
    scheduler.add(_envelope("r2"), now=0.0)

    assert scheduler.pop_ready_batches(now=0.005) == []

    batches = scheduler.pop_ready_batches(now=0.03)
    assert len(batches) == 1
    assert [item.request_id for item in batches[0]] == ["r1", "r2"]


def test_full_batch_dispatches_immediately() -> None:
    scheduler = BatchScheduler(batch_window_ms=30, max_batch_size=2, max_pending_items=100)
    scheduler.add(_envelope("r1"), now=0.0)
    scheduler.add(_envelope("r2"), now=0.0)

    batches = scheduler.pop_ready_batches(now=0.0)
    assert len(batches) == 1
    assert [item.request_id for item in batches[0]] == ["r1", "r2"]


def test_scheduler_separates_batch_keys() -> None:
    scheduler = BatchScheduler(batch_window_ms=20, max_batch_size=8, max_pending_items=100)
    scheduler.add(_envelope("a1", model="model-a", max_tokens=32), now=0.0)
    scheduler.add(_envelope("a2", model="model-a", max_tokens=32), now=0.0)
    scheduler.add(_envelope("b1", model="model-b", max_tokens=32), now=0.0)
    scheduler.add(_envelope("c1", model="model-a", max_tokens=16), now=0.0)

    batches = scheduler.pop_ready_batches(now=0.03)
    grouped = [sorted(item.request_id for item in batch) for batch in batches]

    assert sorted(grouped) == [["a1", "a2"], ["b1"], ["c1"]]

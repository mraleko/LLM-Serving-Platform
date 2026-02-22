from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from common.logging import configure_logging
from common.metrics import GatewayMetrics
from common.schemas import GenerateRequest, GenerateResponse, RequestEnvelope, WorkerEvent
from services.gateway.app.config import GatewaySettings, load_settings
from services.gateway.app.queue_client import (
    InternalTimeoutError,
    QueueOverloadedError,
    RedisGatewayClient,
    WorkerExecutionError,
)
from services.gateway.app.rate_limiter import RedisRateLimiter
from services.gateway.app.sse import encode_sse, worker_events_to_sse

settings = load_settings()
configure_logging(service_name=settings.service_name, level=settings.log_level)
logger = logging.getLogger("gateway")


def _error_payload(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


def _resolve_identity(request: Request, api_key: str | None) -> str:
    if api_key:
        return f"api_key:{api_key}"
    if request.client and request.client.host:
        return f"ip:{request.client.host}"
    return "ip:unknown"


def _track_event_metrics(event: WorkerEvent, metrics: GatewayMetrics) -> None:
    if event.type == "batch" and event.batch_size:
        metrics.batch_size_histogram.observe(event.batch_size)
    elif event.type == "token":
        metrics.tokens_generated_total.inc()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    gateway_client = RedisGatewayClient(
        redis_url=settings.redis_url,
        request_queue_key=settings.request_queue_key,
        response_channel_prefix=settings.response_channel_prefix,
        queue_max_depth=settings.queue_max_depth,
        retry_config=settings.retry_config,
        logger=logger,
    )
    await gateway_client.connect()
    app.state.settings = settings
    app.state.gateway_client = gateway_client
    app.state.metrics = GatewayMetrics()
    app.state.rate_limiter = RedisRateLimiter(
        gateway_client.redis_client(),
        limit=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
        retry_config=settings.retry_config,
        logger=logger,
    )
    logger.info("gateway started")
    try:
        yield
    finally:
        await gateway_client.close()
        logger.info("gateway stopped")


app = FastAPI(title="LLM Gateway", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, str]:
    gateway_client: RedisGatewayClient = request.app.state.gateway_client
    try:
        await gateway_client.ping()
    except Exception as exc:  # pragma: no cover - exercised in integration
        logger.warning("health check failed", extra={"error": str(exc)})
        raise HTTPException(
            status_code=503,
            detail=_error_payload("UNHEALTHY", "redis unavailable"),
        ) from exc
    return {"status": "ok"}


@app.get("/metrics")
async def metrics(request: Request) -> Response:
    metrics_store: GatewayMetrics = request.app.state.metrics
    return Response(
        content=metrics_store.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.post("/v1/generate")
async def generate(
    body: GenerateRequest,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> Response:
    app_settings: GatewaySettings = request.app.state.settings
    gateway_client: RedisGatewayClient = request.app.state.gateway_client
    limiter: RedisRateLimiter = request.app.state.rate_limiter
    metrics_store: GatewayMetrics = request.app.state.metrics

    request_id = str(uuid.uuid4())
    started = time.monotonic()
    status_label = "success"
    streaming_response_started = False
    metrics_store.inflight_requests.inc()

    try:
        if app_settings.api_key and x_api_key != app_settings.api_key:
            status_label = "unauthorized"
            metrics_store.errors_total.labels(code="UNAUTHORIZED").inc()
            raise HTTPException(
                status_code=401,
                detail=_error_payload("UNAUTHORIZED", "missing or invalid API key"),
            )

        identity = _resolve_identity(request, x_api_key)
        decision = await limiter.allow(identity)
        if not decision.allowed:
            status_label = "rate_limited"
            metrics_store.errors_total.labels(code="RATE_LIMITED").inc()
            raise HTTPException(
                status_code=429,
                detail=_error_payload("RATE_LIMITED", "rate limit exceeded"),
                headers={"Retry-After": str(decision.reset_after_seconds)},
            )

        envelope = RequestEnvelope(
            request_id=request_id,
            created_at=time.time(),
            prompt=body.prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            model=body.model,
            stream=body.stream,
        )
        try:
            await gateway_client.enqueue_request(envelope)
        except QueueOverloadedError as exc:
            status_label = "overloaded"
            metrics_store.errors_total.labels(code="OVERLOADED").inc()
            raise HTTPException(
                status_code=429,
                detail=_error_payload("OVERLOADED", "request queue is full"),
            ) from exc

        if body.stream:
            streaming_response_started = True
            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id,
            }

            async def stream_events() -> AsyncIterator[bytes]:
                nonlocal status_label
                try:
                    async for chunk in worker_events_to_sse(
                        gateway_client.iter_worker_events(
                            request_id,
                            timeout_seconds=app_settings.request_timeout_seconds,
                        ),
                        on_internal_event=lambda event: _track_event_metrics(event, metrics_store),
                    ):
                        yield chunk
                except InternalTimeoutError:
                    status_label = "timeout"
                    metrics_store.errors_total.labels(code="INFERENCE_TIMEOUT").inc()
                    yield encode_sse(
                        "error",
                        {
                            "request_id": request_id,
                            "code": "INFERENCE_TIMEOUT",
                            "message": "timed out waiting for worker",
                        },
                    )
                except Exception as exc:  # pragma: no cover - integration path
                    status_label = "worker_error"
                    metrics_store.errors_total.labels(code="WORKER_ERROR").inc()
                    logger.exception(
                        "streaming request failed",
                        extra={"request_id": request_id, "error": str(exc)},
                    )
                    yield encode_sse(
                        "error",
                        {
                            "request_id": request_id,
                            "code": "WORKER_ERROR",
                            "message": "worker failed while streaming",
                        },
                    )
                finally:
                    metrics_store.observe_latency(time.monotonic() - started)
                    metrics_store.request_count.labels(status=status_label).inc()
                    metrics_store.inflight_requests.dec()

            return StreamingResponse(stream_events(), media_type="text/event-stream", headers=headers)

        completion = await gateway_client.collect_completion(
            request_id,
            timeout_seconds=app_settings.request_timeout_seconds,
        )
        if completion.batch_size is not None:
            metrics_store.batch_size_histogram.observe(completion.batch_size)
        metrics_store.tokens_generated_total.inc(completion.generated_tokens)

        response = GenerateResponse(
            request_id=request_id,
            model=body.model,
            text=completion.text,
            generated_tokens=completion.generated_tokens,
        )
        return JSONResponse(content=response.model_dump(), headers={"X-Request-ID": request_id})
    except InternalTimeoutError as exc:
        status_label = "timeout"
        metrics_store.errors_total.labels(code="INFERENCE_TIMEOUT").inc()
        raise HTTPException(
            status_code=504,
            detail=_error_payload("INFERENCE_TIMEOUT", "timed out waiting for worker"),
        ) from exc
    except WorkerExecutionError as exc:
        status_label = "worker_error"
        metrics_store.errors_total.labels(code="WORKER_ERROR").inc()
        raise HTTPException(
            status_code=502,
            detail=_error_payload("WORKER_ERROR", str(exc)),
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - integration path
        status_label = "internal_error"
        metrics_store.errors_total.labels(code="INTERNAL_ERROR").inc()
        logger.exception("unhandled gateway error", extra={"request_id": request_id, "error": str(exc)})
        raise HTTPException(
            status_code=500,
            detail=_error_payload("INTERNAL_ERROR", "unexpected gateway error"),
        ) from exc
    finally:
        if not streaming_response_started:
            metrics_store.observe_latency(time.monotonic() - started)
            metrics_store.request_count.labels(status=status_label).inc()
            metrics_store.inflight_requests.dec()

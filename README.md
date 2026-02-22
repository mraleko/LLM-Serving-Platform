# LLM Serving Platform (FastAPI + Redis + Worker Batching)

Production-style local LLM serving stack with:
- Gateway API (`FastAPI`) with SSE streaming, rate limiting, backpressure, and Prometheus metrics
- Worker service with Redis queue consumption, micro-batching, and pluggable token backend
- Shared schemas/logging/retry utilities in `libs/common`
- Docker Compose local runtime
- Unit, integration, and chaos tests

## ASCII Diagram

Architecture file: `docs/architecture_diagram.txt`

```text
                          +---------------------------+
                          |        Client / SDK       |
                          +-------------+-------------+
                                        |
                                        | POST /v1/generate
                                        | stream=true/false
                                        v
                     +----------------------------------------+
                     | Gateway Service (FastAPI / Uvicorn)   |
                     | - Request validation                   |
                     | - API key auth (optional)             |
                     | - IP/API-key rate limiting            |
                     | - Queue overload protection (429)      |
                     | - SSE relay                            |
                     | - Prometheus metrics + JSON logs       |
                     +-------------------+--------------------+
                                         |
                                         | RPUSH inference:queue
                                         v
                              +----------------------+
                              | Redis                |
                              | - Request queue      |
                              | - Pub/Sub channels   |
                              +----------+-----------+
                                         ^
                                         | PUBLISH inference:response:<request_id>
                     +-------------------+--------------------+
                     | Worker Service                           |
                     | - BLPOP queue polling                    |
                     | - Batch scheduler by (model,max_tokens) |
                     | - Microbatch window (10-30ms)           |
                     | - Mock pluggable backend                |
                     | - Token-by-token publish                |
                     +-------------------+--------------------+
```

## Repo Layout

```text
services/gateway/        # FastAPI gateway
services/worker/         # batching scheduler + inference worker
libs/common/             # shared schemas, logging, metrics, retry utils
scripts/                 # demo, load test, benchmark
tests/                   # unit + integration + chaos tests
docs/                    # architecture diagram
artifacts/benchmark/     # benchmark result artifacts
docker-compose.yml
Makefile
```

## API

### `POST /v1/generate`

Body:

```json
{
  "prompt": "hello",
  "max_tokens": 64,
  "temperature": 0.7,
  "stream": true
}
```

- `stream=false`: JSON response
- `stream=true`: `text/event-stream` with `token`, `done`, `error` events

### `GET /healthz`
Checks gateway + Redis.

### `GET /metrics`
Prometheus text metrics from gateway.

## Run

```bash
docker compose up --build
```

Gateway: `http://localhost:8000`  
Worker metrics: `http://localhost:9100/metrics`

## Clean Demo Command (One-Liner)

```bash
docker compose up --build -d && python3.11 scripts/demo_stream.py && docker compose down -v
```

Equivalent Make target:

```bash
make demo
```

## Benchmark Script + Results

Benchmark script: `scripts/benchmark.py`

Real endpoint benchmark:

```bash
python3.11 scripts/benchmark.py --url http://localhost:8000/v1/generate --requests 500 --concurrency 50 --stream --out artifacts/benchmark/latest.json
```

Synthetic fallback benchmark (works without Docker/Redis):

```bash
python3.11 scripts/benchmark.py --synthetic --requests 600 --concurrency 60 --stream --max-tokens 64 --synthetic-failure-rate 0 --out artifacts/benchmark/latest.json
```

Checked-in result artifact:
- `artifacts/benchmark/latest.json`
- `artifacts/benchmark/RESULTS.md`

Latest checked-in summary:

| Metric | Value |
|---|---:|
| Requests total | 600 |
| Success / Failed | 600 / 0 |
| Throughput (req/s) | 415.1138 |
| p95 latency (s) | 0.206259 |
| p99 latency (s) | 0.342013 |
| Token throughput (tok/s) | 26567.2859 |

Note: these checked-in values come from synthetic fallback mode because this sandbox lacks Docker/Redis runtime.

## Chaos Test (Distributed Behavior)

Chaos test file: `tests/integration/test_chaos_worker_restart.py`

What it does:
- Starts stack via docker compose
- Opens streaming request
- Kills worker during in-flight stream (`docker compose kill worker`)
- Restarts worker
- Verifies the system recovers and serves a new streaming request with multiple tokens

Run:

```bash
make test-chaos
```

## Security + Reliability

- Optional API key (`X-API-Key`) via `GATEWAY_API_KEY`
- Fixed-window Redis rate limiting per API key or IP
- Queue backpressure with bounded depth and `429 OVERLOADED`
- Internal retries for Redis ops with exponential backoff + jitter
- Graceful shutdown in gateway and worker
- Structured JSON logging

## Metrics

Gateway:
- `llm_gateway_request_count_total` (status label)
- `llm_gateway_inflight_requests`
- `llm_gateway_batch_size` histogram
- `llm_gateway_request_latency_seconds` histogram
- `llm_gateway_request_latency_p50_seconds`
- `llm_gateway_request_latency_p95_seconds`
- `llm_gateway_request_latency_p99_seconds`
- `llm_gateway_tokens_generated_total`
- `llm_gateway_errors_total` (code label)

Worker:
- `llm_worker_batch_size` histogram
- `llm_worker_tokens_generated_total`
- `llm_worker_errors_total`

## Tests

Unit:

```bash
make test
```

Integration:

```bash
make test-integration
```

Chaos:

```bash
make test-chaos
```

## Make Targets

- `make fmt`
- `make lint`
- `make test`
- `make test-integration`
- `make test-chaos`
- `make run`
- `make loadtest`
- `make benchmark`
- `make demo`

## Design Tradeoffs

1. Redis queue/pubsub instead of Kafka/NATS:
Simple local operations and low setup cost; weaker delivery guarantees and replay semantics.
2. Worker-side micro-batching by `(model,max_tokens)`:
Improves throughput and amortizes overhead; adds slight queueing delay and higher tail latency under skew.
3. Fixed-window rate limiting:
Cheap Redis ops and easy operability; boundary bursts are possible versus token-bucket/sliding-window approaches.
4. Gateway-local rolling p50/p95/p99 gauges:
Fast and dependency-free; quantiles are per-instance approximations, not globally exact across replicas.
5. Mock model backend by default:
Fast to run and test locally; does not reflect true model runtime memory/compute behavior until real backend is plugged in.

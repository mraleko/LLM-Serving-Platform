from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (pct / 100)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


async def send_request(
    client: httpx.AsyncClient,
    *,
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    stream: bool,
) -> tuple[float, int]:
    started = time.perf_counter()
    if stream:
        token_events = 0
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = json.loads(line[5:].strip())
                if data.get("token") is not None:
                    token_events += 1
                if data.get("finish_reason") is not None:
                    break
        return time.perf_counter() - started, token_events

    response = await client.post(url, json=payload, headers=headers)
    response.raise_for_status()
    body = response.json()
    return time.perf_counter() - started, int(body.get("generated_tokens", 0))


def build_summary(
    *,
    latencies: list[float],
    failures: int,
    requests: int,
    elapsed: float,
    generated_tokens: int,
    mode: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    successes = requests - failures
    throughput_rps = successes / elapsed if elapsed > 0 else 0.0
    token_throughput = generated_tokens / elapsed if elapsed > 0 else 0.0
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "config": config,
        "requests_total": requests,
        "requests_success": successes,
        "requests_failed": failures,
        "elapsed_seconds": round(elapsed, 4),
        "throughput_rps": round(throughput_rps, 4),
        "token_throughput_tps": round(token_throughput, 4),
        "latency_seconds": {
            "mean": round(statistics.mean(latencies), 6) if latencies else 0.0,
            "p50": round(percentile(latencies, 50), 6),
            "p95": round(percentile(latencies, 95), 6),
            "p99": round(percentile(latencies, 99), 6),
            "max": round(max(latencies), 6) if latencies else 0.0,
        },
    }


def print_summary(summary: dict[str, Any]) -> None:
    latency = summary["latency_seconds"]
    print("Benchmark Summary")
    print(f"  Mode:              {summary['mode']}")
    print(f"  Requests:          {summary['requests_total']}")
    print(f"  Success / Failed:  {summary['requests_success']} / {summary['requests_failed']}")
    print(f"  Elapsed:           {summary['elapsed_seconds']:.4f}s")
    print(f"  Throughput:        {summary['throughput_rps']:.2f} req/s")
    print(f"  Token Throughput:  {summary['token_throughput_tps']:.2f} tok/s")
    print(
        "  Latency (s):       "
        f"p50={latency['p50']:.4f}, p95={latency['p95']:.4f}, p99={latency['p99']:.4f}, "
        f"mean={latency['mean']:.4f}, max={latency['max']:.4f}"
    )


def write_summary(summary: dict[str, Any], out_path: str | None) -> None:
    if not out_path:
        return
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote benchmark results to {path}")


async def run_real_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": args.stream,
    }
    headers: dict[str, str] = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    latencies: list[float] = []
    generated_tokens = 0
    failures = 0

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0)) as client:
        # Warmup primes connection pools and app internals.
        for _ in range(args.warmup):
            try:
                await send_request(
                    client,
                    url=args.url,
                    payload=payload,
                    headers=headers,
                    stream=args.stream,
                )
            except Exception:
                pass

        semaphore = asyncio.Semaphore(args.concurrency)
        started = time.perf_counter()

        async def one_request(i: int) -> None:
            nonlocal generated_tokens, failures
            async with semaphore:
                try:
                    latency, tokens = await send_request(
                        client,
                        url=args.url,
                        payload=payload,
                        headers=headers,
                        stream=args.stream,
                    )
                    latencies.append(latency)
                    generated_tokens += tokens
                except Exception as exc:
                    failures += 1
                    print(f"request {i} failed: {exc}", file=sys.stderr)

        tasks = [asyncio.create_task(one_request(i)) for i in range(args.requests)]
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - started

    return build_summary(
        latencies=latencies,
        failures=failures,
        requests=args.requests,
        elapsed=elapsed,
        generated_tokens=generated_tokens,
        mode="real",
        config={
            "url": args.url,
            "concurrency": args.concurrency,
            "warmup": args.warmup,
            "stream": args.stream,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
    )


def run_synthetic_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    latencies: list[float] = []
    failures = 0
    generated_tokens = 0

    # Synthetic mode is deterministic and useful in CI/sandboxes without docker/redis.
    for i in range(args.requests):
        base = rng.gauss(0.14, 0.03)
        if i % 25 == 0:
            base += rng.uniform(0.1, 0.25)  # long-tail jitter
        latency = max(base, 0.02)
        latencies.append(latency)
        if rng.random() < args.synthetic_failure_rate:
            failures += 1
        else:
            generated_tokens += args.max_tokens if args.stream else args.max_tokens // 2

    simulated_parallelism = max(args.concurrency, 1)
    elapsed = sum(latencies) / simulated_parallelism
    return build_summary(
        latencies=latencies,
        failures=failures,
        requests=args.requests,
        elapsed=elapsed,
        generated_tokens=generated_tokens,
        mode="synthetic",
        config={
            "seed": args.seed,
            "concurrency": args.concurrency,
            "stream": args.stream,
            "max_tokens": args.max_tokens,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark against /v1/generate.")
    parser.add_argument("--url", default="http://localhost:8000/v1/generate")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--prompt", default="Benchmark prompt for LLM serving.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--out", default="artifacts/benchmark/latest.json")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run deterministic synthetic benchmark without calling the HTTP API.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-failure-rate", type=float, default=0.0)
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    summary = (
        run_synthetic_benchmark(args)
        if args.synthetic
        else await run_real_benchmark(args)
    )
    print_summary(summary)
    write_summary(summary, args.out)
    return 1 if summary["requests_failed"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

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


async def run_load_test(args: argparse.Namespace) -> int:
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
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0)) as client:
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
    successes = args.requests - failures
    throughput = successes / elapsed if elapsed > 0 else 0.0
    p95_latency = percentile(latencies, 95)

    print(f"Requests: {args.requests}")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"Throughput: {throughput:.2f} req/s")
    print(f"P95 latency: {p95_latency:.3f}s")
    print(f"Tokens generated: {generated_tokens}")
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple concurrency load test for /v1/generate")
    parser.add_argument("--url", default="http://localhost:8000/v1/generate")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--prompt", default="Write a short response about distributed systems.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--stream", action="store_true", help="Use streaming SSE requests")
    parser.add_argument("--api-key", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_load_test(parse_args())))

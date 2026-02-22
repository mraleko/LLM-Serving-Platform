from __future__ import annotations

import argparse
import json
import time
from urllib.parse import urlsplit, urlunsplit

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one streaming request and print tokens.")
    parser.add_argument("--url", default="http://localhost:8000/v1/generate")
    parser.add_argument("--prompt", default="Explain why batching helps LLM serving.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--wait-seconds", type=float, default=60.0)
    parser.add_argument("--retry-interval", type=float, default=1.0)
    return parser.parse_args()


def _healthz_url(generate_url: str) -> str:
    parts = urlsplit(generate_url)
    return urlunsplit((parts.scheme, parts.netloc, "/healthz", "", ""))


def wait_for_gateway_ready(*, generate_url: str, wait_seconds: float, retry_interval: float) -> None:
    health_url = _healthz_url(generate_url)
    deadline = time.time() + wait_seconds
    last_error: str | None = None
    while time.time() < deadline:
        try:
            response = requests.get(health_url, timeout=3)
            if response.status_code == 200:
                return
            last_error = f"status={response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(retry_interval)
    raise RuntimeError(f"gateway not ready within {wait_seconds}s ({health_url}, last_error={last_error})")


def main() -> int:
    args = parse_args()
    payload = {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
    }
    headers: dict[str, str] = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    wait_for_gateway_ready(
        generate_url=args.url,
        wait_seconds=args.wait_seconds,
        retry_interval=args.retry_interval,
    )

    with requests.post(args.url, json=payload, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        print("Streaming tokens:")
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = json.loads(line[5:].strip())
            if data.get("token") is not None:
                print(data["token"], end="", flush=True)
            if data.get("finish_reason") is not None:
                break
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

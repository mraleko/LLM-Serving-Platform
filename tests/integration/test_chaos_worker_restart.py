from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

import pytest
import requests

pytestmark = [pytest.mark.integration, pytest.mark.chaos]


def _wait_for_health(base_url: str, timeout_seconds: float = 120.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/healthz", timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise AssertionError("gateway did not become healthy")


def _stream_request_tokens(
    *,
    url: str,
    prompt: str,
    max_tokens: int,
    timeout: float = 90,
) -> tuple[int, bool]:
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }
    token_events = 0
    done = False
    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as response:
            if response.status_code != 200:
                return 0, False
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                try:
                    data = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                if data.get("token") is not None:
                    token_events += 1
                if data.get("finish_reason") is not None:
                    done = True
                    break
                if data.get("code") is not None and data.get("message") is not None:
                    break
    except requests.RequestException:
        return 0, False
    return token_events, done


@pytest.mark.integration
@pytest.mark.chaos
def test_worker_restart_chaos_recovery() -> None:
    if os.getenv("RUN_INTEGRATION") != "1":
        pytest.skip("Set RUN_INTEGRATION=1 to run compose integration tests.")
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed")

    compose_up = ["docker", "compose", "up", "--build", "-d"]
    compose_down = ["docker", "compose", "down", "-v", "--remove-orphans"]
    stream_url = "http://localhost:8000/v1/generate"

    subprocess.run(compose_up, check=True)
    try:
        _wait_for_health("http://localhost:8000", timeout_seconds=120)

        # Start one stream and kill worker after first token to simulate node failure.
        payload = {
            "prompt": "Chaos request that should be interrupted by worker restart.",
            "max_tokens": 128,
            "temperature": 0.7,
            "stream": True,
        }
        saw_first_token = False
        with requests.post(stream_url, json=payload, stream=True, timeout=90) as response:
            assert response.status_code == 200
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = json.loads(line[5:].strip())
                if data.get("token") is not None:
                    saw_first_token = True
                    subprocess.run(["docker", "compose", "kill", "worker"], check=True)
                    break
        assert saw_first_token

        subprocess.run(["docker", "compose", "up", "-d", "worker"], check=True)

        deadline = time.time() + 120
        recovered = False
        while time.time() < deadline:
            tokens, done = _stream_request_tokens(
                url=stream_url,
                prompt="Post-chaos recovery request",
                max_tokens=24,
                timeout=30,
            )
            if tokens >= 2 and done:
                recovered = True
                break
            time.sleep(2)

        assert recovered, "service did not recover after worker restart chaos"
    finally:
        subprocess.run(compose_down, check=False)

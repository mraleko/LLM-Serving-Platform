from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

import pytest
import requests

pytestmark = pytest.mark.integration


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


@pytest.mark.integration
def test_streaming_request_emits_multiple_tokens_via_docker_compose() -> None:
    if os.getenv("RUN_INTEGRATION") != "1":
        pytest.skip("Set RUN_INTEGRATION=1 to run compose integration tests.")
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed")

    compose_up = ["docker", "compose", "up", "--build", "-d"]
    compose_down = ["docker", "compose", "down", "-v", "--remove-orphans"]

    subprocess.run(compose_up, check=True)
    try:
        _wait_for_health("http://localhost:8000", timeout_seconds=120)

        payload = {
            "prompt": "Integration test streaming response",
            "max_tokens": 16,
            "temperature": 0.7,
            "stream": True,
        }
        token_events = 0
        saw_done = False
        with requests.post(
            "http://localhost:8000/v1/generate",
            json=payload,
            stream=True,
            timeout=60,
        ) as response:
            assert response.status_code == 200
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = json.loads(line[5:].strip())
                if data.get("token") is not None:
                    token_events += 1
                if data.get("finish_reason") is not None:
                    saw_done = True
                    break

        assert token_events >= 2
        assert saw_done
    finally:
        subprocess.run(compose_down, check=False)

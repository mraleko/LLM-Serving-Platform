from __future__ import annotations

import asyncio
import hashlib
from typing import AsyncIterator

from common.schemas import RequestEnvelope

_VOCAB = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "lambda",
    "omega",
]


class MockTokenBackend:
    def __init__(self, *, token_delay_seconds: float = 0.02) -> None:
        self._token_delay_seconds = token_delay_seconds

    async def stream_batch_tokens(
        self,
        requests: list[RequestEnvelope],
    ) -> AsyncIterator[tuple[RequestEnvelope, int, str]]:
        max_tokens = max((request.max_tokens for request in requests), default=0)
        for index in range(max_tokens):
            for request in requests:
                if index >= request.max_tokens:
                    continue
                yield request, index, self._build_token(request.prompt, index, request.temperature)
            await asyncio.sleep(self._token_delay_seconds)

    def _build_token(self, prompt: str, index: int, temperature: float) -> str:
        seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
        vocab_index = (seed + index) % len(_VOCAB)
        token = _VOCAB[vocab_index]
        if temperature > 1.2 and index % 4 == 0:
            token = token.upper()
        return f"{token} "

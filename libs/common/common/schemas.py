from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1, max_length=8000)
    max_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = Field(default=False)
    model: str = Field(default="mock-v1", min_length=1, max_length=128)


class GenerateResponse(BaseModel):
    request_id: str
    model: str
    text: str
    generated_tokens: int


class RequestEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    created_at: float
    prompt: str
    max_tokens: int
    temperature: float
    model: str
    stream: bool = False


class WorkerEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["batch", "token", "end", "error"]
    request_id: str
    token: str | None = None
    index: int | None = None
    generated_tokens: int | None = None
    finish_reason: str | None = None
    error: str | None = None
    batch_size: int | None = None

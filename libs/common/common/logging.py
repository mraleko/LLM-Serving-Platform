from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

_RESERVED_LOG_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class JsonFormatter(logging.Formatter):
    def __init__(self, service_name: str) -> None:
        super().__init__()
        self._service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self._service_name,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extras: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_KEYS:
                extras[key] = value
        if extras:
            payload["extra"] = extras

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(service_name: str, level: str = "INFO") -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service_name=service_name))
    root_logger.addHandler(handler)

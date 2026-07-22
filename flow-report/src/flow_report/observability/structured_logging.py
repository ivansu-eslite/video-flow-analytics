from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel


def _normalize_log_value(value: Any) -> Any:
    """Convert application objects into JSON-safe logging values."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_log_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_normalize_log_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class StructuredLogger:
    """Emit one-line JSON logs that Cloud Logging can parse reliably."""

    def __init__(self, *, component: str) -> None:
        self.component = component

    def info(self, message: str, **fields: Any) -> None:
        """Write an informational structured log record."""
        self._emit("INFO", message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        """Write a warning structured log record."""
        self._emit("WARNING", message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        """Write an error structured log record."""
        self._emit("ERROR", message, **fields)

    def exception(
        self,
        message: str,
        *,
        error: BaseException,
        **fields: Any,
    ) -> None:
        """Write an exception record with stack trace metadata."""
        self._emit(
            "ERROR",
            message,
            error={
                "type": type(error).__name__,
                "message": str(error),
                "stacktrace": "".join(traceback.format_exception(error)).strip(),
            },
            **fields,
        )

    def _emit(self, severity: str, message: str, **fields: Any) -> None:
        """Serialize a log record as a single JSON line for Cloud Logging."""
        payload = {
            "severity": severity,
            "message": message,
            "component": self.component,
            **{
                key: _normalize_log_value(value)
                for key, value in fields.items()
            },
        }
        # ensure_ascii=False：本包的 log 訊息是中文，逸出成 \uXXXX 會讓地端直接看
        # 終端機的維運性變差；輸出仍是合法的 UTF-8 JSON，解析結果完全相同。
        output = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        stream = sys.stderr if severity in {"ERROR", "CRITICAL"} else sys.stdout
        print(output, file=stream, flush=True)

"""Structured JSON logs (design doc: Monitoring → logs).

THERAFLO_ADS_LOG_FORMAT=json emits one JSON object per line. Whatever the
format, a redaction filter removes bearer tokens, API keys, and 32-byte hex
secrets from messages before they are written: logs are copied to more places
than any database, so they must never hold a credential.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime

_REDACT = re.compile(
    r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+|(x-advertiser-key[\"']?\s*[:=]\s*[\"']?)[^\s\"',]+"
    r"|\b[0-9a-f]{64}\b|(sk-ant-[A-Za-z0-9-]{4})[A-Za-z0-9_-]+"
)
_STANDARD = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()) | {
    "message",
    "asctime",
    "color_message",  # uvicorn's ANSI-colored duplicate of msg
}


def redact(text: str) -> str:
    def sub(match: re.Match[str]) -> str:
        prefix = match.group(1) or match.group(2) or match.group(3) or ""
        return f"{prefix}[redacted]"

    return _REDACT.sub(sub, text)


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        cleaned = redact(message)
        if cleaned != message:
            record.msg, record.args = cleaned, None
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD and not key.startswith("_"):
                data[key] = value
        if record.exc_info:
            data["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(data, default=str)


def configure_logging(fmt: str | None = None, level: str | None = None) -> None:
    fmt = (fmt or os.environ.get("THERAFLO_ADS_LOG_FORMAT", "text")).lower()
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel((level or os.environ.get("THERAFLO_ADS_LOG_LEVEL", "INFO")).upper())

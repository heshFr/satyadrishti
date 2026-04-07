"""
Satya Drishti — Logging Configuration
=======================================
Structured logging with JSON format for production (log aggregation via
Loki/CloudWatch/Papertrail) and human-readable format for development.

Set SATYA_LOG_FORMAT=json for structured JSON logs.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields if any
        for key in ("request_id", "client_ip", "endpoint", "latency",
                     "modality", "verdict", "confidence", "scan_id",
                     "user_id", "status_code"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        return json.dumps(log_entry, default=str)


class HumanFormatter(logging.Formatter):
    """Human-readable colored format for development."""

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET
        return f"{color}[{record.levelname}]{reset} {record.name} — {record.getMessage()}"


def setup_logging(level: str = "INFO"):
    """Configure root logger for the application."""
    log_format = os.environ.get("SATYA_LOG_FORMAT", "human")

    handler = logging.StreamHandler(sys.stdout)
    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(HumanFormatter())

    root = logging.getLogger("satyadrishti")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers on reload
    if not root.handlers:
        root.addHandler(handler)
    root.propagate = False

    return root

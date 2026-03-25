"""
Satya Drishti — Logging Configuration
=======================================
Centralized logging setup. All modules use `logging.getLogger(__name__)`.
"""

import logging
import sys


def setup_logging(level: str = "INFO"):
    """Configure root logger for the application."""
    fmt = "[%(levelname)s] %(name)s — %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger("satyadrishti")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)
    root.propagate = False

    return root

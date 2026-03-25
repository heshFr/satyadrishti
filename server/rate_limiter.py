"""
Satya Drishti — Rate Limiter
=============================
Simple in-memory sliding window rate limiter for FastAPI.
No external dependencies required.
"""

import time
from collections import defaultdict
from functools import wraps

from fastapi import HTTPException, Request


class RateLimiter:
    """
    In-memory sliding window rate limiter.

    Args:
        default_limit: Default requests per window
        default_window: Default window in seconds
    """

    def __init__(self, default_limit: int = 60, default_window: int = 60):
        self.default_limit = default_limit
        self.default_window = default_window
        # {client_key: [(timestamp, ...), ...]}
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _client_key(self, request: Request, endpoint: str = "") -> str:
        """Get rate limit key from client IP + endpoint."""
        client_ip = request.client.host if request.client else "unknown"
        return f"{client_ip}:{endpoint}"

    def _cleanup(self, key: str, window: int):
        """Remove expired entries outside the window."""
        cutoff = time.time() - window
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]

    def check(self, request: Request, limit: int = None, window: int = None, endpoint: str = ""):
        """
        Check if request is within rate limits. Raises 429 if exceeded.

        Args:
            request: FastAPI request
            limit: Max requests per window (default: self.default_limit)
            window: Window size in seconds (default: self.default_window)
            endpoint: Endpoint identifier for per-route limiting
        """
        limit = limit or self.default_limit
        window = window or self.default_window
        key = self._client_key(request, endpoint)

        self._cleanup(key, window)

        if len(self._requests[key]) >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {limit} requests per {window}s.",
                headers={"Retry-After": str(window)},
            )

        self._requests[key].append(time.time())

    def cleanup_all(self):
        """Remove all expired entries. Call periodically to free memory."""
        now = time.time()
        max_window = self.default_window * 2
        empty_keys = []
        for key in self._requests:
            self._requests[key] = [t for t in self._requests[key] if now - t < max_window]
            if not self._requests[key]:
                empty_keys.append(key)
        for key in empty_keys:
            del self._requests[key]


# Global rate limiter instance
limiter = RateLimiter(default_limit=60, default_window=60)

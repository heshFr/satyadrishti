"""
Satya Drishti — Request Middleware
===================================
Adds request ID tracking, latency measurement, structured logging,
security headers, and metrics collection to every request.
"""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from . import metrics as m

log = logging.getLogger("satyadrishti.middleware")


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """
    Adds to every request:
    - X-Request-ID header (for tracing across services)
    - X-Response-Time header (latency in ms)
    - Prometheus metrics (latency histogram, status counter)
    - Structured access log
    """

    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:12])
        start_time = time.time()

        # Determine endpoint label (strip path params for cardinality control)
        path = request.url.path
        endpoint = _normalize_path(path)
        method = request.method

        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as exc:
            status = 500
            m.request_errors.inc(labels={"endpoint": endpoint, "error": type(exc).__name__})
            raise
        finally:
            latency = time.time() - start_time

            # Record metrics
            m.request_latency.observe(latency, {"endpoint": endpoint, "method": method})
            m.request_total.inc(labels={"endpoint": endpoint, "status": str(status)})

            if status >= 400:
                error_type = "client_error" if status < 500 else "server_error"
                m.request_errors.inc(labels={"endpoint": endpoint, "error": error_type})

            # Structured access log (only for API endpoints, skip metrics/health)
            if not path.startswith("/metrics") and path != "/api/health":
                log.info(
                    "request_id=%s method=%s path=%s status=%d latency=%.3fs client=%s",
                    request_id, method, path, status, latency,
                    request.client.host if request.client else "unknown",
                )

        # Add response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{latency * 1000:.1f}ms"
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Prevent MIME sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        # XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Permissions policy (disable unnecessary browser features)
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        # HSTS (Strict Transport Security) — enforce HTTPS
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        # Content Security Policy for API
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        # Cache control for API responses
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"

        return response


def _normalize_path(path: str) -> str:
    """
    Normalize URL path for metrics labels to prevent high cardinality.
    /api/scans/abc123 → /api/scans/{id}
    """
    parts = path.strip("/").split("/")
    normalized = []
    for i, part in enumerate(parts):
        # Replace UUIDs and IDs with placeholder
        if len(part) > 20 or (len(part) > 8 and "-" in part):
            normalized.append("{id}")
        elif part.isdigit():
            normalized.append("{id}")
        else:
            normalized.append(part)
    return "/" + "/".join(normalized)

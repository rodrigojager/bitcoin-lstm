from __future__ import annotations

import time
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram


REQ_COUNT = Counter("http_requests_total", "Total de requests HTTP", ["method", "path", "status"])
REQ_LATENCY = Histogram("http_request_seconds", "Latência HTTP (segundos)", ["method", "path"])


def instrument_app(app: FastAPI) -> None:
    @app.middleware("http")
    async def _prometheus_middleware(request: Request, call_next):
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = getattr(response, "status_code", 500)
            return response
        finally:
            elapsed = time.perf_counter() - start
            path = request.url.path
            try:
                REQ_LATENCY.labels(request.method, path).observe(elapsed)
                REQ_COUNT.labels(request.method, path, str(status)).inc()
            except Exception:
                pass








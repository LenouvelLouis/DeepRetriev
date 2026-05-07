"""FastAPI application factory for RAGLab."""

import json
import logging
import time
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.endpoints import router
from src.api.metrics import record_request

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status code, and duration."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        log_data = {
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_s": round(duration, 6),
        }
        logger.info(json.dumps(log_data))

        # Feed the metrics collector
        record_request(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration=duration,
        )

        return response


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    app = FastAPI(
        title="RAGLab API",
        description=(
            "Retrieval-Augmented Generation pipeline API. "
            "Query a knowledge base built from Wikipedia, ingest new documents, "
            "run evaluations, and inspect system health."
        ),
        version="0.3.0",
    )

    # CORS — allow all origins for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging + metrics
    app.add_middleware(RequestLoggingMiddleware)

    # Routes
    app.include_router(router)

    return app

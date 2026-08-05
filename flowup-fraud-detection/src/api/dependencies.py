"""FastAPI dependency injection and middleware.

Provides:
- Singleton predictor instance via FastAPI's dependency injection
- Correlation ID middleware for request tracing
"""

import logging
import uuid
from typing import AsyncGenerator

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.core.logging_config import correlation_id_var
from src.models.predictor import FraudPredictor

logger = logging.getLogger(__name__)

# Module-level predictor singleton (set during app startup)
_predictor_instance: FraudPredictor | None = None


def set_predictor(predictor: FraudPredictor) -> None:
    """Register the global predictor instance.

    Called during application startup to initialize the singleton.

    Args:
        predictor: Initialized FraudPredictor instance.
    """
    global _predictor_instance
    _predictor_instance = predictor
    logger.info("Predictor registered in dependency injection")


def get_predictor() -> FraudPredictor:
    """FastAPI dependency that provides the predictor singleton.

    Returns:
        FraudPredictor: The loaded predictor instance.

    Raises:
        RuntimeError: If the predictor hasn't been initialized.
    """
    if _predictor_instance is None:
        raise RuntimeError(
            "Predictor not initialized. Ensure the model is loaded at startup."
        )
    return _predictor_instance


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns a unique correlation ID to each request.

    The correlation ID is:
    1. Read from the X-Correlation-ID header (if provided by the client)
    2. Generated as a UUID4 if not provided
    3. Stored in contextvars for structured logging
    4. Returned in the response headers
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process the request, injecting a correlation ID.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            Response: HTTP response with X-Correlation-ID header.
        """
        # Extract or generate correlation ID
        correlation_id = request.headers.get(
            "X-Correlation-ID", str(uuid.uuid4())
        )
        correlation_id_var.set(correlation_id)

        # Process request
        response = await call_next(request)

        # Include correlation ID in response headers
        response.headers["X-Correlation-ID"] = correlation_id

        return response

"""FastAPI application factory and entrypoint.

Creates the FastAPI app with:
- Lifespan management (model loading on startup)
- CORS middleware
- Correlation ID middleware
- Router registration
- Uvicorn server configuration
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import CorrelationIdMiddleware, set_predictor
from src.api.routes import router
from src.core.config import get_settings
from src.core.logging_config import setup_logging
from src.models.predictor import FraudPredictor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: load model on startup, cleanup on shutdown.

    Initializes structured logging and loads the trained model artifact
    before the application begins serving requests.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Control returns to FastAPI to handle requests.
    """
    settings = get_settings()

    # Initialize structured logging
    setup_logging(
        log_level=settings.log_level,
        log_dir=Path("logs"),
    )

    logger.info(
        "Application starting",
        extra={
            "model_path": str(settings.model_path),
            "threshold": settings.threshold,
            "version": settings.model_version,
        },
    )

    # Load model
    try:
        predictor = FraudPredictor(
            model_path=settings.model_path,
            threshold_override=settings.threshold,
        )
        set_predictor(predictor)
        logger.info("Model loaded successfully, application ready")
    except FileNotFoundError:
        logger.warning(
            "Model artifact not found. Run 'make train' to generate it. "
            "The /predict endpoint will return 500 until a model is available.",
            extra={"model_path": str(settings.model_path)},
        )

    yield

    # Shutdown
    logger.info("Application shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="FlowUp Fraud Detection API",
        description=(
            "Real-time credit card fraud detection microservice. "
            "Evaluates transactions using an ML pipeline (XGBoost/RandomForest) "
            "trained on PCA-transformed credit card data."
        ),
        version=settings.model_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Correlation ID middleware
    app.add_middleware(CorrelationIdMiddleware)

    # Register routes
    app.include_router(router)

    return app


# Application instance
app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )

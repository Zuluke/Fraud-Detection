"""API route definitions for the fraud detection service.

Endpoints:
- GET  /health  → Service health check
- POST /predict → Real-time fraud prediction
"""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from src.api.dependencies import get_predictor
from src.api.schemas import HealthResponse, PredictionResponse, TransactionInput
from src.core.config import get_settings
from src.models.predictor import FraudPredictor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the service health status, version, and current timestamp.",
    tags=["monitoring"],
)
async def health_check() -> HealthResponse:
    """Check if the service is healthy and responsive.

    Returns:
        HealthResponse: Service status with version and timestamp.
    """
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.model_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict fraud probability",
    description=(
        "Receives a credit card transaction with 30 features and returns "
        "a fraud probability score with an approve/deny decision."
    ),
    tags=["predictions"],
)
async def predict_fraud(
    transaction: TransactionInput,
    predictor: FraudPredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Evaluate a transaction for fraud risk.

    Processes the transaction through the ML pipeline and returns:
    - Fraud probability [0, 1]
    - Binary decision based on optimized threshold
    - Latency measurement for SLA monitoring

    Args:
        transaction: Validated transaction input with 30 features.
        predictor: Injected FraudPredictor singleton.

    Returns:
        PredictionResponse: Complete prediction result with metadata.
    """
    settings = get_settings()

    # Run prediction
    features = transaction.to_feature_dict()
    result = predictor.predict(features)

    # Build response
    transaction_id = str(uuid.uuid4())

    response = PredictionResponse(
        transaction_id=transaction_id,
        probability=result["probability"],
        decision=result["decision"],
        threshold_used=predictor.threshold,
        latency_ms=result["latency_ms"],
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_version=settings.model_version,
    )

    logger.info(
        "Prediction served",
        extra={
            "transaction_id": transaction_id,
            "decision": result["decision"],
            "probability": result["probability"],
            "latency_ms": result["latency_ms"],
            "input_hash": result["input_hash"],
        },
    )

    return response

"""Pydantic schemas for API request/response validation.

Defines strict validation rules for transaction input:
- V1–V28: floats in [-20, 20] (typical PCA-transformed range)
- Time: non-negative float (seconds since first transaction)
- Amount: non-negative float (transaction value)

Response schemas include all prediction metadata for audit trails.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """Input schema for a single credit card transaction.

    All 30 fields are required. Validation enforces value ranges
    consistent with PCA-transformed credit card fraud data.

    Attributes:
        Time: Seconds elapsed since first transaction in dataset (>= 0).
        V1–V28: PCA-transformed features, typically in [-20, 20].
        Amount: Transaction amount in currency units (>= 0).
    """

    Time: float = Field(..., ge=0, description="Seconds elapsed since first transaction")
    V1: float = Field(..., ge=-20, le=20, description="PCA feature V1")
    V2: float = Field(..., ge=-20, le=20, description="PCA feature V2")
    V3: float = Field(..., ge=-20, le=20, description="PCA feature V3")
    V4: float = Field(..., ge=-20, le=20, description="PCA feature V4")
    V5: float = Field(..., ge=-20, le=20, description="PCA feature V5")
    V6: float = Field(..., ge=-20, le=20, description="PCA feature V6")
    V7: float = Field(..., ge=-20, le=20, description="PCA feature V7")
    V8: float = Field(..., ge=-20, le=20, description="PCA feature V8")
    V9: float = Field(..., ge=-20, le=20, description="PCA feature V9")
    V10: float = Field(..., ge=-20, le=20, description="PCA feature V10")
    V11: float = Field(..., ge=-20, le=20, description="PCA feature V11")
    V12: float = Field(..., ge=-20, le=20, description="PCA feature V12")
    V13: float = Field(..., ge=-20, le=20, description="PCA feature V13")
    V14: float = Field(..., ge=-20, le=20, description="PCA feature V14")
    V15: float = Field(..., ge=-20, le=20, description="PCA feature V15")
    V16: float = Field(..., ge=-20, le=20, description="PCA feature V16")
    V17: float = Field(..., ge=-20, le=20, description="PCA feature V17")
    V18: float = Field(..., ge=-20, le=20, description="PCA feature V18")
    V19: float = Field(..., ge=-20, le=20, description="PCA feature V19")
    V20: float = Field(..., ge=-20, le=20, description="PCA feature V20")
    V21: float = Field(..., ge=-20, le=20, description="PCA feature V21")
    V22: float = Field(..., ge=-20, le=20, description="PCA feature V22")
    V23: float = Field(..., ge=-20, le=20, description="PCA feature V23")
    V24: float = Field(..., ge=-20, le=20, description="PCA feature V24")
    V25: float = Field(..., ge=-20, le=20, description="PCA feature V25")
    V26: float = Field(..., ge=-20, le=20, description="PCA feature V26")
    V27: float = Field(..., ge=-20, le=20, description="PCA feature V27")
    V28: float = Field(..., ge=-20, le=20, description="PCA feature V28")
    Amount: float = Field(..., ge=0, description="Transaction amount (>= 0)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Time": 406.0,
                    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
                    "V5": -0.34, "V6": -0.47, "V7": 0.24, "V8": 0.10,
                    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
                    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
                    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
                    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": -0.34,
                    "V25": -0.05, "V26": -0.23, "V27": 0.04, "V28": 0.01,
                    "Amount": 149.62,
                }
            ]
        }
    }

    def to_feature_dict(self) -> dict[str, float]:
        """Convert the validated input to a feature dictionary.

        Returns the features in the exact order expected by the
        preprocessing pipeline.

        Returns:
            dict[str, float]: Ordered feature name → value mapping.
        """
        return self.model_dump()


class PredictionResponse(BaseModel):
    """Response schema for a fraud prediction.

    Contains the prediction result along with metadata for
    audit logging and debugging.

    Attributes:
        transaction_id: Unique UUID for this prediction request.
        probability: Fraud probability score [0.0, 1.0].
        decision: Binary decision: "approved" or "suspected_fraud".
        threshold_used: Decision threshold applied.
        latency_ms: Total prediction latency in milliseconds.
        timestamp: ISO 8601 timestamp of the prediction.
        model_version: Semantic version of the deployed model.
    """

    transaction_id: str = Field(..., description="Unique transaction UUID")
    probability: float = Field(..., ge=0.0, le=1.0, description="Fraud probability")
    decision: str = Field(..., description="approved | suspected_fraud")
    threshold_used: float = Field(..., description="Decision threshold used")
    latency_ms: float = Field(..., description="Prediction latency in ms")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    model_version: str = Field(..., description="Model version identifier")


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint.

    Attributes:
        status: Service health status ("healthy").
        version: Application version.
        timestamp: ISO 8601 timestamp.
    """

    status: str = Field(default="healthy", description="Service status")
    version: str = Field(default="1.0.0", description="Application version")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Current UTC timestamp",
    )

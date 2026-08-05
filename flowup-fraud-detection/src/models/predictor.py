"""Real-time fraud prediction service.

Loads a trained model artifact (model + preprocessing pipeline + metadata)
and provides thread-safe, low-latency predictions with timing measurement.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FraudPredictor:
    """Loads and serves fraud predictions from a trained model artifact.

    The artifact contains:
    - model: trained classifier (XGBoost or RandomForest)
    - pipeline: fitted preprocessing pipeline
    - threshold: optimized decision threshold
    - feature_names: expected input feature order

    Attributes:
        model: The trained classifier.
        pipeline: The fitted preprocessing pipeline.
        threshold: Decision threshold for fraud classification.
        model_name: Name of the selected model architecture.
        feature_names: Ordered list of expected input features.
    """

    def __init__(self, model_path: Path, threshold_override: float | None = None) -> None:
        """Initialize predictor by loading the model artifact.

        Args:
            model_path: Path to the .joblib artifact file.
            threshold_override: If provided, overrides the artifact's threshold.

        Raises:
            FileNotFoundError: If model_path does not exist.
            KeyError: If the artifact is missing required keys.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        logger.info("Loading model artifact", extra={"path": str(model_path)})
        artifact: dict[str, Any] = joblib.load(model_path)

        self.model = artifact["model"]
        self.pipeline = artifact["pipeline"]
        self.threshold = threshold_override or artifact.get("threshold", 0.5)
        self.model_name: str = artifact.get("model_name", "unknown")
        self.feature_names: list[str] = artifact.get(
            "feature_names",
            ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"],
        )

        logger.info(
            "Model loaded successfully",
            extra={
                "model_name": self.model_name,
                "threshold": self.threshold,
                "n_features": len(self.feature_names),
            },
        )

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        """Run a single fraud prediction.

        Processes the input through the preprocessing pipeline,
        generates a probability score, and applies the threshold
        to make a binary decision.

        Args:
            features: Dictionary mapping feature names to float values.
                      Must contain all features in self.feature_names.

        Returns:
            dict[str, Any]: Prediction result with keys:
                - probability (float): Fraud probability [0, 1]
                - decision (str): "approved" or "suspected_fraud"
                - latency_ms (float): Processing time in milliseconds
                - input_hash (str): SHA256 of input for audit logging
        """
        start_time = time.perf_counter()

        # Build input DataFrame with correct column order
        input_df = pd.DataFrame([features], columns=self.feature_names)

        # Apply preprocessing pipeline
        transformed = self.pipeline.transform(input_df)

        # Get fraud probability
        probabilities = self.model.predict_proba(transformed)
        fraud_probability = float(probabilities[0, 1])

        # Apply threshold
        decision = "suspected_fraud" if fraud_probability >= self.threshold else "approved"

        # Calculate latency
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Generate input hash for audit logging
        input_hash = hashlib.sha256(
            json.dumps(features, sort_keys=True).encode()
        ).hexdigest()[:16]

        result = {
            "probability": round(fraud_probability, 6),
            "decision": decision,
            "latency_ms": latency_ms,
            "input_hash": input_hash,
        }

        logger.info(
            "Prediction completed",
            extra={
                "input_hash": input_hash,
                "decision": decision,
                "probability": result["probability"],
                "latency_ms": latency_ms,
            },
        )

        return result

"""Input validation helpers and data integrity utilities.

Provides additional validation logic beyond Pydantic schema enforcement,
including feature range checking and input hashing for audit logs.
"""

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Expected feature ranges based on the Kaggle credit card fraud dataset
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "Time": (0.0, float("inf")),
    "Amount": (0.0, float("inf")),
    **{f"V{i}": (-20.0, 20.0) for i in range(1, 29)},
}

REQUIRED_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def validate_feature_ranges(features: dict[str, float]) -> list[str]:
    """Validate that all feature values fall within expected ranges.

    Performs range checking beyond basic Pydantic validation. Returns
    a list of human-readable error messages for any violations.

    Args:
        features: Dictionary of feature name → value pairs.

    Returns:
        list[str]: List of validation error messages (empty if all valid).
    """
    errors: list[str] = []

    for name, (low, high) in FEATURE_RANGES.items():
        if name not in features:
            errors.append(f"Missing required feature: {name}")
            continue

        value = features[name]
        if not isinstance(value, (int, float)):
            errors.append(f"{name}: expected numeric, got {type(value).__name__}")
            continue

        if value < low or value > high:
            errors.append(f"{name}: value {value} out of range [{low}, {high}]")

    return errors


def validate_completeness(features: dict[str, Any]) -> list[str]:
    """Check that all required features are present.

    Args:
        features: Dictionary of feature name → value pairs.

    Returns:
        list[str]: List of missing feature names (empty if complete).
    """
    missing = [f for f in REQUIRED_FEATURES if f not in features]
    if missing:
        logger.warning("Missing features in input", extra={"missing": missing})
    return missing


def hash_input(features: dict[str, float]) -> str:
    """Generate a deterministic hash of the input features for audit logging.

    Uses SHA-256 with sorted keys to ensure consistent hashing
    regardless of dictionary insertion order.

    Args:
        features: Feature dictionary to hash.

    Returns:
        str: First 16 characters of the SHA-256 hex digest.
    """
    serialized = json.dumps(features, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def sanitize_input(features: dict[str, Any]) -> dict[str, float]:
    """Coerce input values to float and strip unexpected fields.

    Accepts only the known feature names and converts values to float.
    Raises ValueError for non-numeric or missing fields.

    Args:
        features: Raw input dictionary.

    Returns:
        dict[str, float]: Sanitized feature dictionary.

    Raises:
        ValueError: If a required feature is missing or non-numeric.
    """
    sanitized: dict[str, float] = {}

    for name in REQUIRED_FEATURES:
        if name not in features:
            raise ValueError(f"Missing required feature: {name}")

        try:
            sanitized[name] = float(features[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Feature '{name}' must be numeric, got: {features[name]!r}"
            ) from exc

    return sanitized

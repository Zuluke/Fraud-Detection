"""Shared test fixtures for FlowUp Fraud Detection tests.

Provides reusable fixtures:
- Sample transaction data (valid and invalid)
- Mock model artifacts
- FastAPI test client with loaded predictor
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import set_predictor
from src.main import create_app
from src.models.predictor import FraudPredictor


def _make_valid_transaction() -> dict[str, float]:
    """Generate a valid transaction dictionary with realistic values.

    Returns:
        dict[str, float]: Transaction with all 30 required features.
    """
    return {
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


@pytest.fixture
def valid_transaction() -> dict[str, float]:
    """Fixture: a valid transaction dictionary with all 30 features.

    Returns:
        dict[str, float]: Valid transaction data.
    """
    return _make_valid_transaction()


@pytest.fixture
def invalid_transaction_missing_field() -> dict[str, float]:
    """Fixture: transaction missing the Amount field.

    Returns:
        dict[str, float]: Incomplete transaction data.
    """
    txn = _make_valid_transaction()
    del txn["Amount"]
    return txn


@pytest.fixture
def invalid_transaction_out_of_range() -> dict[str, float]:
    """Fixture: transaction with V1 outside the allowed [-20, 20] range.

    Returns:
        dict[str, float]: Transaction with out-of-range V1 value.
    """
    txn = _make_valid_transaction()
    txn["V1"] = 999.0  # Way out of [-20, 20] range
    return txn


@pytest.fixture
def invalid_transaction_negative_amount() -> dict[str, float]:
    """Fixture: transaction with negative Amount.

    Returns:
        dict[str, float]: Transaction with invalid Amount.
    """
    txn = _make_valid_transaction()
    txn["Amount"] = -50.0
    return txn


@pytest.fixture
def mock_predictor() -> FraudPredictor:
    """Fixture: a mock FraudPredictor that returns deterministic results.

    Creates a mock that bypasses model loading and returns a fixed
    prediction for any input.

    Returns:
        FraudPredictor: Mocked predictor instance.
    """
    predictor = MagicMock(spec=FraudPredictor)
    predictor.threshold = 0.5
    predictor.model_name = "MockXGBoost"
    predictor.feature_names = (
        ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    )

    predictor.predict.return_value = {
        "probability": 0.15,
        "decision": "approved",
        "latency_ms": 1.23,
        "input_hash": "abc123def456",
    }

    return predictor


@pytest.fixture
def test_client(mock_predictor: FraudPredictor) -> TestClient:
    """Fixture: FastAPI test client with mocked predictor.

    Creates a TestClient with the predictor dependency overridden
    to use the mock, avoiding the need for a real model file.

    Args:
        mock_predictor: Mocked FraudPredictor instance.

    Returns:
        TestClient: Configured test client for API testing.
    """
    app = create_app()
    set_predictor(mock_predictor)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def sample_features_array() -> np.ndarray:
    """Fixture: numpy array of sample features for pipeline tests.

    Returns:
        np.ndarray: Array of shape (5, 30) with random features.
    """
    rng = np.random.RandomState(42)
    return rng.normal(size=(5, 30))


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Fixture: sample binary labels for pipeline tests.

    Returns:
        np.ndarray: Array of 5 binary labels.
    """
    return np.array([0, 0, 0, 1, 0])

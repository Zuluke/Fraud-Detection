"""Unit tests for FraudPredictor: model loading, prediction, and error handling."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.predictor import FraudPredictor


class TestFraudPredictor:
    """Tests for the FraudPredictor class."""

    def test_file_not_found_raises(self) -> None:
        """FraudPredictor should raise FileNotFoundError for missing models."""
        with pytest.raises(FileNotFoundError, match="Model artifact not found"):
            FraudPredictor(model_path=Path("/nonexistent/model.joblib"))

    @patch("src.models.predictor.joblib.load")
    def test_load_artifact_success(self, mock_load: MagicMock, tmp_path: Path) -> None:
        """FraudPredictor should correctly unpack a valid artifact."""
        # Create a dummy model file
        model_file = tmp_path / "test_model.joblib"
        model_file.touch()

        mock_model = MagicMock()
        mock_pipeline = MagicMock()

        mock_load.return_value = {
            "model": mock_model,
            "pipeline": mock_pipeline,
            "threshold": 0.42,
            "model_name": "TestXGBoost",
            "feature_names": ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"],
        }

        predictor = FraudPredictor(model_path=model_file)

        assert predictor.model == mock_model
        assert predictor.pipeline == mock_pipeline
        assert predictor.threshold == 0.42
        assert predictor.model_name == "TestXGBoost"
        assert len(predictor.feature_names) == 30

    @patch("src.models.predictor.joblib.load")
    def test_threshold_override(self, mock_load: MagicMock, tmp_path: Path) -> None:
        """FraudPredictor should use threshold_override when provided."""
        model_file = tmp_path / "test_model.joblib"
        model_file.touch()

        mock_load.return_value = {
            "model": MagicMock(),
            "pipeline": MagicMock(),
            "threshold": 0.42,
            "model_name": "Test",
            "feature_names": ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"],
        }

        predictor = FraudPredictor(model_path=model_file, threshold_override=0.7)
        assert predictor.threshold == 0.7

    @patch("src.models.predictor.joblib.load")
    def test_predict_returns_correct_structure(
        self, mock_load: MagicMock, tmp_path: Path
    ) -> None:
        """FraudPredictor.predict should return dict with required keys."""
        model_file = tmp_path / "test_model.joblib"
        model_file.touch()

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])

        mock_pipeline = MagicMock()
        mock_pipeline.transform.return_value = np.array([[0.0] * 31])

        mock_load.return_value = {
            "model": mock_model,
            "pipeline": mock_pipeline,
            "threshold": 0.5,
            "model_name": "XGBoost",
            "feature_names": ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"],
        }

        predictor = FraudPredictor(model_path=model_file)
        features = {
            "Time": 0.0,
            **{f"V{i}": 0.0 for i in range(1, 29)},
            "Amount": 10.0,
        }
        result = predictor.predict(features)

        assert "probability" in result
        assert "decision" in result
        assert "latency_ms" in result
        assert "input_hash" in result
        assert isinstance(result["probability"], float)
        assert isinstance(result["latency_ms"], float)

    @patch("src.models.predictor.joblib.load")
    def test_decision_approved_below_threshold(
        self, mock_load: MagicMock, tmp_path: Path
    ) -> None:
        """Probability below threshold should yield 'approved' decision."""
        model_file = tmp_path / "test_model.joblib"
        model_file.touch()

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])

        mock_load.return_value = {
            "model": mock_model,
            "pipeline": MagicMock(transform=MagicMock(return_value=np.zeros((1, 31)))),
            "threshold": 0.5,
            "model_name": "XGBoost",
            "feature_names": ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"],
        }

        predictor = FraudPredictor(model_path=model_file)
        result = predictor.predict({
            "Time": 0.0, **{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": 10.0
        })

        assert result["decision"] == "approved"

    @patch("src.models.predictor.joblib.load")
    def test_decision_fraud_above_threshold(
        self, mock_load: MagicMock, tmp_path: Path
    ) -> None:
        """Probability above threshold should yield 'suspected_fraud' decision."""
        model_file = tmp_path / "test_model.joblib"
        model_file.touch()

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        mock_load.return_value = {
            "model": mock_model,
            "pipeline": MagicMock(transform=MagicMock(return_value=np.zeros((1, 31)))),
            "threshold": 0.5,
            "model_name": "XGBoost",
            "feature_names": ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"],
        }

        predictor = FraudPredictor(model_path=model_file)
        result = predictor.predict({
            "Time": 0.0, **{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": 10.0
        })

        assert result["decision"] == "suspected_fraud"

    @patch("src.models.predictor.joblib.load")
    def test_latency_is_measured(self, mock_load: MagicMock, tmp_path: Path) -> None:
        """Prediction latency should be a positive number in milliseconds."""
        model_file = tmp_path / "test_model.joblib"
        model_file.touch()

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])

        mock_load.return_value = {
            "model": mock_model,
            "pipeline": MagicMock(transform=MagicMock(return_value=np.zeros((1, 31)))),
            "threshold": 0.5,
            "model_name": "XGBoost",
            "feature_names": ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"],
        }

        predictor = FraudPredictor(model_path=model_file)
        result = predictor.predict({
            "Time": 0.0, **{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": 10.0
        })

        assert result["latency_ms"] >= 0

    @patch("src.models.predictor.joblib.load")
    def test_input_hash_deterministic(
        self, mock_load: MagicMock, tmp_path: Path
    ) -> None:
        """Same input should always produce the same hash."""
        model_file = tmp_path / "test_model.joblib"
        model_file.touch()

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])

        mock_load.return_value = {
            "model": mock_model,
            "pipeline": MagicMock(transform=MagicMock(return_value=np.zeros((1, 31)))),
            "threshold": 0.5,
            "model_name": "XGBoost",
            "feature_names": ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"],
        }

        predictor = FraudPredictor(model_path=model_file)
        features = {"Time": 1.0, **{f"V{i}": 0.5 for i in range(1, 29)}, "Amount": 50.0}

        r1 = predictor.predict(features)
        r2 = predictor.predict(features)

        assert r1["input_hash"] == r2["input_hash"]

"""Integration tests for the end-to-end ML pipeline."""

import numpy as np
import pandas as pd
import pytest

# pyrefly: ignore [missing-import]
from src.data.feature_engineering import CyclicTimeEncoder
# pyrefly: ignore [missing-import]
from src.data.preprocessor import AmountScaler, OutlierCapper
from src.models.pipeline_builder import ALL_FEATURES, build_preprocessing_pipeline
from src.utils.metrics import compute_psi, compute_pr_auc, compute_youden_threshold


class TestPreprocessingPipeline:
    """Integration tests for the full preprocessing pipeline."""

    def _make_sample_df(self, n: int = 100) -> pd.DataFrame:
        """Create a sample DataFrame matching expected pipeline input.

        Args:
            n: Number of samples.

        Returns:
            pd.DataFrame: DataFrame with Time, V1–V28, Amount columns.
        """
        rng = np.random.RandomState(42)
        data = {
            "Time": rng.uniform(0, 172800, n),
            **{f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)},
            "Amount": rng.lognormal(3, 1.5, n),
        }
        return pd.DataFrame(data)

    def test_pipeline_builds_successfully(self) -> None:
        """build_preprocessing_pipeline should return a valid Pipeline."""
        pipeline = build_preprocessing_pipeline()
        assert pipeline is not None
        assert len(pipeline.steps) == 1  # Single ColumnTransformer step

    def test_pipeline_fit_transform(self) -> None:
        """Pipeline should transform data without errors."""
        pipeline = build_preprocessing_pipeline()
        df = self._make_sample_df(50)

        transformed = pipeline.fit_transform(df)

        assert transformed is not None
        assert transformed.shape[0] == 50
        # 2 (Time sin/cos) + 28 (V features scaled) + 1 (Amount scaled) = 31
        assert transformed.shape[1] == 31

    def test_pipeline_no_leakage(self) -> None:
        """Pipeline fitted on train should not refit on test data."""
        pipeline = build_preprocessing_pipeline()
        train_df = self._make_sample_df(80)
        test_df = self._make_sample_df(20)

        pipeline.fit(train_df)

        # Get scaler params after fit
        preprocessor = pipeline.named_steps["preprocessor"]
        pca_pipeline = preprocessor.named_transformers_["pca_features"]
        scaler = pca_pipeline.named_steps["scaler"]
        mean_after_fit = scaler.mean_.copy()

        # Transform test data
        pipeline.transform(test_df)

        # Scaler params should not change
        np.testing.assert_array_equal(scaler.mean_, mean_after_fit)

    def test_pipeline_handles_single_sample(self) -> None:
        """Pipeline should handle a single sample (prediction scenario)."""
        pipeline = build_preprocessing_pipeline()
        train_df = self._make_sample_df(100)
        pipeline.fit(train_df)

        single = self._make_sample_df(1)
        result = pipeline.transform(single)

        assert result.shape == (1, 31)

    def test_all_features_constant(self) -> None:
        """ALL_FEATURES should have exactly 30 entries."""
        assert len(ALL_FEATURES) == 30
        assert ALL_FEATURES[0] == "Time"
        assert ALL_FEATURES[-1] == "Amount"


class TestMetricsIntegration:
    """Integration tests for metrics computation."""

    def test_pr_auc_perfect_classifier(self) -> None:
        """Perfect classifier should have PR-AUC close to 1.0."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
        pr_auc = compute_pr_auc(y_true, y_proba)
        assert pr_auc > 0.9

    def test_pr_auc_random_classifier(self) -> None:
        """Random classifier should have PR-AUC around the positive class ratio."""
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 100 + [1] * 10)
        y_proba = rng.uniform(0, 1, 110)
        pr_auc = compute_pr_auc(y_true, y_proba)
        # Random should be near the fraction of positives (~0.09)
        assert pr_auc < 0.5

    def test_youden_threshold_valid_range(self) -> None:
        """Youden threshold should be between 0 and 1."""
        y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0])
        y_proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.15, 0.75, 0.25])
        threshold = compute_youden_threshold(y_true, y_proba)
        assert 0.0 <= threshold <= 1.0

    def test_psi_identical_distributions(self) -> None:
        """PSI of identical distributions should be near zero."""
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.1

    def test_psi_shifted_distribution_alerts(self) -> None:
        """PSI of significantly shifted distributions should be > 0.2."""
        rng = np.random.RandomState(42)
        reference = rng.normal(0, 1, 1000)
        shifted = rng.normal(3, 1, 1000)  # Mean shifted by 3 std
        psi = compute_psi(reference, shifted)
        assert psi > 0.2

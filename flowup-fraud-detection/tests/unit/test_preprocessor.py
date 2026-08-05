"""Unit tests for data preprocessing: OutlierCapper, AmountScaler, feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import CyclicTimeEncoder
from src.data.preprocessor import AmountScaler, OutlierCapper, preprocess_dataframe


class TestOutlierCapper:
    """Tests for the IQR-based OutlierCapper transformer."""

    def test_fit_computes_bounds(self) -> None:
        """OutlierCapper.fit should compute lower and upper IQR bounds."""
        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [100.0]])
        capper = OutlierCapper(multiplier=1.5)
        capper.fit(data)

        assert capper.lower_bounds_ is not None
        assert capper.upper_bounds_ is not None
        assert len(capper.lower_bounds_) == 1
        assert len(capper.upper_bounds_) == 1

    def test_transform_clips_outliers(self) -> None:
        """OutlierCapper.transform should clip extreme values to IQR bounds."""
        # Create data with clear quartiles
        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        capper = OutlierCapper(multiplier=1.5)
        capper.fit(data)

        # Test with outliers
        test_data = np.array([[-100.0], [3.0], [100.0]])
        transformed = capper.transform(test_data)

        # Outliers should be clipped
        assert transformed[0, 0] >= capper.lower_bounds_[0]
        assert transformed[2, 0] <= capper.upper_bounds_[0]
        # Normal value should be unchanged
        assert transformed[1, 0] == 3.0

    def test_no_data_leakage(self) -> None:
        """Bounds fitted on train data should be used on test data unchanged."""
        train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        test = np.array([[0.5], [3.0], [50.0]])

        capper = OutlierCapper(multiplier=1.5)
        capper.fit(train)

        # Bounds should come from training data, not test
        bounds_before = (capper.lower_bounds_.copy(), capper.upper_bounds_.copy())
        _ = capper.transform(test)
        bounds_after = (capper.lower_bounds_.copy(), capper.upper_bounds_.copy())

        np.testing.assert_array_equal(bounds_before[0], bounds_after[0])
        np.testing.assert_array_equal(bounds_before[1], bounds_after[1])

    def test_multicolumn_support(self) -> None:
        """OutlierCapper should handle multiple columns independently."""
        data = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ])
        capper = OutlierCapper(multiplier=1.5)
        capper.fit(data)

        assert len(capper.lower_bounds_) == 2
        assert len(capper.upper_bounds_) == 2


class TestAmountScaler:
    """Tests for the AmountScaler transformer."""

    def test_fit_transform_scales(self) -> None:
        """AmountScaler should produce zero-mean, unit-variance output."""
        data = np.array([[10.0], [20.0], [30.0], [40.0], [50.0]])
        scaler = AmountScaler()
        scaler.fit(data)
        transformed = scaler.transform(data)

        np.testing.assert_almost_equal(transformed.mean(), 0.0, decimal=5)
        np.testing.assert_almost_equal(transformed.std(ddof=0), 1.0, decimal=5)

    def test_inverse_consistency(self) -> None:
        """AmountScaler transform should be consistent across calls."""
        data = np.array([[100.0], [200.0], [300.0]])
        scaler = AmountScaler()
        scaler.fit(data)

        t1 = scaler.transform(data)
        t2 = scaler.transform(data)
        np.testing.assert_array_equal(t1, t2)


class TestCyclicTimeEncoder:
    """Tests for the CyclicTimeEncoder transformer."""

    def test_output_shape(self) -> None:
        """CyclicTimeEncoder should produce 2 columns (sin, cos)."""
        data = np.array([[0.0], [43200.0], [86400.0]])
        encoder = CyclicTimeEncoder()
        result = encoder.fit_transform(data)

        assert result.shape == (3, 2)

    def test_cyclic_property(self) -> None:
        """Time=0 and Time=86400 should produce the same encoding."""
        data = np.array([[0.0], [86400.0]])
        encoder = CyclicTimeEncoder()
        result = encoder.fit_transform(data)

        np.testing.assert_almost_equal(result[0], result[1], decimal=5)

    def test_midday_encoding(self) -> None:
        """Time=43200 (noon) should produce sin≈0, cos≈-1."""
        data = np.array([[43200.0]])  # Half day
        encoder = CyclicTimeEncoder()
        result = encoder.fit_transform(data)

        # At half period: sin(π) ≈ 0, cos(π) ≈ -1
        np.testing.assert_almost_equal(result[0, 0], 0.0, decimal=5)
        np.testing.assert_almost_equal(result[0, 1], -1.0, decimal=5)

    def test_values_bounded(self) -> None:
        """All sin/cos values should be in [-1, 1]."""
        data = np.array([[i * 3600.0] for i in range(24)])  # Every hour
        encoder = CyclicTimeEncoder()
        result = encoder.fit_transform(data)

        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_feature_names_out(self) -> None:
        """get_feature_names_out should return Time_sin and Time_cos."""
        encoder = CyclicTimeEncoder()
        names = encoder.get_feature_names_out()
        assert names == ["Time_sin", "Time_cos"]


class TestPreprocessDataframe:
    """Tests for the preprocess_dataframe utility function."""

    def test_separates_features_and_target(self) -> None:
        """preprocess_dataframe should split X and y correctly."""
        data = {
            "Time": [0.0, 1.0],
            **{f"V{i}": [0.0, 0.0] for i in range(1, 29)},
            "Amount": [10.0, 20.0],
            "Class": [0, 1],
        }
        df = pd.DataFrame(data)
        X, y = preprocess_dataframe(df)

        assert "Class" not in X.columns
        assert len(X.columns) == 30
        assert len(y) == 2
        assert y.iloc[0] == 0
        assert y.iloc[1] == 1

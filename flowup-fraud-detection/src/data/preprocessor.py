"""Data preprocessing: scaling, outlier treatment, and transformation.

Implements IQR-based outlier capping and StandardScaler as sklearn-compatible
transformers to ensure no data leakage when used inside a Pipeline.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Cap outliers using the Interquartile Range (IQR) method.

    Values below Q1 - multiplier*IQR are clipped to that lower bound.
    Values above Q3 + multiplier*IQR are clipped to that upper bound.
    Bounds are fitted on training data only (no leakage).

    Attributes:
        multiplier: IQR multiplier for determining bounds (default 1.5).
        lower_bounds_: Fitted lower bounds per feature.
        upper_bounds_: Fitted upper bounds per feature.
    """

    def __init__(self, multiplier: float = 1.5) -> None:
        """Initialize OutlierCapper.

        Args:
            multiplier: IQR multiplier for outlier detection bounds.
        """
        self.multiplier = multiplier
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "OutlierCapper":
        """Compute IQR bounds from training data.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Ignored, present for API compatibility.

        Returns:
            OutlierCapper: Fitted transformer.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        q1 = np.percentile(X_arr, 25, axis=0)
        q3 = np.percentile(X_arr, 75, axis=0)
        iqr = q3 - q1

        self.lower_bounds_ = q1 - self.multiplier * iqr
        self.upper_bounds_ = q3 + self.multiplier * iqr

        n_features = X_arr.shape[1] if X_arr.ndim > 1 else 1
        logger.debug(
            "OutlierCapper fitted",
            extra={"n_features": n_features, "multiplier": self.multiplier},
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip values to fitted IQR bounds.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Capped features with the same shape.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)


class AmountScaler(BaseEstimator, TransformerMixin):
    """StandardScaler wrapper specifically for the Amount feature.

    Wraps sklearn's StandardScaler to work on a single column within a
    ColumnTransformer pipeline.

    Attributes:
        scaler: Internal StandardScaler instance.
    """

    def __init__(self) -> None:
        """Initialize AmountScaler with an internal StandardScaler."""
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AmountScaler":
        """Fit the scaler on training data.

        Args:
            X: Amount feature of shape (n_samples, 1).
            y: Ignored, present for API compatibility.

        Returns:
            AmountScaler: Fitted transformer.
        """
        self.scaler.fit(X)
        logger.debug(
            "AmountScaler fitted",
            extra={"mean": float(self.scaler.mean_[0]), "std": float(self.scaler.scale_[0])},
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale the Amount feature.

        Args:
            X: Amount feature of shape (n_samples, 1).

        Returns:
            np.ndarray: Scaled Amount with zero mean and unit variance.
        """
        return self.scaler.transform(X)


def preprocess_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target from the raw dataset.

    Performs minimal preprocessing before pipeline application:
    - Extracts feature columns (Time, V1–V28, Amount)
    - Extracts target column (Class)

    Args:
        df: Raw dataset with all expected columns.

    Returns:
        tuple[pd.DataFrame, pd.Series]: (features, target) pair.
    """
    feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X = df[feature_cols].copy()
    y = df["Class"].copy()

    logger.info(
        "DataFrame preprocessed",
        extra={
            "n_features": len(feature_cols),
            "n_samples": len(df),
            "fraud_rate": f"{y.mean() * 100:.4f}%",
        },
    )
    return X, y

"""Feature engineering: cyclic time encoding and derived features.

Converts raw Time (seconds since first transaction) into cyclic
sine/cosine components to preserve the periodic nature of time-of-day
patterns in transaction behavior.
"""

import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

SECONDS_IN_DAY = 86_400


class CyclicTimeEncoder(BaseEstimator, TransformerMixin):
    """Encode time as cyclic sine/cosine features.

    Converts raw time in seconds to time-of-day position on the unit circle:
    - sin(2π · time_of_day / 86400)
    - cos(2π · time_of_day / 86400)

    This preserves the periodic nature: 23:59 is close to 00:01.

    Attributes:
        period: The period for cyclic encoding in seconds (default: 86,400 = 1 day).
    """

    def __init__(self, period: int = SECONDS_IN_DAY) -> None:
        """Initialize CyclicTimeEncoder.

        Args:
            period: Number of seconds in one full cycle (default: 86400 for daily).
        """
        self.period = period

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "CyclicTimeEncoder":
        """No-op fit (stateless transformer).

        Args:
            X: Time feature of shape (n_samples, 1).
            y: Ignored, present for API compatibility.

        Returns:
            CyclicTimeEncoder: The unchanged transformer instance.
        """
        logger.debug("CyclicTimeEncoder fitted (stateless)", extra={"period": self.period})
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform raw time into sine and cosine components.

        Args:
            X: Time feature of shape (n_samples, 1), values in seconds.

        Returns:
            np.ndarray: Array of shape (n_samples, 2) with [sin, cos] columns.
        """
        X_arr = np.asarray(X, dtype=np.float64).ravel()
        time_of_day = X_arr % self.period
        angle = 2 * np.pi * time_of_day / self.period

        time_sin = np.sin(angle)
        time_cos = np.cos(angle)

        return np.column_stack([time_sin, time_cos])

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return output feature names for the transformer.

        Args:
            input_features: Ignored.

        Returns:
            list[str]: ["Time_sin", "Time_cos"].
        """
        return ["Time_sin", "Time_cos"]

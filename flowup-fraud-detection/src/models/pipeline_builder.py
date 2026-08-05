"""Sklearn Pipeline construction with ColumnTransformer.

Builds a preprocessing pipeline that applies:
- Cyclic encoding to Time
- IQR outlier capping to V1–V28
- StandardScaler to Amount
- Passthrough for PCA features V1–V28 (after capping)

All transformations are encapsulated in a single Pipeline to prevent
data leakage: fit only on training data, transform on both train and test.
"""

import logging

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.feature_engineering import CyclicTimeEncoder
from src.data.preprocessor import AmountScaler, OutlierCapper

logger = logging.getLogger(__name__)

# Feature groups
TIME_FEATURES = ["Time"]
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
AMOUNT_FEATURES = ["Amount"]
ALL_FEATURES = TIME_FEATURES + PCA_FEATURES + AMOUNT_FEATURES


def build_preprocessing_pipeline() -> Pipeline:
    """Build the complete preprocessing pipeline.

    Architecture:
    1. ColumnTransformer (parallel):
       - Time → CyclicTimeEncoder → [Time_sin, Time_cos]
       - V1–V28 → OutlierCapper(IQR) → StandardScaler
       - Amount → AmountScaler (StandardScaler)
    2. Output: transformed feature matrix ready for model input.

    Returns:
        Pipeline: Sklearn Pipeline with a single ColumnTransformer step.
    """
    # Sub-pipeline for PCA features: cap outliers, then scale
    pca_pipeline = Pipeline(
        steps=[
            ("outlier_capper", OutlierCapper(multiplier=1.5)),
            ("scaler", StandardScaler()),
        ]
    )

    # ColumnTransformer applies different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("time_cyclic", CyclicTimeEncoder(), TIME_FEATURES),
            ("pca_features", pca_pipeline, PCA_FEATURES),
            ("amount_scaler", AmountScaler(), AMOUNT_FEATURES),
        ],
        remainder="drop",  # Drop any unexpected columns
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
        ]
    )

    logger.info(
        "Preprocessing pipeline built",
        extra={
            "n_time_features": len(TIME_FEATURES),
            "n_pca_features": len(PCA_FEATURES),
            "n_amount_features": len(AMOUNT_FEATURES),
            "total_input_features": len(ALL_FEATURES),
        },
    )

    return pipeline


def get_feature_names() -> list[str]:
    """Return the ordered list of input feature names expected by the pipeline.

    Returns:
        list[str]: Feature names in the order [Time, V1..V28, Amount].
    """
    return ALL_FEATURES.copy()

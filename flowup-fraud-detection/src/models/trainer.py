"""Model training: XGBoost vs RandomForest comparison with SMOTE.

Trains both classifiers on SMOTE-augmented data, selects the best
by PR-AUC, optimizes the decision threshold via Youden's J statistic,
and persists the full artifact (model + pipeline + metadata) via joblib.
"""

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.core.logging_config import get_logger
from src.data.dataset_loader import load_dataset
from src.data.preprocessor import preprocess_dataframe
from src.models.pipeline_builder import build_preprocessing_pipeline
from src.utils.metrics import compute_youden_threshold

logger = get_logger(__name__)


def _compute_scale_pos_weight(y: np.ndarray) -> float:
    """Compute scale_pos_weight for XGBoost from class distribution.

    Args:
        y: Binary target array.

    Returns:
        float: Ratio of negative to positive samples.
    """
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    return n_neg / max(n_pos, 1)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float,
) -> XGBClassifier:
    """Train an XGBoost classifier with fraud-optimized hyperparameters.

    Args:
        X_train: Preprocessed training features.
        y_train: Training labels (0/1).
        scale_pos_weight: Class weight ratio for imbalanced learning.

    Returns:
        XGBClassifier: Trained XGBoost model.
    """
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    logger.info("Training XGBoost", extra={"scale_pos_weight": scale_pos_weight})
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    logger.info("XGBoost trained", extra={"training_time_s": round(elapsed, 2)})

    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """Train a RandomForest classifier with balanced class weights.

    Args:
        X_train: Preprocessed training features.
        y_train: Training labels (0/1).

    Returns:
        RandomForestClassifier: Trained RandomForest model.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    logger.info("Training RandomForest")
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    logger.info("RandomForest trained", extra={"training_time_s": round(elapsed, 2)})

    return model


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict[str, Any]:
    """Evaluate a trained classifier and return comprehensive metrics.

    Args:
        model: Trained sklearn-compatible classifier.
        X_test: Preprocessed test features.
        y_test: True test labels.
        model_name: Name identifier for logging.

    Returns:
        dict[str, Any]: Metrics including pr_auc, f1, recall, confusion_matrix.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    pr_auc = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "model_name": model_name,
        "pr_auc": round(float(pr_auc), 4),
        "f1_score": round(float(f1), 4),
        "recall": round(float(recall), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    logger.info(
        f"{model_name} evaluation",
        extra={
            "pr_auc": metrics["pr_auc"],
            "f1_score": metrics["f1_score"],
            "recall": metrics["recall"],
        },
    )

    return metrics


def run_training(
    output_dir: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Execute the full training pipeline.

    Steps:
    1. Load dataset (Kaggle or synthetic fallback)
    2. Split train/test (stratified)
    3. Fit preprocessing pipeline on training data
    4. Apply SMOTE to preprocessed training data
    5. Train XGBoost and RandomForest
    6. Compare by PR-AUC, select best
    7. Optimize threshold via Youden's J
    8. Save model + pipeline + metadata

    Args:
        output_dir: Directory to save model artifacts. Defaults to 'models/'.
        test_size: Fraction of data reserved for testing.
        random_state: Seed for reproducibility.

    Returns:
        dict[str, Any]: Training results with metrics and artifact paths.
    """
    if output_dir is None:
        output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    logger.info("=== Starting training pipeline ===")
    df = load_dataset()
    X, y = preprocess_dataframe(df)

    # 2. Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Train/test split",
        extra={
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_fraud_rate": f"{y_train.mean() * 100:.4f}%",
        },
    )

    # 3. Fit preprocessing pipeline on training data only
    pipeline = build_preprocessing_pipeline()
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    logger.info(
        "Pipeline fitted and data transformed",
        extra={
            "train_shape": str(X_train_transformed.shape),
            "test_shape": str(X_test_transformed.shape),
        },
    )

    # 4. SMOTE on preprocessed training data (not raw, not test)
    smote = SMOTE(random_state=random_state, sampling_strategy="auto")
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_transformed, y_train
    )
    logger.info(
        "SMOTE applied",
        extra={
            "original_shape": str(X_train_transformed.shape),
            "resampled_shape": str(X_train_resampled.shape),
            "new_fraud_rate": f"{y_train_resampled.mean() * 100:.2f}%",
        },
    )

    # 5. Train both models
    scale_pos_weight = _compute_scale_pos_weight(y_train.values)

    xgb_model = train_xgboost(X_train_resampled, y_train_resampled, scale_pos_weight)
    rf_model = train_random_forest(X_train_resampled, y_train_resampled)

    # 6. Evaluate and compare
    xgb_metrics = evaluate_model(xgb_model, X_test_transformed, y_test, "XGBoost")
    rf_metrics = evaluate_model(rf_model, X_test_transformed, y_test, "RandomForest")

    # Select best by PR-AUC
    if xgb_metrics["pr_auc"] >= rf_metrics["pr_auc"]:
        best_model = xgb_model
        best_metrics = xgb_metrics
        best_name = "XGBoost"
    else:
        best_model = rf_model
        best_metrics = rf_metrics
        best_name = "RandomForest"

    logger.info(
        f"Best model selected: {best_name}",
        extra={"pr_auc": best_metrics["pr_auc"]},
    )

    # 7. Optimize threshold with Youden's J
    y_proba_test = best_model.predict_proba(X_test_transformed)[:, 1]
    optimal_threshold = compute_youden_threshold(y_test.values, y_proba_test)
    logger.info("Threshold optimized", extra={"optimal_threshold": optimal_threshold})

    # 8. Save artifacts
    artifact = {
        "model": best_model,
        "pipeline": pipeline,
        "threshold": optimal_threshold,
        "model_name": best_name,
        "metrics": {
            "xgboost": xgb_metrics,
            "random_forest": rf_metrics,
            "best": best_metrics,
        },
        "feature_names": list(X.columns),
    }

    artifact_path = output_dir / "fraud_model.joblib"
    joblib.dump(artifact, artifact_path)
    logger.info("Artifacts saved", extra={"path": str(artifact_path)})

    return {
        "best_model": best_name,
        "optimal_threshold": optimal_threshold,
        "metrics": {
            "xgboost": xgb_metrics,
            "random_forest": rf_metrics,
        },
        "artifact_path": str(artifact_path),
    }


if __name__ == "__main__":  # pragma: no cover
    from src.core.logging_config import setup_logging

    setup_logging(log_level="INFO", log_dir=Path("logs"))
    results = run_training()
    print(f"\n{'='*60}")
    print(f"Best model: {results['best_model']}")
    print(f"Optimal threshold: {results['optimal_threshold']}")
    print(f"XGBoost PR-AUC: {results['metrics']['xgboost']['pr_auc']}")
    print(f"RandomForest PR-AUC: {results['metrics']['random_forest']['pr_auc']}")
    print(f"Artifact saved to: {results['artifact_path']}")
    print(f"{'='*60}")

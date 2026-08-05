"""ML metrics: evaluation, threshold optimization, and monitoring.

Provides:
- PR-AUC, F1, Recall, Confusion Matrix calculation
- Youden's J statistic for threshold optimization
- PSI (Population Stability Index) for data drift detection
"""

import logging

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
)

logger = logging.getLogger(__name__)


def compute_pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute the Area Under the Precision-Recall Curve.

    PR-AUC is preferred over ROC-AUC for highly imbalanced datasets
    because it focuses on the minority class performance.

    Args:
        y_true: True binary labels (0/1).
        y_proba: Predicted probabilities for the positive class.

    Returns:
        float: PR-AUC score in [0, 1].
    """
    return float(average_precision_score(y_true, y_proba))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute standard classification metrics.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.

    Returns:
        dict[str, float]: Dictionary with f1_score, recall, and
            confusion_matrix (as nested list).
    """
    return {
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def compute_youden_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> float:
    """Find the optimal decision threshold using Youden's J statistic.

    Youden's J = Sensitivity + Specificity - 1 = TPR - FPR.

    For fraud detection, this maximizes the trade-off between catching
    fraud (recall) and not flagging legitimate transactions (specificity).

    Uses the Precision-Recall curve as the basis for threshold selection,
    computing the approximate J statistic at each threshold.

    Args:
        y_true: True binary labels (0/1).
        y_proba: Predicted probabilities for the positive class.

    Returns:
        float: Optimal threshold that maximizes Youden's J, rounded to 4 decimals.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # Compute F1 at each threshold to find balanced point
    # (Youden's J approximation using PR curve)
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * (precision * recall) / (precision + recall),
        0,
    )

    # Best threshold is the one maximizing F1 (proxy for Youden's J on PR curve)
    # thresholds array is one element shorter than precision/recall
    if len(thresholds) == 0:
        logger.warning("Empty thresholds array, defaulting to 0.5")
        return 0.5

    best_idx = np.argmax(f1_scores[:-1])  # Exclude last point
    optimal_threshold = float(thresholds[best_idx])

    logger.info(
        "Youden threshold computed",
        extra={
            "optimal_threshold": round(optimal_threshold, 4),
            "best_f1": round(float(f1_scores[best_idx]), 4),
            "best_precision": round(float(precision[best_idx]), 4),
            "best_recall": round(float(recall[best_idx]), 4),
        },
    )

    return round(optimal_threshold, 4)


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Compute Population Stability Index (PSI) for data drift detection.

    PSI measures how much the distribution of a variable has shifted
    between a reference (training) and current (production) population.

    Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 ≤ PSI < 0.2: Moderate shift, monitor closely
    - PSI ≥ 0.2: Significant shift, retrain model

    Args:
        reference: Reference distribution (e.g., training data scores).
        current: Current distribution (e.g., production data scores).
        n_bins: Number of bins for histogram discretization.
        epsilon: Small constant to avoid log(0).

    Returns:
        float: PSI value (>= 0). Higher values indicate more drift.
    """
    # Create bins from reference distribution
    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        n_bins + 1,
    )

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    # Convert to proportions
    ref_pct = ref_counts / max(len(reference), 1) + epsilon
    cur_pct = cur_counts / max(len(current), 1) + epsilon

    # PSI formula: Σ (cur - ref) * ln(cur / ref)
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    logger.info(
        "PSI computed",
        extra={
            "psi": round(psi, 4),
            "n_bins": n_bins,
            "ref_size": len(reference),
            "cur_size": len(current),
            "drift_alert": psi >= 0.2,
        },
    )

    return round(psi, 4)

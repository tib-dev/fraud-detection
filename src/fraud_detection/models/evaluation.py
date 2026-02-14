from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# ----------------------------
# Internal Utility
# ----------------------------
def _safe_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[int, int, int, int]:
    """
    Safely compute confusion matrix values (tn, fp, fn, tp).

    Always forces binary layout using labels=[0,1].
    Handles edge cases where only one class is present.
    """
    if len(y_true) == 0:
        return 0, 0, 0, 0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    if cm.shape != (2, 2):
        return 0, 0, 0, 0

    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


# ----------------------------
# Core classification metrics
# ----------------------------
def ensure_confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float]
) -> Dict[str, float]:
    """
    Ensure confusion matrix counts exist in metrics dictionary.
    """
    tn, fp, fn, tp = _safe_confusion_matrix(y_true, y_pred)

    metrics.update({
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    })

    return metrics


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute classification metrics including optional probability metrics.
    """
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        if len(np.unique(y_true)) > 1:
            metrics["auc_pr"] = average_precision_score(y_true, y_proba)
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        else:
            metrics["auc_pr"] = 0.0
            metrics["roc_auc"] = 0.0

    return ensure_confusion_counts(y_true, y_pred, metrics)


# ----------------------------
# Business / cost metrics
# ----------------------------
def compute_expected_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float
) -> float:
    """
    Compute expected business loss.

    Safe against single-class predictions.
    """
    tn, fp, fn, tp = _safe_confusion_matrix(y_true, y_pred)

    return float(fp * false_positive_cost + fn * false_negative_cost)


def compute_business_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float
) -> float:
    """
    Compute normalized business score between 0 and 1.
    """
    if len(y_true) == 0:
        return 0.0

    expected_loss = compute_expected_loss(
        y_true,
        y_pred,
        false_positive_cost,
        false_negative_cost
    )

    max_loss = len(y_true) * false_negative_cost

    if max_loss == 0:
        return 0.0

    return float(1 - (expected_loss / max_loss))


# ----------------------------
# Threshold optimization
# ----------------------------
def find_best_threshold_metric_based(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    steps: int = 200
) -> Tuple[float, float]:
    """
    Optimize threshold by maximizing a metric.
    """
    thresholds = np.linspace(0.01, 0.99, steps)

    best_threshold = 0.5
    best_score = -np.inf

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        if metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = t

    return float(best_threshold), float(best_score)


def find_best_threshold_cost_based(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
    steps: int = 200
) -> Tuple[float, float]:
    """
    Optimize threshold by minimizing expected business loss.
    """
    thresholds = np.linspace(0.01, 0.99, steps)

    best_threshold = 0.5
    lowest_loss = np.inf

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        loss = compute_expected_loss(
            y_true,
            y_pred,
            false_positive_cost,
            false_negative_cost
        )

        if loss < lowest_loss:
            lowest_loss = loss
            best_threshold = t

    return float(best_threshold), float(lowest_loss)

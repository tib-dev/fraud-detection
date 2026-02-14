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
# Core classification metrics
# ----------------------------
def ensure_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Ensure the confusion matrix counts exist in the metrics dictionary.

    Args:
        y_true (np.ndarray): True binary labels.
        y_pred (np.ndarray): Predicted binary labels.
        metrics (dict): Dictionary containing existing metrics.

    Returns:
        dict: Metrics dictionary updated with 'tn', 'fp', 'fn', 'tp'.
    """
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    except ValueError:
        metrics.update({"tn": 0, "fp": 0, "fn": 0, "tp": 0})
    return metrics


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute standard classification metrics and optionally probability-based metrics.

    Args:
        y_true (np.ndarray): True binary labels.
        y_pred (np.ndarray): Predicted binary labels.
        y_proba (np.ndarray, optional): Predicted probabilities for positive class.

    Returns:
        dict: Metrics including precision, recall, f1, tn, fp, fn, tp,
              and optionally auc_pr and roc_auc if y_proba is provided.
    """
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["auc_pr"] = average_precision_score(y_true, y_proba)
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

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
    Compute expected business loss given misclassification costs.

    Args:
        y_true (np.ndarray): True binary labels.
        y_pred (np.ndarray): Predicted binary labels.
        false_positive_cost (float): Cost for false positive prediction.
        false_negative_cost (float): Cost for false negative prediction.

    Returns:
        float: Total expected loss.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * false_positive_cost + fn * false_negative_cost


def compute_business_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float
) -> float:
    """
    Compute normalized business score between 0 and 1.

    Args:
        y_true (np.ndarray): True binary labels.
        y_pred (np.ndarray): Predicted binary labels.
        false_positive_cost (float): Cost for false positive prediction.
        false_negative_cost (float): Cost for false negative prediction.

    Returns:
        float: Business score where 1 = best (lowest loss), 0 = worst.
    """
    expected_loss = compute_expected_loss(y_true, y_pred, false_positive_cost, false_negative_cost)
    max_loss = len(y_true) * false_negative_cost
    return 1 - (expected_loss / max_loss)


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
    Find the optimal probability threshold by maximizing a classification metric.

    Args:
        y_true (np.ndarray): True binary labels.
        y_proba (np.ndarray): Predicted probabilities for the positive class.
        metric (str): Metric to optimize ('f1', 'precision', 'recall').
        steps (int): Number of thresholds to evaluate between 0.01 and 0.99.

    Returns:
        Tuple[float, float]: Best threshold and the corresponding metric score.
    """
    thresholds = np.linspace(0.01, 0.99, steps)
    best_threshold, best_score = 0.5, -np.inf
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = {
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
        }.get(metric, 0.0)
        if score > best_score:
            best_score, best_threshold = score, t
    return float(best_threshold), float(best_score)


def find_best_threshold_cost_based(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
    steps: int = 200
) -> Tuple[float, float]:
    """
    Find the optimal probability threshold by minimizing expected business loss.

    Args:
        y_true (np.ndarray): True binary labels.
        y_proba (np.ndarray): Predicted probabilities for the positive class.
        false_positive_cost (float): Cost for false positive prediction.
        false_negative_cost (float): Cost for false negative prediction.
        steps (int): Number of thresholds to evaluate between 0.01 and 0.99.

    Returns:
        Tuple[float, float]: Best threshold and the corresponding expected loss.
    """
    thresholds = np.linspace(0.01, 0.99, steps)
    best_threshold, lowest_loss = 0.5, np.inf
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        loss = compute_expected_loss(y_true, y_pred, false_positive_cost, false_negative_cost)
        if loss < lowest_loss:
            lowest_loss, best_threshold = loss, t
    return float(best_threshold), float(lowest_loss)

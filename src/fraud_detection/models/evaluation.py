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

# ---------------------------
# Core Metrics
# ---------------------------
def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if y_proba is not None:
        metrics["auc_pr"] = average_precision_score(y_true, y_proba)
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics

# ---------------------------
# Business Score
# ---------------------------
def compute_expected_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * false_positive_cost + fn * false_negative_cost

def compute_business_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
) -> float:
    """Normalized business score in [0,1]. Higher = better."""
    expected_loss = compute_expected_loss(y_true, y_pred, false_positive_cost, false_negative_cost)
    max_loss = len(y_true) * false_negative_cost
    return 1 - (expected_loss / max_loss)

# ---------------------------
# Threshold Optimization
# ---------------------------
def find_best_threshold_metric_based(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    steps: int = 200
) -> Tuple[float, float]:
    thresholds = np.linspace(0.01, 0.99, steps)
    best_threshold, best_score = 0.5, -np.inf
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = {
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
        }.get(metric)
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
    thresholds = np.linspace(0.01, 0.99, steps)
    best_threshold, lowest_loss = 0.5, np.inf
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        loss = compute_expected_loss(y_true, y_pred, false_positive_cost, false_negative_cost)
        if loss < lowest_loss:
            lowest_loss, best_threshold = loss, t
    return float(best_threshold), float(lowest_loss)

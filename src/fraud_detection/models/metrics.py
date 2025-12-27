"""
Evaluation metrics for fraud detection models.

This module provides functions to compute classification metrics
appropriate for imbalanced datasets (AUC-PR, F1, etc.) and
visualization utilities.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics for binary fraud detection.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_pred : np.ndarray
        Predicted binary labels (0 or 1).
    y_proba : np.ndarray, optional
        Predicted probabilities for the positive class (fraud).
        Required for AUC-based metrics.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
        - auc_pr: Average precision (AUC-PR), if y_proba provided
        - roc_auc: ROC AUC, if y_proba provided
        - tn, fp, fn, tp: Confusion matrix values
    """
    metrics = {}

    # Basic metrics from predictions
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)

    # Probability-based metrics
    if y_proba is not None:
        metrics["auc_pr"] = average_precision_score(y_true, y_proba)
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Find the optimal classification threshold for a given metric.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_proba : np.ndarray
        Predicted probabilities for the positive class.
    metric : str, optional
        Metric to optimize. Options: 'f1', 'precision', 'recall'.
        Default is 'f1'.

    Returns
    -------
    Tuple[float, float]
        - best_threshold: Optimal threshold value
        - best_score: Score achieved at optimal threshold
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Compute metric for each threshold
    if metric == "f1":
        # F1 = 2 * (precision * recall) / (precision + recall)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
            f1_scores = np.nan_to_num(f1_scores)
        # Note: precision_recall_curve returns n+1 precision/recall values
        # but only n thresholds, so we take [:-1]
        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx]
        best_score = f1_scores[best_idx]
    elif metric == "precision":
        best_idx = np.argmax(precisions[:-1])
        best_threshold = thresholds[best_idx]
        best_score = precisions[best_idx]
    elif metric == "recall":
        # For recall, we want the lowest threshold that still gives high recall
        best_idx = np.argmax(recalls[:-1])
        best_threshold = thresholds[best_idx]
        best_score = recalls[best_idx]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return float(best_threshold), float(best_score)



def get_classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Generate classification report as a DataFrame.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred : np.ndarray
        Predicted binary labels.

    Returns
    -------
    pd.DataFrame
        Classification report with precision, recall, f1-score, support.
    """
    report = classification_report(
        y_true, y_pred,
        target_names=["Non-Fraud", "Fraud"],
        output_dict=True,
    )
    return pd.DataFrame(report).T

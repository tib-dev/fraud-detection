"""
Visualization utilities for fraud detection models.

Includes functions for confusion matrix, precision-recall curve, and
classification report plotting.
"""

from sklearn.metrics import roc_curve, roc_auc_score
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, classification_report


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a confusion matrix with counts and percentages.
    """
    cm = confusion_matrix(y_true, y_pred)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    classes = ["Non-Fraud", "Fraud"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = count / cm.sum() * 100
            ax.text(
                j, i,
                f"{count:,}\n({pct:.1f}%)",
                ha="center", va="center",
                color="white" if count > thresh else "black",
                fontsize=12,
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    return ax


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot Precision-Recall curve with AUC-PR annotation.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auc_pr = average_precision_score(y_true, y_proba)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(recall, precision, label=f"{model_name} (AUC-PR = {auc_pr:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(True, alpha=0.3)

    # Add baseline (random classifier)
    baseline = y_true.mean()
    ax.axhline(y=baseline, color="gray", linestyle="--",
               label=f"Baseline ({baseline:.4f})")
    ax.legend(loc="best")

    return ax


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot ROC curve with AUC annotation.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_roc = roc_auc_score(y_true, y_proba)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_roc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    return ax



def get_classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Generate classification report as a DataFrame.
    """
    report = classification_report(
        y_true, y_pred,
        target_names=["Non-Fraud", "Fraud"],
        output_dict=True,
    )
    return pd.DataFrame(report).T

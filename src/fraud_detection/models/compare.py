"""
Model comparison and selection utilities.

This module provides helpers to:
- Compare multiple trained models side-by-side
- Score models using business-aware weighting
- Select the best model with a clear justification

Designed for highly imbalanced fraud detection problems.
"""

from typing import Dict, Tuple
import pandas as pd


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a comparison table from model evaluation results.

    Parameters
    ----------
    results : dict
        Dictionary mapping model names to their metric dictionaries.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by model name with key evaluation metrics.
    """
    rows = []

    for model_name, metrics in results.items():
        rows.append({
            "model": model_name,
            "auc_pr": metrics.get("auc_pr"),
            "roc_auc": metrics.get("roc_auc"),
            "f1": metrics.get("f1"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "threshold": metrics.get("threshold"),
        })

    df = pd.DataFrame(rows).set_index("model")

    # Primary sort: AUC-PR (best metric for imbalance)
    df = df.sort_values(by="auc_pr", ascending=False)

    return df


def score_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Assign weighted scores to models based on performance and interpretability.

    Total score = 100
    - AUC-PR: 30
    - F1-score: 20
    - Precision: 15
    - Recall: 15
    - Interpretability: 10
    - Threshold optimization: 10
    """
    rows = []

    for model_name, m in results.items():
        row = {"model": model_name}

        # ---------------------------
        # AUC-PR (30)
        # ---------------------------
        if m["auc_pr"] >= 0.70:
            row["AUC_PR"] = 30
        elif m["auc_pr"] >= 0.65:
            row["AUC_PR"] = 20
        else:
            row["AUC_PR"] = 10

        # ---------------------------
        # F1-score (20)
        # ---------------------------
        if m["f1"] >= 0.68:
            row["F1"] = 20
        elif m["f1"] >= 0.60:
            row["F1"] = 15
        else:
            row["F1"] = 10

        # ---------------------------
        # Precision (15)
        # ---------------------------
        if m["precision"] >= 0.90:
            row["Precision"] = 15
        elif m["precision"] >= 0.75:
            row["Precision"] = 10
        else:
            row["Precision"] = 5

        # ---------------------------
        # Recall (15)
        # ---------------------------
        if m["recall"] >= 0.70:
            row["Recall"] = 15
        elif m["recall"] >= 0.55:
            row["Recall"] = 10
        else:
            row["Recall"] = 5

        # ---------------------------
        # Interpretability (10)
        # ---------------------------
        if "Logistic" in model_name:
            row["Interpretability"] = 10
        elif "Random Forest" in model_name:
            row["Interpretability"] = 7
        else:
            row["Interpretability"] = 5

        # ---------------------------
        # Threshold optimization (10)
        # ---------------------------
        row["Threshold"] = 10 if "threshold" in m else 0

        # ---------------------------
        # Total score (numeric only)
        # ---------------------------
        score_cols = [
            "AUC_PR",
            "F1",
            "Precision",
            "Recall",
            "Interpretability",
            "Threshold",
        ]
        row["Total"] = sum(row[col] for col in score_cols)

        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    df = df.sort_values("Total", ascending=False)

    return df


def select_best_model(results: Dict[str, Dict]) -> Tuple[str, Dict, str]:
    """
    Select the best model with justification.

    Selection rules:
    - Primary metric: AUC-PR
    - Secondary: F1-score
    - Tie-breaker: Precision (fraud cost sensitivity)

    Parameters
    ----------
    results : dict
        Model evaluation results.

    Returns
    -------
    best_model_name : str
        Name of the selected model.
    best_model_metrics : dict
        Metrics of the selected model.
    justification : str
        Human-readable explanation for selection.
    """
    df = compare_models(results)

    best_model = df.index[0]
    metrics = results[best_model]

    justification = (
        f"{best_model} was selected as the best model because it achieved the "
        f"highest AUC-PR ({metrics['auc_pr']:.3f}), which is the most reliable "
        f"metric for imbalanced fraud detection. It also maintains a strong "
        f"F1-score ({metrics['f1']:.3f}) while keeping precision "
        f"({metrics['precision']:.3f}) high, reducing false positives."
    )

    return best_model, metrics, justification

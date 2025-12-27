"""
Training and evaluation utilities for fraud detection models with MLflow integration.

Handles model training, evaluation, saving/loading, and MLflow logging.
"""

from typing import Dict, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.base import clone
from fraud_detection.utils.mlflow_tracking import set_mlflow_tracking
from fraud_detection.models.metrics import compute_classification_metrics, find_best_threshold


def train_and_evaluate(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    optimize_threshold: bool = False,
    threshold_metric: str = "f1",
    mlflow_log: bool = True,
    model_name: str = "model"
) -> Tuple[Any, Dict[str, float], float]:
    """
    Train a pipeline and evaluate on test set, optionally logging to MLflow.
    """
    # Train pipeline
    pipeline.fit(X_train, y_train)

    # Predict probabilities
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Optimize threshold if needed
    if optimize_threshold:
        y_train_proba = cross_val_predict(
            clone(pipeline), X_train, y_train, cv=3, method="predict_proba"
        )[:, 1]
        threshold, _ = find_best_threshold(
            y_train.values, y_train_proba, metric=threshold_metric)

    # Predictions with threshold
    y_pred = (y_proba >= threshold).astype(int)

    # Compute metrics
    metrics = compute_classification_metrics(y_test.values, y_pred, y_proba)
    metrics["threshold"] = threshold

    # MLflow logging
    if mlflow_log:
        set_mlflow_tracking()
        with mlflow.start_run(run_name=model_name):
            # Log model
            mlflow.sklearn.log_model(pipeline, name="model")

            # Log params
            mlflow.log_params({
                "model_name": model_name,
                "threshold_optimized": optimize_threshold,
                "threshold_metric": threshold_metric,
            })

            # Log metrics
            mlflow.log_metrics({
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "auc_pr": metrics["auc_pr"],
                "roc_auc": metrics["roc_auc"],
                "threshold": metrics["threshold"]
            })

    return pipeline, metrics, threshold


def cross_validate_model(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Stratified cross-validation for model evaluation.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

        pipeline_clone = clone(pipeline)
        pipeline_clone.fit(X_train_fold, y_train_fold)

        y_proba = pipeline_clone.predict_proba(X_val_fold)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = compute_classification_metrics(
            y_val_fold.values, y_pred, y_proba)
        metrics["fold"] = fold_idx + 1
        fold_metrics.append(metrics)

    metrics_df = pd.DataFrame(fold_metrics)
    numeric_cols = [c for c in ["precision", "recall", "f1",
                                "auc_pr", "roc_auc"] if c in metrics_df.columns]

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": metrics_df[numeric_cols].mean().to_dict(),
        "std_metrics": metrics_df[numeric_cols].std().to_dict()
    }


def save_model(pipeline: Any, model_name: str, metrics: Dict[str, float], output_dir: Path) -> Path:
    """
    Save trained model and metrics to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_name = model_name.lower().replace(" ", "_").replace("+", "")
    model_path = output_dir / f"{clean_name}.joblib"
    metrics_path = output_dir / f"{clean_name}_metrics.joblib"

    joblib.dump(pipeline, model_path)
    joblib.dump(metrics, metrics_path)

    return model_path


def load_model(model_path: Path) -> Any:
    """
    Load a saved model.
    """
    return joblib.load(model_path)


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple models side-by-side.
    """
    rows = []
    for name, metrics in results.items():
        row = {"model": name}
        for key in ["auc_pr", "roc_auc", "f1", "precision", "recall", "threshold"]:
            if key in metrics:
                row[key] = metrics[key]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    if "auc_pr" in df.columns:
        df = df.sort_values("auc_pr", ascending=False)
    return df

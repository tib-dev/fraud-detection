"""
Train and evaluate ML models for fraud detection.

Features:
- Profile-aware: reads scoring weights and business costs from settings
- Threshold optimization: metric-based or cost-based
- Computes classification metrics, business score, and total_score
- Logs everything to MLflow, including the trained pipeline
"""

from typing import Tuple, Dict
import mlflow
import mlflow.sklearn
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict

from fraud_detection.models.evaluation import (
    compute_classification_metrics,
    compute_business_score,
    find_best_threshold_metric_based,
    find_best_threshold_cost_based,
)
from fraud_detection.models.tracker import set_mlflow_tracking
from fraud_detection.core.settings import settings


# ----------------------------
# Profile helpers
# ----------------------------
def get_profile(profile_name: str) -> dict:
    """Fetch a fraud detection profile safely from settings."""
    profile = settings.get("profiles", {}).get(profile_name)
    if not profile:
        raise ValueError(f"Profile '{profile_name}' not found in settings.")
    return profile


def compute_total_score(metrics: dict, profile_name: str) -> float:
    """
    Compute total_score = weighted ML performance + business score
    """
    profile = get_profile(profile_name)
    weights = profile.get("evaluation", {}).get("scoring_weights", {})
    perf_score = sum(metrics.get(metric, 0.0) * weight for metric, weight in weights.items())
    cost_score = metrics.get("business_score", 0.0)
    return perf_score + cost_score


# ----------------------------
# Training & evaluation
# ----------------------------
def train_and_evaluate(
    pipeline: Pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    model_name: str,
    profile_name: str,
) -> Tuple[Pipeline, Dict[str, float], float, str]:
    """
    Train, evaluate, and log a model with MLflow.

    Returns:
        pipeline: trained sklearn pipeline
        metrics: dict of metrics + business score + total_score
        threshold: float used for binary classification
        run_id: MLflow run id
    """
    profile = get_profile(profile_name)

    # MLflow experiment & model names
    experiment_name = profile.get("registry", {}).get("experiment_name", "default_experiment")
    registered_model_name = profile.get("registry", {}).get("registered_model_name", "default_model")

    # Cross-validation folds
    cv_folds = settings.get("global", {}).get("evaluation", {}).get("cross_validation_folds", 5)

    # Initialize MLflow tracking
    set_mlflow_tracking(experiment_name)

    # ----------------------------
    # Train the pipeline
    # ----------------------------
    pipeline.fit(X_train, y_train)

    # ----------------------------
    # Threshold optimization
    # ----------------------------
    threshold_config = profile.get("thresholds", {})
    strategy = threshold_config.get("strategy", "metric_based")
    threshold = 0.5

    if threshold_config.get("enabled", True):
        y_train_proba = cross_val_predict(
            clone(pipeline),
            X_train,
            y_train,
            cv=cv_folds,
            method="predict_proba"
        )[:, 1]

        if strategy == "metric_based":
            metric = threshold_config.get("metric", "f1")
            threshold, _ = find_best_threshold_metric_based(y_train.values, y_train_proba, metric)
        elif strategy == "cost_based":
            costs = profile.get("business", {}).get("costs", {})
            threshold, _ = find_best_threshold_cost_based(
                y_train.values,
                y_train_proba,
                costs.get("false_positive", 0),
                costs.get("false_negative", 0),
            )

    # ----------------------------
    # Evaluation on test set
    # ----------------------------
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = compute_classification_metrics(y_test.values, y_pred, y_proba)

    # Business score
    costs = profile.get("business", {}).get("costs", {})
    metrics["business_score"] = compute_business_score(
        y_test.values,
        y_pred,
        false_positive_cost=costs.get("false_positive", 0),
        false_negative_cost=costs.get("false_negative", 0),
    )
    metrics["threshold"] = threshold

    # Total score (weighted ML metrics + business)
    metrics["total_score"] = compute_total_score(metrics, profile_name)

    # ----------------------------
    # MLflow logging
    # ----------------------------
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params({
            "model_name": model_name,
            "profile": profile_name,
            "threshold_strategy": strategy,
            "threshold": threshold,
        })

        metric_keys = ["precision", "recall", "f1", "auc_pr", "roc_auc", "business_score", "total_score"]
        mlflow.log_metrics({k: metrics[k] for k in metric_keys if k in metrics})

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=registered_model_name
        )

        run_id = run.info.run_id

    return pipeline, metrics, threshold, run_id

from typing import Tuple, Dict
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
import mlflow
import mlflow.sklearn

from fraud_detection.core.settings import settings
from fraud_detection.models.tracker import set_mlflow_tracking
from fraud_detection.models.evaluation import (
    compute_classification_metrics,
    compute_business_score,
    find_best_threshold_metric_based,
    find_best_threshold_cost_based,
)

# ----------------------------
# Helpers
# ----------------------------
def get_profile(profile_name: str) -> dict:
    """
    Retrieve a fraud detection profile from settings.

    Args:
        profile_name (str): Name of the profile to load.

    Returns:
        dict: Profile configuration.

    Raises:
        ValueError: If profile is not found.
    """
    profile = settings.get("profiles", {}).get(profile_name)
    if not profile:
        raise ValueError(f"Profile '{profile_name}' not found in settings.")
    return profile


def clean_feature_names(X: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names to remove special characters for tree-based models.

    Args:
        X (pd.DataFrame): Input feature dataframe.

    Returns:
        pd.DataFrame: Dataframe with cleaned column names.
    """
    X = X.copy()
    X.columns = [str(c).replace(" ", "_").replace("-", "_").replace(".", "_") for c in X.columns]
    return X


def compute_total_score(metrics: dict, profile_name: str) -> float:
    """
    Compute total_score as sum of weighted ML metrics and business score.

    Args:
        metrics (dict): Dictionary containing ML metrics and business score.
        profile_name (str): Name of the active profile.

    Returns:
        float: Combined total score.
    """
    profile = get_profile(profile_name)
    weights = profile.get("evaluation", {}).get("scoring_weights", {})
    perf_score = sum(metrics.get(metric, 0.0) * weight for metric, weight in weights.items())
    cost_score = metrics.get("business_score", 0.0)
    return perf_score + cost_score


# ----------------------------
# Train & evaluate
# ----------------------------
def train_and_evaluate(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    model_name: str,
    profile_name: str,
) -> Tuple[Pipeline, Dict[str, float], float, str]:
    """
    Train a machine learning pipeline, optimize threshold, evaluate metrics,
    and log results to MLflow.

    Args:
        pipeline (Pipeline): scikit-learn pipeline to train.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        model_name (str): Name of the model/run for MLflow.
        profile_name (str): Name of the fraud detection profile.

    Returns:
        Tuple[Pipeline, dict, float, str]:
            - Trained pipeline.
            - Metrics dictionary including ML metrics, business score, and total_score.
            - Threshold used for binary classification.
            - MLflow run ID.
    """
    profile = get_profile(profile_name)

    # Clean feature names
    X_train = clean_feature_names(X_train)
    X_test = clean_feature_names(X_test)

    # MLflow setup
    experiment_name = profile.get("registry", {}).get("experiment_name", "default_experiment")
    registered_model_name = profile.get("registry", {}).get("registered_model_name", "default_model")
    cv_folds = settings.get("global", {}).get("evaluation", {}).get("cross_validation_folds", 5)
    set_mlflow_tracking(experiment_name)

    # Train pipeline
    pipeline.fit(X_train, y_train)

    # Threshold optimization
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

    # Evaluate
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
    metrics["total_score"] = compute_total_score(metrics, profile_name)

    # Log to MLflow
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

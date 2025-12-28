import mlflow
import mlflow.sklearn
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

from fraud_detection.models.metrics import (
    compute_classification_metrics,
    find_best_threshold,
)
from fraud_detection.models.tracker import set_mlflow_tracking
from fraud_detection.core.settings import settings


def train_and_evaluate(
    pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    model_name: str,
    profile_name: str = None,
    optimize_threshold: bool = True,
    threshold_metric: str = None,
):
    """
    Train a classification pipeline, optionally optimize threshold, evaluate on test set,
    compute business score based on profile weights, and log everything to MLflow.
    Automatically uses the profile's registered model name and experiment.
    """

    # ------------------------------------------------------
    # Get profile (if provided)
    # ------------------------------------------------------
    profile = None
    if profile_name:
        profile = settings.get("profiles", {}).get(profile_name)
        if profile is None:
            raise ValueError(
                f"Profile '{profile_name}' not found in settings.")

    # Get experiment name and registered model from profile
    experiment_name = profile["registry"]["experiment_name"] if profile else "default_experiment"
    registered_model_name = profile["registry"]["registered_model_name"] if profile else "default_model"

    # ------------------------------------------------------
    # MLflow setup
    # ------------------------------------------------------
    set_mlflow_tracking(experiment_name=experiment_name)

    # ------------------------------------------------------
    # Train model
    # ------------------------------------------------------
    pipeline.fit(X_train, y_train)

    # ------------------------------------------------------
    # Threshold optimization
    # ------------------------------------------------------
    threshold = 0.5
    if optimize_threshold or (profile and profile.get("thresholds", {}).get("optimize", False)):
        y_train_proba = cross_val_predict(
            clone(pipeline),
            X_train,
            y_train,
            cv=3,
            method="predict_proba",
        )[:, 1]

        threshold_metric = threshold_metric or profile.get(
            "thresholds", {}).get("metric", "f1")
        threshold, _ = find_best_threshold(
            y_train.values, y_train_proba, metric=threshold_metric)

    # ------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = compute_classification_metrics(y_test.values, y_pred, y_proba)
    metrics["threshold"] = threshold
    metrics["y_true"] = y_test.values
    metrics["y_pred"] = y_pred

    # ------------------------------------------------------
    # Compute business / profile score
    # ------------------------------------------------------
    if profile:
        weights = profile.get("scoring", {}).get("weights", {})
        business_score = sum(metrics[k] * w for k,
                             w in weights.items() if k in metrics)
    else:
        business_score = 0.4 * metrics["recall"] + 0.3 * \
            metrics["precision"] + 0.3 * metrics["auc_pr"]

    metrics["total_score"] = business_score

    # ------------------------------------------------------
    # MLflow logging
    # ------------------------------------------------------
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params({
            "model_name": model_name,
            "threshold_metric": threshold_metric,
            "threshold": threshold,
        })

        mlflow.log_metrics({
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auc_pr": metrics["auc_pr"],
            "roc_auc": metrics["roc_auc"],
            "total_score": business_score,
        })

        # Log model to profile-specific registry path
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name=registered_model_name,
        )

        run_id = run.info.run_id

    return pipeline, metrics, threshold, run_id

"""
Model evaluation, selection, and MLflow promotion for fraud detection.

- Profile-aware: reads scoring weights and business costs from settings
- Computes ML metrics, business score, total_score
- Supports best-model selection and MLflow registry promotion
"""

from typing import Dict, Tuple
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import confusion_matrix
import pandas as pd

from fraud_detection.core.settings import settings
from fraud_detection.models.evaluation import compute_business_score

# -------------------------------------------------------------------
# Profile helper
# -------------------------------------------------------------------
def get_profile(profile_name: str) -> dict:
    profile = settings.get("profiles", {}).get(profile_name)
    if not profile:
        raise ValueError(f"Profile '{profile_name}' not found in settings.")
    return profile

# -------------------------------------------------------------------
# Confusion matrix / metrics helpers
# -------------------------------------------------------------------
def ensure_confusion_counts(metrics: dict) -> dict:
    """Ensure metrics have TN, FP, FN, TP."""
    required_keys = {"true_negatives", "false_positives", "false_negatives", "true_positives"}
    if required_keys.issubset(metrics):
        return metrics

    if "y_true" in metrics and "y_pred" in metrics:
        tn, fp, fn, tp = confusion_matrix(metrics["y_true"], metrics["y_pred"]).ravel()
        metrics.update({
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        })
    else:
        # fallback defaults
        metrics.setdefault("false_positives", 0)
        metrics.setdefault("false_negatives", 0)
        metrics.setdefault("true_positives", 0)
        metrics.setdefault("true_negatives", 0)

    return metrics

def compute_cost_score(metrics: dict, costs: dict) -> float:
    """Compute normalized business score (higher is better)."""
    metrics = ensure_confusion_counts(metrics)
    return compute_business_score(
        y_true=[0]*(metrics["true_negatives"] + metrics["false_positives"]) + [1]*(metrics["true_positives"] + metrics["false_negatives"]),
        y_pred=[0]*metrics["true_negatives"] + [1]*metrics["false_positives"] + [0]*metrics["false_negatives"] + [1]*metrics["true_positives"],
        false_positive_cost=costs.get("false_positive", 0),
        false_negative_cost=costs.get("false_negative", 0),
    )

def compute_weighted_score(metrics: dict, weights: dict) -> float:
    """Compute weighted ML performance score. Missing metrics count as zero."""
    return sum(metrics.get(metric, 0.0) * weight for metric, weight in weights.items())

# -------------------------------------------------------------------
# Model scoring & comparison
# -------------------------------------------------------------------
def score_models(results: dict, profile_name: str) -> pd.DataFrame:
    """
    Compute performance_score, cost_score, and total_score for all models.
    """
    profile = settings.get("profiles", {}).get(profile_name, {})
    weights = profile.get("evaluation", {}).get("scoring_weights", {})
    costs = profile.get("business", {}).get("costs", {})

    rows = []
    for model_name, metrics in results.items():
        perf_score = compute_weighted_score(metrics, weights)
        cost_score = compute_cost_score(metrics, costs)
        total_score = perf_score + cost_score

        rows.append({
            "model": model_name,
            "performance_score": perf_score,
            "cost_score": cost_score,
            "total_score": total_score,
        })

    return pd.DataFrame(rows).set_index("model").sort_values("total_score", ascending=False)

def compare_models(results: dict) -> pd.DataFrame:
    """Side-by-side comparison of raw metrics."""
    return pd.DataFrame.from_dict(results, orient="index").sort_values("auc_pr", ascending=False)

def select_best_model(results: dict, profile_name: str) -> Tuple[str, dict, str]:
    """Select the best model based on total_score."""
    scored = score_models(results, profile_name)
    best_model = scored.index[0]

    reason = (
        f"Selected '{best_model}' because it achieved the highest total_score "
        f"({scored.loc[best_model, 'total_score']:.4f}) under the "
        f"'{profile_name}' business profile."
    )

    return best_model, scored.loc[best_model].to_dict(), reason

# -------------------------------------------------------------------
# MLflow registry promotion
# -------------------------------------------------------------------
def promote_best_model(*, profile_name: str, run_id: str) -> str:
    """
    Promote a model to Production if it beats the current Production model
    based on profile rules.
    """
    profile = settings.get("profiles", {}).get(profile_name, {})
    registry_name = profile["registry"]["registered_model_name"]
    min_delta = profile["promotion"]["min_improvement"]
    target_stage = profile["promotion"]["stage_on_promote"]

    client = MlflowClient()
    run = mlflow.get_run(run_id)
    run_metrics = run.data.metrics

    if "total_score" not in run_metrics:
        raise KeyError(
            "Run is missing 'total_score'. Ensure train_and_evaluate logs it."
        )

    new_score = run_metrics["total_score"]

    # ensure registry exists
    try:
        client.get_registered_model(registry_name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(registry_name)

    versions = client.search_model_versions(f"name='{registry_name}'")
    prod_versions = [v for v in versions if v.current_stage == "Production"]

    if not prod_versions:
        # First promotion
        latest = max(versions, key=lambda v: int(v.version))
        client.transition_model_version_stage(
            name=registry_name,
            version=latest.version,
            stage=target_stage,
            archive_existing_versions=True,
        )
        return "No Production model found. Promoted first model."

    # Compare against existing Production
    prod = prod_versions[0]
    prod_run = mlflow.get_run(prod.run_id)
    prod_score = prod_run.data.metrics.get("total_score", float("-inf"))

    if new_score > prod_score + min_delta:
        client.transition_model_version_stage(
            name=registry_name,
            version=prod.version,
            stage=target_stage,
            archive_existing_versions=True,
        )
        return "New model outperformed Production. Promotion completed."

    return "New model did not outperform Production. No promotion."

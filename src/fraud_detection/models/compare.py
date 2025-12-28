import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import confusion_matrix

from fraud_detection.core.settings import settings


# -------------------------------------------------------------------
# Profile helper
# -------------------------------------------------------------------

def get_profile(profile_name: str) -> dict:
    profile = settings.get("profiles", {}).get(profile_name)
    if not profile:
        raise ValueError(f"Profile '{profile_name}' not found in settings.")
    return profile


# -------------------------------------------------------------------
# Metric helpers
# -------------------------------------------------------------------

def ensure_confusion_counts(metrics: dict) -> dict:
    """
    Ensure confusion-matrix-derived counts exist in metrics.
    """
    required = {"false_positives", "false_negatives"}
    if required.issubset(metrics):
        return metrics

    if "y_true" in metrics and "y_pred" in metrics:
        tn, fp, fn, tp = confusion_matrix(
            metrics["y_true"], metrics["y_pred"]
        ).ravel()

        metrics.update({
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        })
    else:
        # Safe fallback
        metrics.setdefault("false_positives", 0)
        metrics.setdefault("false_negatives", 0)

    return metrics


def compute_cost_score(metrics: dict, costs: dict) -> float:
    """
    Compute business cost impact.
    Lower is worse, so return negative.
    """
    metrics = ensure_confusion_counts(metrics)

    fp_cost = metrics["false_positives"] * costs.get("false_positive", 0)
    fn_cost = metrics["false_negatives"] * costs.get("false_negative", 0)

    return -(fp_cost + fn_cost)


def compute_weighted_score(metrics: dict, weights: dict) -> float:
    """
    Weighted ML performance score.
    Missing metrics contribute zero.
    """
    return sum(metrics.get(metric, 0.0) * weight for metric, weight in weights.items())


# -------------------------------------------------------------------
# Model comparison & scoring
# -------------------------------------------------------------------

def compare_models(results: dict) -> pd.DataFrame:
    """
    Side-by-side comparison of raw metrics.
    """
    df = (
        pd.DataFrame.from_dict(results, orient="index")
        .sort_values("auc_pr", ascending=False)
    )
    df.index.name = "model"
    return df


def score_models(results: dict, profile_name: str) -> pd.DataFrame:
    """
    Apply profile-based scoring (ML + business cost).
    """
    profile = get_profile(profile_name)

    weights = profile.get("scoring", {}).get("weights", {})
    costs = profile.get("business_costs", {})

    rows = []
    for model_name, metrics in results.items():
        perf_score = compute_weighted_score(metrics, weights)
        cost_score = compute_cost_score(metrics, costs)

        rows.append({
            "model": model_name,
            "performance_score": perf_score,
            "cost_score": cost_score,
            "total_score": perf_score + cost_score,
        })

    return (
        pd.DataFrame(rows)
        .set_index("model")
        .sort_values("total_score", ascending=False)
    )


def select_best_model(results: dict, profile_name: str):
    """
    Select best model with explanation.
    """
    scored = score_models(results, profile_name)
    best_model = scored.index[0]

    reason = (
        f"Selected '{best_model}' because it achieved the highest total_score "
        f"({scored.loc[best_model, 'total_score']:.4f}) under the "
        f"'{profile_name}' business profile."
    )

    return best_model, scored.loc[best_model].to_dict(), reason


# -------------------------------------------------------------------
# MLflow Registry Promotion
# -------------------------------------------------------------------

def promote_best_model(*, profile_name: str, run_id: str):
    """
    Promote a model to Production if it beats the current Production model
    based on profile rules.
    """
    profile = get_profile(profile_name)

    registry_name = profile["registry"]["registered_model_name"]
    min_delta = profile["promotion"]["min_improvement"]
    target_stage = profile["promotion"]["stage_on_promote"]

    client = MlflowClient()

    # --------------------------------------------------
    # Validate run
    # --------------------------------------------------
    run = mlflow.get_run(run_id)
    run_metrics = run.data.metrics

    if "total_score" not in run_metrics:
        raise KeyError(
            "Run is missing 'total_score'. Ensure train_and_evaluate logs it."
        )

    new_score = run_metrics["total_score"]

    # --------------------------------------------------
    # Ensure registry exists
    # --------------------------------------------------
    try:
        client.get_registered_model(registry_name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(registry_name)

    # --------------------------------------------------
    # Fetch all versions
    # --------------------------------------------------
    versions = client.search_model_versions(f"name='{registry_name}'")

    if not versions:
        raise ValueError(
            f"No model versions found for '{registry_name}'. "
            "Ensure the model was logged with registered_model_name."
        )

    prod_versions = [v for v in versions if v.current_stage == "Production"]

    # --------------------------------------------------
    # First promotion
    # --------------------------------------------------
    if not prod_versions:
        latest = max(versions, key=lambda v: int(v.version))
        client.transition_model_version_stage(
            name=registry_name,
            version=latest.version,
            stage=target_stage,
            archive_existing_versions=True,
        )
        return "No Production model found. Promoted first model."

    # --------------------------------------------------
    # Compare against Production
    # --------------------------------------------------
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

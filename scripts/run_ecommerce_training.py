import matplotlib.pyplot as plt

import fraud_detection.viz.model_plots as viz
from fraud_detection.data.loader import DataHandler
from fraud_detection.models.pipeline import build_pipeline
from fraud_detection.models.train import train_and_evaluate
from fraud_detection.core.settings import settings
import fraud_detection.models.compare as cp


# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
def load_data():
    test_original = DataHandler.from_registry(
        "DATA", "processed_dir", "test_original.parquet"
    ).load()

    train_original = DataHandler.from_registry(
        "DATA", "processed_dir", "train_original.parquet"
    ).load()

    train_resampled = DataHandler.from_registry(
        "DATA", "processed_dir", "train_resampled.parquet"
    ).load()

    return train_original, train_resampled, test_original


# ------------------------------------------------------------------
# Main training pipeline
# ------------------------------------------------------------------
def run_pipeline():
    FEATURES = settings.get("features")
    TARGET = FEATURES["target"]

    profile_name = "ecommerce"
    experiment_name = "fraud_detection_models"

    # Containers
    results = {}
    run_ids = {}
    thresholds = {}
    pipelines = {}

    # Load data
    train_original, train_resampled, test_original = load_data()

    X_train_orig = train_original.drop(columns=[TARGET])
    y_train_orig = train_original[TARGET]

    X_train_res = train_resampled.drop(columns=[TARGET])
    y_train_res = train_resampled[TARGET]

    X_test = test_original.drop(columns=[TARGET])
    y_test = test_original[TARGET]

    # --------------------------------------------------------------
    # Baseline models (original data)
    # --------------------------------------------------------------
    baseline_models = {
        "Logistic Regression": {
            "pipeline": build_pipeline("logistic_regression"),
            "threshold_metric": "f1",
        },
    }

    for model_name, cfg in baseline_models.items():
        pipe, metrics, threshold, run_id = train_and_evaluate(
            pipeline=cfg["pipeline"],
            X_train=X_train_orig,
            y_train=y_train_orig,
            X_test=X_test,
            y_test=y_test,
            optimize_threshold=True,
            threshold_metric=cfg["threshold_metric"],
            model_name=model_name,
            profile_name=profile_name,
            experiment_name=experiment_name,
        )

        results[model_name] = metrics
        run_ids[model_name] = run_id
        thresholds[model_name] = threshold
        pipelines[model_name] = pipe

    # --------------------------------------------------------------
    # Ensemble models (resampled data)
    # --------------------------------------------------------------
    ensemble_models = {
        "Random Forest": {
            "pipeline": build_pipeline("random_forest"),
            "threshold_metric": "f1",
        },
        "XGBoost": {
            "pipeline": build_pipeline("xgboost"),
            "threshold_metric": "f1",
        },
    }

    for model_name, cfg in ensemble_models.items():
        pipe, metrics, threshold, run_id = train_and_evaluate(
            pipeline=cfg["pipeline"],
            X_train=X_train_res,
            y_train=y_train_res,
            X_test=X_test,
            y_test=y_test,
            optimize_threshold=True,
            threshold_metric=cfg["threshold_metric"],
            model_name=model_name,
            profile_name=profile_name,
            experiment_name=experiment_name,
        )

        results[model_name] = metrics
        run_ids[model_name] = run_id
        thresholds[model_name] = threshold
        pipelines[model_name] = pipe

    # --------------------------------------------------------------
    # Model comparison & selection
    # --------------------------------------------------------------
    scored_df = cp.score_models(
        results=results,
        profile_name=profile_name,
    )
    print("\n=== Model Scores ===")
    print(scored_df)

    best_model, best_metrics, reason = cp.select_best_model(
        results=results,
        profile_name=profile_name,
    )
    print("\n=== Model Selection ===")
    print(reason)

    # --------------------------------------------------------------
    # Promotion
    # --------------------------------------------------------------
    best_run_id = run_ids[best_model]
    promotion_result = cp.promote_best_model(
        profile_name=profile_name,
        run_id=best_run_id,
    )

    print(f"\n{best_model} promotion result:", promotion_result)

    return {
        "best_model": best_model,
        "scores": scored_df,
        "thresholds": thresholds,
        "pipelines": pipelines,
    }


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()

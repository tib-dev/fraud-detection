"""
run_training.py
Main entry point for training, evaluating, and promoting fraud detection models
using pipeline.py, training.py, tuning.py modules.
"""

from fraud_detection.models.pipeline import build_pipeline
from fraud_detection.models.train import train_and_evaluate
from fraud_detection.models.compare import score_models, compare_models,promote_best_model
from fraud_detection.core.settings import settings

# ---------------------------
# Data loading
# ---------------------------
from fraud_detection.data.loader import DataHandler

test_original = DataHandler.from_registry("DATA", "processed_dir", "test_original.parquet").load()
train_original = DataHandler.from_registry("DATA", "processed_dir", "train_original.parquet").load()
train_resampled = DataHandler.from_registry("DATA", "processed_dir", "train_resampled.parquet").load()

FEATURES = settings.get("features")
TARGET = FEATURES["target"]

X_train_orig = train_original.drop(columns=[TARGET])
y_train_orig = train_original[TARGET]

X_train_res = train_resampled.drop(columns=[TARGET])
y_train_res = train_resampled[TARGET]

X_test = test_original.drop(columns=[TARGET])
y_test = test_original[TARGET]

# ---------------------------
# Config
# ---------------------------
PROFILE_NAME = "credit_card"
EXPERIMENT_NAME = settings["profiles"][PROFILE_NAME]["registry"]["experiment_name"]

results = {}
run_ids = {}
thresholds = {}
pipelines = {}

# ---------------------------
# Baseline models (original data)
# ---------------------------
baseline_models = {
    "Logistic Regression": {"pipeline": build_pipeline("logistic_regression"), "threshold_metric": "f1"},
}

for model_name, cfg in baseline_models.items():
    print(f"\n=== Training {model_name} (baseline) ===")
    pipe, metrics, threshold, run_id = train_and_evaluate(
        pipeline=cfg["pipeline"],
        X_train=X_train_orig,
        y_train=y_train_orig,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name,
        profile_name=PROFILE_NAME,
    )
    results[model_name] = metrics
    run_ids[model_name] = run_id
    thresholds[model_name] = threshold
    pipelines[model_name] = pipe

# ---------------------------
# Ensemble / more complex models (resampled data)
# ---------------------------
ensemble_models = {
    "Random Forest": {"pipeline": build_pipeline("random_forest"), "threshold_metric": "f1"},
    "XGBoost": {"pipeline": build_pipeline("xgboost"), "threshold_metric": "f1"},
    "LightGBM": {"pipeline": build_pipeline("lightgbm"), "threshold_metric": "f1"},
}

for model_name, cfg in ensemble_models.items():
    print(f"\n=== Training {model_name} (resampled) ===")
    pipe, metrics, threshold, run_id = train_and_evaluate(
        pipeline=cfg["pipeline"],
        X_train=X_train_res,
        y_train=y_train_res,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name,
        profile_name=PROFILE_NAME,
    )
    results[model_name] = metrics
    run_ids[model_name] = run_id
    thresholds[model_name] = threshold
    pipelines[model_name] = pipe

# ---------------------------
# Optional: hyperparameter tuning
# ---------------------------
# Example usage:
# from sklearn.model_selection import ParameterGrid
# tuning_config = {
#     "Random Forest": {
#         "pipeline": pipelines["Random Forest"],
#         "param_grid": {"model__n_estimators": [200, 400], "model__max_depth": [8, 12]},
#     },
#     "XGBoost": {
#         "pipeline": pipelines["XGBoost"],
#         "param_grid": {"model__learning_rate": [0.05, 0.1], "model__max_depth": [4, 6]},
#     },
# }
# tuned_estimators, tuning_results = tune_multiple_models(tuning_config, X_train_res, y_train_res)

# ---------------------------
# Compare & score models
# ---------------------------
print("\n=== Raw metrics comparison ===")
print(compare_models(results))

print("\n=== Profile-based scoring ===")
scored_df = score_models(results, PROFILE_NAME)
print(scored_df)

# ---------------------------
# Promote best model
# ---------------------------
best_model_name = scored_df.index[0]
best_metrics = scored_df.iloc[0].to_dict()
run_id_to_promote = run_ids[best_model_name]
promotion_msg = promote_best_model(profile_name=PROFILE_NAME, run_id=run_id_to_promote)

print(f"\nPromotion result: {promotion_msg}")
print(f"Best model selected: {best_model_name}")
print(f"Metrics: {best_metrics}")
print(f"Thresholds: {thresholds}")

"""
run_tuning.py
Demonstrates hyperparameter tuning using tune_model and tune_multiple_models
with pipelines and training modules.
"""

from fraud_detection.models.pipeline import build_pipeline
from fraud_detection.models.tuning import tune_model, tune_multiple_models
from fraud_detection.models.train import train_and_evaluate
from fraud_detection.core.settings import settings
from fraud_detection.data.loader import DataHandler

# ---------------------------
# Load data
# ---------------------------
test_original = DataHandler.from_registry("DATA", "processed_dir", "test_original.parquet").load()
train_resampled = DataHandler.from_registry("DATA", "processed_dir", "train_resampled.parquet").load()

FEATURES = settings.get("features")
TARGET = FEATURES["target"]

X_train_res = train_resampled.drop(columns=[TARGET])
y_train_res = train_resampled[TARGET]

X_test = test_original.drop(columns=[TARGET])
y_test = test_original[TARGET]

PROFILE_NAME = "credit_card"

# ---------------------------
# Define pipelines
# ---------------------------
rf_pipeline = build_pipeline("random_forest")
xgb_pipeline = build_pipeline("xgboost")

# ---------------------------
# Single model tuning example
# ---------------------------
rf_param_grid = {
    "model__n_estimators": [200, 400, 600],
    "model__max_depth": [8, 12, 16],
    "model__min_samples_leaf": [10, 30, 50]
}

best_rf_pipeline, rf_cv_results, rf_best_score, rf_best_params = tune_model(
    pipeline=rf_pipeline,
    param_grid=rf_param_grid,
    X_train=X_train_res,
    y_train=y_train_res,
    scoring="f1",
    cv=3,
    search_type="grid"
)

print("Random Forest tuning complete")
print("Best params:", rf_best_params)
print("Best CV score:", rf_best_score)

# ---------------------------
# Multi-model tuning example
# ---------------------------
models_config = {
    "Random Forest": {
        "pipeline": build_pipeline("random_forest"),
        "param_grid": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [8, 12],
            "model__min_samples_leaf": [20, 30]
        }
    },
    "XGBoost": {
        "pipeline": build_pipeline("xgboost"),
        "param_grid": {
            "model__n_estimators": [300, 500],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [4, 6]
        }
    }
}

best_estimators, all_cv_results = tune_multiple_models(
    models_config=models_config,
    X_train=X_train_res,
    y_train=y_train_res,
    scoring="f1",
    cv=3,
    search_type="grid"
)

print("\nMulti-model tuning complete")
for name, est in best_estimators.items():
    print(f"{name} best estimator: {est}")
    print(f"Best CV score: {all_cv_results[name]['best_score']}")
    print(f"Best params: {all_cv_results[name]['best_params']}")

# ---------------------------
# Optionally: train & evaluate tuned models
# ---------------------------
results = {}
for model_name, pipeline in best_estimators.items():
    print(f"\n=== Train & evaluate tuned {model_name} ===")
    pipe, metrics, threshold, run_id = train_and_evaluate(
        pipeline=pipeline,
        X_train=X_train_res,
        y_train=y_train_res,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name,
        profile_name=PROFILE_NAME
    )
    results[model_name] = metrics
    print(f"{model_name} metrics: {metrics}")

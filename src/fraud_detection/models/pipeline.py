"""
Model pipeline factory for fraud detection.
Feature engineering is assumed to be completed upstream.
"""

from typing import Any

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


SUPPORTED_MODELS = {
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
}


def build_pipeline(model_name: str) -> Pipeline:
    """
    Build a single model pipeline.

    Parameters
    ----------
    model_name : str
        One of:
        - 'logistic_regression'
        - 'random_forest'
        - 'xgboost'
        - 'lightgbm'

    Returns
    -------
    Pipeline
        Sklearn-compatible pipeline with a single model.
    """

    model_name = model_name.lower()

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Choose from {sorted(SUPPORTED_MODELS)}"
        )

    if model_name == "logistic_regression":
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )

    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=30,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    elif model_name == "xgboost":
        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )

    elif model_name == "lightgbm":
        model = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    return Pipeline(
        steps=[
            ("model", model)
        ]
    )

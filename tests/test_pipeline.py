import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from fraud_detection.models.pipeline import (
    build_pipeline,
    SUPPORTED_MODELS,
)


# --------------------------------------------------
# Basic Construction Tests
# --------------------------------------------------

@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_build_pipeline_returns_pipeline(model_name):
    pipeline = build_pipeline(model_name)

    assert isinstance(pipeline, Pipeline)
    assert "model" in pipeline.named_steps


@pytest.mark.parametrize(
    "model_name, expected_type",
    [
        ("logistic_regression", LogisticRegression),
        ("random_forest", RandomForestClassifier),
        ("xgboost", XGBClassifier),
        ("lightgbm", LGBMClassifier),
    ],
)
def test_build_pipeline_correct_model_type(model_name, expected_type):
    pipeline = build_pipeline(model_name)
    model = pipeline.named_steps["model"]

    assert isinstance(model, expected_type)


# --------------------------------------------------
# Case Insensitivity Test
# --------------------------------------------------

def test_build_pipeline_case_insensitive():
    pipeline = build_pipeline("LoGiStIc_ReGrEsSiOn")
    model = pipeline.named_steps["model"]

    assert isinstance(model, LogisticRegression)


# --------------------------------------------------
# Unsupported Model Test
# --------------------------------------------------

def test_build_pipeline_unsupported_model():
    with pytest.raises(ValueError) as exc:
        build_pipeline("svm")

    assert "Unsupported model" in str(exc.value)


# --------------------------------------------------
# Hyperparameter Validation Tests
# --------------------------------------------------

def test_logistic_regression_parameters():
    pipeline = build_pipeline("logistic_regression")
    model = pipeline.named_steps["model"]

    assert model.max_iter == 1000
    assert model.class_weight == "balanced"
    assert model.random_state == 42


def test_random_forest_parameters():
    pipeline = build_pipeline("random_forest")
    model = pipeline.named_steps["model"]

    assert model.n_estimators == 400
    assert model.max_depth == 12
    assert model.min_samples_leaf == 30
    assert model.random_state == 42


def test_xgboost_parameters():
    pipeline = build_pipeline("xgboost")
    model = pipeline.named_steps["model"]

    assert model.n_estimators == 500
    assert model.max_depth == 6
    assert model.learning_rate == 0.05
    assert model.random_state == 42


def test_lightgbm_parameters():
    pipeline = build_pipeline("lightgbm")
    model = pipeline.named_steps["model"]

    assert model.n_estimators == 600
    assert model.learning_rate == 0.05
    assert model.num_leaves == 31
    assert model.random_state == 42


# --------------------------------------------------
# Fit & Predict Sanity Test
# --------------------------------------------------
@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_pipeline_fit_and_predict(model_name: str):
    pipeline = build_pipeline(model_name)

    # Create synthetic data
    x_raw = np.random.rand(20, 5)
    y = np.random.randint(0, 2, 20)


    feature_names = [f"feature_{i}" for i in range(x_raw.shape[1])]
    X = pd.DataFrame(x_raw, columns=feature_names)

    # Execute the pipeline lifecycle
    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    # Assertions
    assert len(preds) == len(y), f"Predictions length mismatch for {model_name}"
    assert set(np.unique(preds)).issubset({0, 1}), f"Invalid predictions in {model_name}"

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.models.train import (
    get_profile,
    clean_feature_names,
    compute_total_score,
    train_and_evaluate,
)


# --------------------------------------------------
# Fixtures
# --------------------------------------------------

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        "feature 1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "feature-2": [1, 0, 1, 0, 1, 0],
    })
    y = pd.Series([0, 1, 0, 1, 0, 1])
    return X.iloc[:4], y.iloc[:4], X.iloc[4:], y.iloc[4:]


@pytest.fixture
def sample_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])


@pytest.fixture
def mock_settings():
    return {
        "profiles": {
            "test_profile": {
                "evaluation": {
                    "scoring_weights": {
                        "precision": 0.5,
                        "recall": 0.5
                    }
                },
                "thresholds": {
                    "enabled": True,
                    "strategy": "metric_based",
                    "metric": "f1"
                },
                "business": {
                    "costs": {
                        "false_positive": 10,
                        "false_negative": 50
                    }
                },
                "registry": {
                    "experiment_name": "test_experiment",
                    "registered_model_name": "test_model"
                }
            }
        },
        "global": {
            "evaluation": {
                "cross_validation_folds": 2
            }
        }
    }


# --------------------------------------------------
# Unit Tests
# --------------------------------------------------

def test_get_profile_success(monkeypatch, mock_settings):
    monkeypatch.setattr(
        "fraud_detection.models.train.settings",
        mock_settings
    )
    profile = get_profile("test_profile")
    assert isinstance(profile, dict)


def test_get_profile_failure(monkeypatch, mock_settings):
    monkeypatch.setattr(
        "fraud_detection.models.train.settings",
        mock_settings
    )
    with pytest.raises(ValueError):
        get_profile("invalid_profile")


def test_clean_feature_names():
    df = pd.DataFrame(columns=["feature 1", "feature-2", "feature.3"])
    cleaned = clean_feature_names(df)

    assert "feature_1" in cleaned.columns
    assert "feature_2" in cleaned.columns
    assert "feature_3" in cleaned.columns


def test_compute_total_score(monkeypatch, mock_settings):
    monkeypatch.setattr(
        "fraud_detection.models.train.settings",
        mock_settings
    )

    metrics = {
        "precision": 0.8,
        "recall": 0.6,
        "business_score": 0.9
    }

    score = compute_total_score(metrics, "test_profile")

    expected_perf = (0.8 * 0.5) + (0.6 * 0.5)
    expected_total = expected_perf + 0.9

    assert score == pytest.approx(expected_total)


# --------------------------------------------------
# Integration Test (MLflow Mocked)
# --------------------------------------------------

@patch("fraud_detection.models.train.mlflow")
@patch("fraud_detection.models.train.set_mlflow_tracking")
def test_train_and_evaluate_metric_based(
    mock_set_tracking,
    mock_mlflow,
    sample_data,
    sample_pipeline,
    monkeypatch,
    mock_settings
):
    monkeypatch.setattr(
        "fraud_detection.models.train.settings",
        mock_settings
    )

    # Mock MLflow context manager
    mock_run = MagicMock()
    mock_run.info.run_id = "12345"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

    X_train, y_train, X_test, y_test = sample_data

    pipeline, metrics, threshold, run_id = train_and_evaluate(
        sample_pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
        model_name="test_model",
        profile_name="test_profile"
    )

    assert isinstance(pipeline, Pipeline)
    assert isinstance(metrics, dict)
    assert "precision" in metrics
    assert "business_score" in metrics
    assert "total_score" in metrics
    assert 0 <= threshold <= 1
    assert run_id == "12345"


# --------------------------------------------------
# Cost-Based Threshold Test
# --------------------------------------------------

@patch("fraud_detection.models.train.mlflow")
@patch("fraud_detection.models.train.set_mlflow_tracking")
def test_train_and_evaluate_cost_based(
    mock_set_tracking,
    mock_mlflow,
    sample_data,
    sample_pipeline,
    monkeypatch,
    mock_settings
):
    # Modify profile to cost-based strategy
    mock_settings["profiles"]["test_profile"]["thresholds"]["strategy"] = "cost_based"

    monkeypatch.setattr(
        "fraud_detection.models.train.settings",
        mock_settings
    )

    mock_run = MagicMock()
    mock_run.info.run_id = "cost_run"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

    X_train, y_train, X_test, y_test = sample_data

    _, metrics, threshold, run_id = train_and_evaluate(
        sample_pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
        model_name="test_model",
        profile_name="test_profile"
    )

    assert "business_score" in metrics
    assert 0 <= threshold <= 1
    assert run_id == "cost_run"

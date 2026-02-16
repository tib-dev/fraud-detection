import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from fraud_detection.models.tuning import (
    tune_model,
    tune_multiple_models,
)


# --------------------------------------------------
# Fixtures
# --------------------------------------------------

import pandas as pd

@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=60,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )

    columns = [f"feature_{i}" for i in range(X.shape[1])]
    x_df = pd.DataFrame(X, columns=columns)

    return x_df, y


@pytest.fixture
def base_pipeline():
    return Pipeline([
        ("model", LogisticRegression(max_iter=500))
    ])


@pytest.fixture
def small_param_grid():
    return {
        "model__C": [0.1, 1.0]
    }


# --------------------------------------------------
# tune_model - Grid Search
# --------------------------------------------------

def test_tune_model_grid(sample_data, base_pipeline, small_param_grid):
    X, y = sample_data

    best_estimator, cv_results, best_score, best_params = tune_model(
        pipeline=base_pipeline,
        param_grid=small_param_grid,
        X_train=X,
        y_train=y,
        scoring="f1",
        cv=2,
        search_type="grid",
    )

    assert isinstance(best_estimator, Pipeline)
    assert isinstance(cv_results, pd.DataFrame)
    assert isinstance(best_score, float)
    assert isinstance(best_params, dict)

    assert "rank_test_score" in cv_results.columns
    assert best_params["model__C"] in small_param_grid["model__C"]


# --------------------------------------------------
# tune_model - Random Search
# --------------------------------------------------

def test_tune_model_random(sample_data, base_pipeline):
    X, y = sample_data

    param_dist = {
        "model__C": [0.01, 0.1, 1.0],
    }

    best_estimator, cv_results, best_score, best_params = tune_model(
        pipeline=base_pipeline,
        param_grid=param_dist,
        X_train=X,
        y_train=y,
        scoring="f1",
        cv=2,
        search_type="random",
        n_iter=2,
        random_state=42,
    )

    assert isinstance(best_estimator, Pipeline)
    assert isinstance(cv_results, pd.DataFrame)
    assert isinstance(best_score, float)
    assert isinstance(best_params, dict)
    assert best_params["model__C"] in param_dist["model__C"]


# --------------------------------------------------
# Invalid search_type
# --------------------------------------------------

def test_tune_model_invalid_search_type(sample_data, base_pipeline, small_param_grid):
    X, y = sample_data

    with pytest.raises(ValueError) as exc:
        tune_model(
            pipeline=base_pipeline,
            param_grid=small_param_grid,
            X_train=X,
            y_train=y,
            search_type="invalid",
        )

    assert "search_type must be" in str(exc.value)


# --------------------------------------------------
# tune_multiple_models
# --------------------------------------------------

def test_tune_multiple_models(sample_data):
    X, y = sample_data

    pipeline = Pipeline([
        ("model", LogisticRegression(max_iter=500))
    ])

    models_config = {
        "logreg_model": {
            "pipeline": pipeline,
            "param_grid": {
                "model__C": [0.1, 1.0]
            }
        }
    }

    best_estimators, all_cv_results = tune_multiple_models(
        models_config=models_config,
        X_train=X,
        y_train=y,
        scoring="f1",
        cv=2,
        search_type="grid",
    )

    # Structure checks
    assert isinstance(best_estimators, dict)
    assert isinstance(all_cv_results, dict)

    assert "logreg_model" in best_estimators
    assert "logreg_model" in all_cv_results

    # Nested structure validation
    model_results = all_cv_results["logreg_model"]

    assert "cv_results" in model_results
    assert "best_score" in model_results
    assert "best_params" in model_results

    assert isinstance(model_results["cv_results"], pd.DataFrame)
    assert isinstance(model_results["best_score"], float)
    assert isinstance(model_results["best_params"], dict)


# --------------------------------------------------
# Deterministic Random Search Test
# --------------------------------------------------

def test_random_search_deterministic(sample_data, base_pipeline):
    X, y = sample_data

    param_dist = {"model__C": [0.01, 0.1, 1.0]}

    result1 = tune_model(
        pipeline=base_pipeline,
        param_grid=param_dist,
        X_train=X,
        y_train=y,
        search_type="random",
        cv=2,
        n_iter=2,
        random_state=42,
    )

    result2 = tune_model(
        pipeline=base_pipeline,
        param_grid=param_dist,
        X_train=X,
        y_train=y,
        search_type="random",
        cv=2,
        n_iter=2,
        random_state=42,
    )

    assert result1[3] == result2[3]  # best_params

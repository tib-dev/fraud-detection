import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from fraud_detection.explainability.feature_importance import (
    _get_estimator,
    get_builtin_feature_importance,
)


def test_get_estimator_plain_model():
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    estimator = _get_estimator(model)

    assert estimator is model


def test_get_estimator_pipeline():
    rf = RandomForestClassifier(n_estimators=10, random_state=42)

    pipeline = Pipeline(
        steps=[
            ("model", rf),
        ]
    )

    estimator = _get_estimator(pipeline)

    assert estimator is rf


def test_get_builtin_feature_importance_success():
    X = pd.DataFrame(
        np.random.rand(100, 4),
        columns=["f1", "f2", "f3", "f4"],
    )
    y = np.random.randint(0, 2, size=100)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    result = get_builtin_feature_importance(
        model=model,
        feature_names=X.columns.tolist(),
        top_n=3,
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["feature", "importance"]
    assert len(result) == 3
    assert result["importance"].is_monotonic_decreasing


def test_get_builtin_feature_importance_pipeline():
    X = pd.DataFrame(
        np.random.rand(50, 3),
        columns=["a", "b", "c"],
    )
    y = np.random.randint(0, 2, size=50)

    rf = RandomForestClassifier(n_estimators=5, random_state=0)
    pipeline = Pipeline(
        steps=[
            ("model", rf),
        ]
    )

    pipeline.fit(X, y)

    result = get_builtin_feature_importance(
        model=pipeline,
        feature_names=X.columns.tolist(),
        top_n=2,
    )

    assert len(result) == 2
    assert "feature" in result
    assert "importance" in result


def test_get_builtin_feature_importance_non_tree_model():
    X = pd.DataFrame(
        np.random.rand(30, 2),
        columns=["x1", "x2"],
    )
    y = np.random.randint(0, 2, size=30)

    model = LogisticRegression()
    model.fit(X, y)

    with pytest.raises(RuntimeError):
        get_builtin_feature_importance(
            model=model,
            feature_names=X.columns.tolist(),
        )


def test_get_builtin_feature_importance_feature_length_mismatch():
    X = pd.DataFrame(
        np.random.rand(40, 3),
        columns=["f1", "f2", "f3"],
    )
    y = np.random.randint(0, 2, size=40)

    model = RandomForestClassifier(n_estimators=5)
    model.fit(X, y)

    with pytest.raises(RuntimeError):
        get_builtin_feature_importance(
            model=model,
            feature_names=["f1", "f2"],  # mismatch
        )

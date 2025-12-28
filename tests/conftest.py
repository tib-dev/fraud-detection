# tests/conftest.py
import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def dummy_data():
    X_train = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5, 6],
        "f2": [2, 1, 2, 1, 2, 1],
    })
    y_train = pd.Series([0, 0, 0, 1, 1, 1])

    X_test = pd.DataFrame({
        "f1": [2, 5],
        "f2": [1, 2],
    })
    y_test = pd.Series([0, 1])

    return X_train, y_train, X_test, y_test


@pytest.fixture
def simple_pipeline():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=200)),
        ]
    )

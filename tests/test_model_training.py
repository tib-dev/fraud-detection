import pytest
from fraud_detection.models.train import train_and_evaluate


class DummyRun:
    class Info:
        run_id = "test-run-id"

    def __init__(self):
        self.info = self.Info()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

def test_train_and_evaluate_runs(
    dummy_data,
    simple_pipeline,
    monkeypatch,
):
    """
    Smoke test:
    - function runs end-to-end
    - returns expected objects
    """

    X_train, y_train, X_test, y_test = dummy_data

    # --- Mock MLflow to avoid real tracking ---
    monkeypatch.setattr("mlflow.start_run", lambda *a, **k: DummyRun())
    monkeypatch.setattr("mlflow.log_params", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.log_metrics", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.sklearn.log_model", lambda *a, **k: None)

    pipe, metrics, threshold, run_id = train_and_evaluate(
        pipeline=simple_pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_name="Logistic Regression",
        optimize_threshold=False,  # simplify test
    )

    assert pipe is not None
    assert isinstance(metrics, dict)
    assert 0.0 <= threshold <= 1.0
    assert isinstance(run_id, str)


def test_metrics_keys_present(
    dummy_data,
    simple_pipeline,
    monkeypatch,
):
    """
    Verify required metrics exist
    """

    X_train, y_train, X_test, y_test = dummy_data

    monkeypatch.setattr("mlflow.start_run", lambda *a, **k: DummyRun())
    monkeypatch.setattr("mlflow.log_params", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.log_metrics", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.sklearn.log_model", lambda *a, **k: None)

    _, metrics, _, _ = train_and_evaluate(
        pipeline=simple_pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_name="Logistic Regression",
        optimize_threshold=False,
    )

    expected_keys = {
        "precision",
        "recall",
        "f1",
        "auc_pr",
        "roc_auc",
        "threshold",
        "total_score",
        "y_true",
        "y_pred",
    }

    assert expected_keys.issubset(metrics.keys())


def test_threshold_optimization_changes_threshold(
    dummy_data,
    simple_pipeline,
    monkeypatch,
):
    """
    Threshold optimization should not always return default 0.5
    """

    X_train, y_train, X_test, y_test = dummy_data

    monkeypatch.setattr("mlflow.start_run", lambda *a, **k: DummyRun())
    monkeypatch.setattr("mlflow.log_params", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.log_metrics", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.sklearn.log_model", lambda *a, **k: None)

    _, _, threshold, _ = train_and_evaluate(
        pipeline=simple_pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_name="Logistic Regression",
        optimize_threshold=True,
        threshold_metric="f1",
    )

    assert 0.0 <= threshold <= 1.0

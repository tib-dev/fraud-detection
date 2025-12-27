import mlflow
import mlflow.sklearn
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

from fraud_detection.models.metrics import (
    compute_classification_metrics,
    find_best_threshold,
)
from fraud_detection.models.register import set_mlflow_tracking


def train_and_evaluate(
    pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    model_name: str,
    registered_model_name: str,
    optimize_threshold: bool = True,
    threshold_metric: str = "f1",
):
    
    """
    Train a classification pipeline, optimize the decision threshold,
    evaluate on a test set, and log everything to MLflow with model registry.

    This function:
    - Fits the provided sklearn / imblearn pipeline
    - Optionally optimizes the classification threshold using cross-validated
      probability predictions on the training set
    - Evaluates model performance on the test set
    - Logs parameters, metrics, and the trained model to MLflow
    - Registers the model under a specified registered model name

    Parameters
    ----------
    pipeline :
        Sklearn or imblearn pipeline with a classifier that implements
        `predict_proba`.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target labels.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Test target labels.
    model_name : str
        Human-readable name for the MLflow run (e.g. "XGBoost Credit Card").
    registered_model_name : str
        Name under which the model will be registered in MLflow
        (e.g. "credit_card_model" or "fraud_model").
    optimize_threshold : bool, default=True
        Whether to optimize the classification threshold using cross-validation
        on the training data.
    threshold_metric : str, default="f1"
        Metric used for threshold optimization. Supported values:
        "f1", "precision", "recall".

    Returns
    -------
    pipeline :
        The trained pipeline.
    metrics : dict
        Dictionary containing evaluation metrics such as precision, recall,
        F1-score, AUC-PR, ROC-AUC, and the selected threshold.
    threshold : float
        Final decision threshold used to convert probabilities into
        binary predictions.
    """

    # ------------------------------------------------------
    # MLflow setup
    # ------------------------------------------------------
    set_mlflow_tracking()

    # ------------------------------------------------------
    # Train
    # ------------------------------------------------------
    pipeline.fit(X_train, y_train)

    # ------------------------------------------------------
    # Threshold optimization (CV-based)
    # ------------------------------------------------------
    threshold = 0.5
    if optimize_threshold:
        y_train_proba = cross_val_predict(
            clone(pipeline),
            X_train,
            y_train,
            cv=3,
            method="predict_proba",
        )[:, 1]

        threshold, _ = find_best_threshold(
            y_train.values,
            y_train_proba,
            metric=threshold_metric,
        )

    # ------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = compute_classification_metrics(
        y_test.values,
        y_pred,
        y_proba,
    )
    metrics["threshold"] = threshold

    # ------------------------------------------------------
    # MLflow logging + model registry
    # ------------------------------------------------------
    with mlflow.start_run(run_name=model_name):

        mlflow.log_params({
            "model_name": model_name,
            "threshold_metric": threshold_metric,
            "threshold": threshold,
        })

        mlflow.log_metrics({
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auc_pr": metrics["auc_pr"],
            "roc_auc": metrics["roc_auc"],
        })

        # ✅ Correct modern API (no artifact_path warning)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name=registered_model_name,
        )

    return pipeline, metrics, threshold


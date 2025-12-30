import numpy as np
import shap
import pandas as pd


def _get_estimator(model):
    if hasattr(model, "named_steps"):
        return model.named_steps.get("model", model)
    return model


def compute_shap_values(model, X):
    """
    Compute SHAP values for tree-based models.

    Parameters
    ----------
    model : Pipeline or estimator
    X : pd.DataFrame

    Returns
    -------
    shap_values : np.ndarray
    explainer : shap.TreeExplainer
    """
    try:
        estimator = _get_estimator(model)

        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X)

        return shap_values, explainer

    except Exception as exc:
        raise RuntimeError("SHAP computation failed") from exc


def get_shap_importance(X, shap_values, top_n=10):
    """
    Compute mean absolute SHAP importance.
    """
    try:
        shap_matrix = (
            shap_values[1] if isinstance(shap_values, list) else shap_values
        )

        importance = (
            np.abs(shap_matrix)
            .mean(axis=0)
        )

        return (
            pd.DataFrame(
                {
                    "feature": X.columns,
                    "shap_importance": importance,
                }
            )
            .sort_values("shap_importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    except Exception as exc:
        raise RuntimeError("Failed to compute SHAP importance") from exc


def explain_single_prediction(explainer, shap_values, X, index):
    """
    Force plot explanation for a single row.
    """
    shap_matrix = (
        shap_values[1] if isinstance(shap_values, list) else shap_values
    )

    row_pos = X.index.get_loc(index)

    shap.force_plot(
        explainer.expected_value,
        shap_matrix[row_pos],
        X.loc[index],
        matplotlib=True,
    )

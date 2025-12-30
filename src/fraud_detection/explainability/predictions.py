import pandas as pd
import numpy as np


def build_prediction_frame(X, y_true, y_proba, threshold=0.5):
    """
    Build a unified dataframe containing predictions and probabilities.

    Parameters
    ----------
    X : pd.DataFrame
    y_true : pd.Series
    y_proba : np.ndarray
    threshold : float

    Returns
    -------
    pd.DataFrame
    """
    try:
        y_pred = (y_proba >= threshold).astype(int)

        df = X.copy()
        df["y_true"] = y_true.values
        df["y_pred"] = y_pred
        df["y_proba"] = y_proba
    except Exception as exc:
        raise RuntimeError("Failed to build prediction dataframe.") from exc

    return df


def sample_error_cases(df):
    """
    Sample representative TP, FP, FN cases.

    Returns
    -------
    dict[str, pd.Series]
    """
    try:
        return {
            "true_positive": df[(df.y_true == 1) & (df.y_pred == 1)].iloc[0],
            "false_positive": df[(df.y_true == 0) & (df.y_pred == 1)].iloc[0],
            "false_negative": df[(df.y_true == 1) & (df.y_pred == 0)].iloc[0],
        }
    except Exception as exc:
        raise RuntimeError("Failed to sample error cases.") from exc

import pandas as pd


def _get_estimator(model):
    """
    Safely extract estimator from a Pipeline or return the model itself.
    """
    if hasattr(model, "named_steps"):
        return model.named_steps.get("model", model)
    return model


def get_builtin_feature_importance(model, feature_names, top_n=10):
    """
    Extract built-in feature importance from tree-based models.

    Supports:
    - XGBoost
    - RandomForest
    - LightGBM
    - Pipeline wrapping these models

    Parameters
    ----------
    model : Pipeline or estimator
    feature_names : list[str]
    top_n : int

    Returns
    -------
    pd.DataFrame
    """
    try:
        estimator = _get_estimator(model)

        if not hasattr(estimator, "feature_importances_"):
            raise AttributeError

        importances = estimator.feature_importances_

        if len(importances) != len(feature_names):
            raise ValueError("Feature names do not match importance length")

        df = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": importances,
                }
            )
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        return df

    except Exception as exc:
        raise RuntimeError(
            "Failed to extract built-in feature importance. "
            "Ensure model is tree-based and feature order is preserved."
        ) from exc

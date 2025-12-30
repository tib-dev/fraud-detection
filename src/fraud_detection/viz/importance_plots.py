import pandas as pd
import shap
import matplotlib.pyplot as plt


def plot_feature_importance(df, title, top_n=10, figsize=(8, 5)):
    """
    Plot horizontal bar chart for feature importance.

    Parameters
    ----------
    df : pd.DataFrame
        Feature importance dataframe.
    title : str
        Plot title.
    top_n : int
        Number of features to display.
    figsize : tuple
        Figure size.
    """
    try:
        data = df.head(top_n)

        plt.figure(figsize=figsize)
        plt.barh(data.iloc[:, 0], data.iloc[:, 1])
        plt.gca().invert_yaxis()
        plt.title(title)
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        raise RuntimeError("Failed to plot feature importance.") from exc


def plot_shap_waterfall(
    explainer,
    shap_values,
    X,
    index,
    max_display=10
):
    """
    Professional single-prediction explanation using SHAP waterfall plot.
    """
    shap_matrix = (
        shap_values[1] if isinstance(shap_values, list) else shap_values
    )

    row_pos = X.index.get_loc(index)

    explanation = shap.Explanation(
        values=shap_matrix[row_pos],
        base_values=explainer.expected_value,
        data=X.loc[index],
        feature_names=X.columns,
    )

    shap.plots.waterfall(
        explanation,
        max_display=max_display,
        show=True,
    )


def plot_single_prediction_bar(
    shap_values,
    X,
    index,
    top_n=10,
):
    """
    Business-friendly explanation of a single prediction using a bar plot.

    Positive SHAP → pushes toward Fraud
    Negative SHAP → pushes toward Non-Fraud
    """

    # Handle binary classifiers (list of arrays)
    shap_matrix = shap_values[1] if isinstance(
        shap_values, list) else shap_values

    row_pos = X.index.get_loc(index)

    shap_row = pd.Series(
        shap_matrix[row_pos],
        index=X.columns,
        name="shap_value",
    )

    top_features = (
        shap_row.abs()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    values = shap_row[top_features].sort_values()

    colors = [
        "#e74c3c" if v > 0 else "#3498db"
        for v in values
    ]

    plt.figure(figsize=(8, 5))
    values.plot(kind="barh", color=colors)

    plt.axvline(0, color="black", linewidth=1)
    plt.title("Top Factors Influencing This Prediction")
    plt.xlabel("Impact on Fraud Probability")
    plt.tight_layout()
    plt.show()


def plot_single_prediction_force(
    explainer,
    shap_values,
    X,
    index,
):
    """
    Debugging-focused SHAP force plot for a single prediction.
    """

    shap_matrix = shap_values[1] if isinstance(
        shap_values, list) else shap_values

    row_pos = X.index.get_loc(index)

    shap.force_plot(
        explainer.expected_value,
        shap_matrix[row_pos],
        X.loc[index],
        matplotlib=True,
        show=True,
       
    )

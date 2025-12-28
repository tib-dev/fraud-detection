from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import pandas as pd


def tune_model(
    pipeline,
    param_grid: dict,
    X_train,
    y_train,
    scoring: str = "f1",
    cv: int = 3,
    search_type: str = "grid",
    n_iter: int = 20,
    random_state: int = 42,
):
    """
    Hyperparameter tuning for a scikit-learn pipeline.
    
    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The ML pipeline to tune.
    param_grid : dict
        Hyperparameters to search.
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training target.
    scoring : str
        Metric to optimize.
    cv : int
        Cross-validation folds.
    search_type : str
        'grid' for GridSearchCV, 'random' for RandomizedSearchCV.
    n_iter : int
        Number of iterations for RandomizedSearchCV.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    best_estimator : sklearn estimator
        Pipeline with best-found hyperparameters.
    cv_results : pd.DataFrame
        Dataframe with CV scores for all parameter combinations.
    best_score : float
        Best CV score achieved.
    best_params : dict
        Best hyperparameters.
    """
    if search_type == "grid":
        searcher = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
        )
    elif search_type == "random":
        searcher = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            scoring=scoring,
            n_iter=n_iter,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            random_state=random_state,
            return_train_score=True,
        )
    else:
        raise ValueError("search_type must be 'grid' or 'random'")

    searcher.fit(X_train, y_train)

    cv_results = pd.DataFrame(searcher.cv_results_).sort_values(
        "rank_test_score"
    )

    return (
        searcher.best_estimator_,
        cv_results,
        searcher.best_score_,
        searcher.best_params_,
    )


# Optional helper for multiple models
def tune_multiple_models(models_config, X_train, y_train, scoring="f1", cv=3):
    """
    Tune multiple pipelines in a single loop.
    
    models_config: dict
        Example:
        {
            "Random Forest": {
                "pipeline": rf_pipeline,
                "param_grid": {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [None, 10, 20]
                },
            },
            "XGBoost": {
                "pipeline": xgb_pipeline,
                "param_grid": {...},
            },
        }
    """
    best_estimators = {}
    all_cv_results = {}

    for name, cfg in models_config.items():
        print(f"\n=== Tuning {name} ===")
        best_est, cv_res, best_score, best_params = tune_model(
            pipeline=cfg["pipeline"],
            param_grid=cfg["param_grid"],
            X_train=X_train,
            y_train=y_train,
            scoring=scoring,
            cv=cv,
        )
        best_estimators[name] = best_est
        all_cv_results[name] = {
            "cv_results": cv_res,
            "best_score": best_score,
            "best_params": best_params,
        }

    return best_estimators, all_cv_results

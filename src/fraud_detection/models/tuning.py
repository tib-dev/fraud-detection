from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

def tune_model(
    pipeline: Pipeline,
    param_grid: Dict[str, Any],
    X_train,
    y_train,
    scoring: str = "f1",
    cv: int = 3,
    search_type: str = "grid",
    n_iter: int = 20,
    random_state: int = 42,
) -> Tuple[Pipeline, pd.DataFrame, float, Dict[str, Any]]:
    """
    Performs hyperparameter tuning for a single scikit-learn pipeline.

    Args:
        pipeline: The scikit-learn Pipeline object to tune.
        param_grid: Dictionary of parameters to search over.
        X_train: Training features.
        y_train: Training labels.
        scoring: Evaluation metric to optimize (e.g., "f1", "roc_auc").
        cv: Number of cross-validation folds.
        search_type: Type of search to perform ('grid' or 'random').
        n_iter: Number of parameter settings sampled (only used for 'random').
        random_state: Seed for reproducibility in random search.

    Returns:
        A tuple containing:
            - The best-fitted estimator (Pipeline).
            - A sorted DataFrame of all CV results.
            - The best score achieved.
            - The dictionary of best parameters found.

    Raises:
        ValueError: If search_type is not 'grid' or 'random'.
    """
    if search_type == "grid":
        searcher = GridSearchCV(
            pipeline, param_grid, scoring=scoring, cv=cv,
            n_jobs=-1, verbose=1, return_train_score=True
        )
    elif search_type == "random":
        searcher = RandomizedSearchCV(
            pipeline, param_distributions=param_grid, scoring=scoring,
            n_iter=n_iter, cv=cv, n_jobs=-1, verbose=1,
            random_state=random_state, return_train_score=True
        )
    else:
        raise ValueError("search_type must be 'grid' or 'random'")

    searcher.fit(X_train, y_train)
    cv_results = pd.DataFrame(searcher.cv_results_).sort_values("rank_test_score")

    return searcher.best_estimator_, cv_results, searcher.best_score_, searcher.best_params_

def tune_multiple_models(
    models_config: Dict[str, Dict[str, Any]],
    X_train,
    y_train,
    scoring: str = "f1",
    cv: int = 3,
    search_type: str = "grid",
    n_iter: int = 20,
    random_state: int = 42
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]:
    """
    Iterates through multiple model configurations to perform hyperparameter tuning.

    Args:
        models_config: Dictionary where keys are model names and values are
            dictionaries containing "pipeline" and "param_grid".
        X_train: Training features.
        y_train: Training labels.
        scoring: Evaluation metric to optimize.
        cv: Number of cross-validation folds.
        search_type: Type of search to perform ('grid' or 'random').
        n_iter: Number of parameter settings sampled (only used for 'random').
        random_state: Seed for reproducibility.

    Returns:
        A tuple containing:
            - A dictionary mapping model names to their best estimators.
            - A dictionary mapping model names to their results (CV results, best score, best params).
    """
    best_estimators, all_cv_results = {}, {}

    for name, cfg in models_config.items():
        print(f"--- Tuning model: {name} ---")
        best_est, cv_res, best_score, best_params = tune_model(
            cfg["pipeline"], cfg["param_grid"], X_train, y_train,
            scoring=scoring, cv=cv, search_type=search_type,
            n_iter=n_iter, random_state=random_state
        )
        best_estimators[name] = best_est
        all_cv_results[name] = {
            "cv_results": cv_res,
            "best_score": best_score,
            "best_params": best_params
        }

    return best_estimators, all_cv_results

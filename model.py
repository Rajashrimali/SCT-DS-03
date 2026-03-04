"""
model.py
--------
Decision Tree training with GridSearchCV hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from typing import Tuple, Dict, Any


DEFAULT_PARAM_GRID = {
    'max_depth':         [3, 5, 7, 10],
    'min_samples_split': [10, 20, 50],
    'min_samples_leaf':  [5, 10, 20],
    'criterion':         ['gini', 'entropy']
}


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, Any] = None,
    cv: int = 5,
    scoring: str = 'f1'
) -> Tuple[DecisionTreeClassifier, Dict[str, Any], float]:
    """
    Run GridSearchCV to find the best Decision Tree hyperparameters.

    Returns
    -------
    best_estimator : fitted DecisionTreeClassifier
    best_params    : dict of best parameters
    best_score     : best cross-validated score
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    grid = GridSearchCV(
        DecisionTreeClassifier(class_weight='balanced', random_state=42),
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune: bool = True,
    **kwargs
) -> Tuple[DecisionTreeClassifier, Dict[str, Any]]:
    """
    Train a Decision Tree classifier.

    Parameters
    ----------
    X_train : training features
    y_train : training labels
    tune    : if True, run GridSearchCV; else use kwargs as params
    **kwargs: passed directly to DecisionTreeClassifier if tune=False

    Returns
    -------
    model       : fitted DecisionTreeClassifier
    best_params : dict of hyperparameters used
    """
    if tune:
        model, best_params, cv_score = tune_hyperparameters(X_train, y_train)
        print(f"  ✓ Best CV F1: {cv_score:.4f}  |  Params: {best_params}")
    else:
        params = {
            'max_depth': 5,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'criterion': 'gini',
            'class_weight': 'balanced',
            'random_state': 42,
            **kwargs
        }
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        best_params = params

    return model, best_params


def cross_validate_model(
    model: DecisionTreeClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5
) -> np.ndarray:
    """Return array of cross-validated ROC-AUC scores."""
    return cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

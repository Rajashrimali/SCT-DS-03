"""Tests for src/model.py"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_generator import generate_bank_data
from src.preprocessor   import preprocess
from src.model          import train_decision_tree, cross_validate_model
from src.evaluate       import compute_metrics


@pytest.fixture(scope='module')
def trained_artifacts():
    df = generate_bank_data(n=600, seed=7)
    X_train, X_test, y_train, y_test, feature_names = preprocess(df)
    model, params = train_decision_tree(X_train, y_train, tune=False)
    return model, X_train, X_test, y_train, y_test, feature_names


def test_model_predicts(trained_artifacts):
    model, _, X_test, _, y_test, _ = trained_artifacts
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(preds).issubset({0, 1})


def test_model_probabilities(trained_artifacts):
    model, _, X_test, _, _, _ = trained_artifacts
    proba = model.predict_proba(X_test)
    assert proba.shape[1] == 2
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_metrics_in_range(trained_artifacts):
    model, _, X_test, _, y_test, _ = trained_artifacts
    metrics = compute_metrics(model, X_test, y_test)
    for key, val in metrics.items():
        assert 0.0 <= val <= 1.0, f"Metric '{key}' out of [0, 1] range: {val}"


def test_cv_scores(trained_artifacts):
    model, X_train, _, y_train, _, _ = trained_artifacts
    cv_scores = cross_validate_model(model, X_train, y_train, cv=3)
    assert len(cv_scores) == 3
    assert all(0.0 <= s <= 1.0 for s in cv_scores)


def test_feature_importances(trained_artifacts):
    model, _, _, _, _, feature_names = trained_artifacts
    importances = model.feature_importances_
    assert len(importances) == len(feature_names)
    assert abs(importances.sum() - 1.0) < 1e-6, "Importances should sum to 1"

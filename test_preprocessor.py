"""Tests for src/preprocessor.py"""

import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_generator import generate_bank_data
from src.preprocessor   import encode_features, preprocess, CAT_COLS


@pytest.fixture
def sample_df():
    return generate_bank_data(n=500, seed=0)


def test_encode_features_shape(sample_df):
    enc, encoders = encode_features(sample_df)
    assert enc.shape == sample_df.shape, "Shape should be unchanged after encoding"


def test_encode_features_no_strings(sample_df):
    enc, _ = encode_features(sample_df)
    for col in CAT_COLS:
        assert enc[col].dtype in [np.int32, np.int64], \
            f"Column '{col}' should be integer after encoding"


def test_preprocess_split_sizes(sample_df):
    X_train, X_test, y_train, y_test, _ = preprocess(sample_df, test_size=0.2)
    total = len(y_test) / (len(y_test) + len(y_train))
    assert abs(total - 0.2) < 0.05, "Test size should be approximately 20%"


def test_preprocess_feature_names(sample_df):
    _, _, _, _, feature_names = preprocess(sample_df)
    assert 'y' not in feature_names, "Target column should not be in feature names"
    assert len(feature_names) == sample_df.shape[1] - 1


def test_preprocess_no_nulls(sample_df):
    X_train, X_test, _, _, _ = preprocess(sample_df)
    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0

"""
preprocessor.py
---------------
Feature encoding, train/test split, and class-imbalance handling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from typing import Tuple, List, Dict


CAT_COLS = [
    'job', 'marital', 'education', 'default',
    'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome'
]


def encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Label-encode all categorical columns (excluding 'y').

    Returns
    -------
    df_encoded : pd.DataFrame
    encoders   : dict mapping column name -> fitted LabelEncoder
    """
    df_enc   = df.copy()
    encoders = {}
    for col in CAT_COLS:
        if col in df_enc.columns:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            encoders[col] = le
    return df_enc, encoders


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    oversample_ratio: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df               : raw DataFrame (must contain column 'y')
    test_size        : fraction for test split
    seed             : random seed
    oversample_ratio : minority / majority ratio after oversampling

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    df_enc, _ = encode_features(df)

    X = df_enc.drop('y', axis=1)
    # Convert target to binary int
    y_raw = df['y']  # use original string column
    y = (y_raw == 'yes').astype(int)
    y.name = 'target'

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Oversample minority class in training set
    train_df  = pd.concat([X_train, y_train], axis=1)
    majority  = train_df[train_df['target'] == 0]
    minority  = train_df[train_df['target'] == 1]

    if len(minority) == 0:
        # No minority samples — return as-is
        return X_train, X_test, y_train, y_test, feature_names

    n_minority_up = max(len(minority), int(len(majority) * oversample_ratio))
    minority_up   = resample(minority, replace=True,
                             n_samples=n_minority_up, random_state=seed)
    balanced  = pd.concat([majority, minority_up])

    X_train_bal = balanced.drop('target', axis=1)
    y_train_bal = balanced['target']

    return X_train_bal, X_test, y_train_bal, y_test, feature_names

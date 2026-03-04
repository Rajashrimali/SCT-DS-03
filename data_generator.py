"""
data_generator.py
-----------------
Generates a synthetic dataset matching the UCI Bank Marketing schema.
Can also load the real CSV if provided.
"""

import numpy as np
import pandas as pd


def generate_bank_data(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset with the UCI Bank Marketing schema.

    Parameters
    ----------
    n    : int   – Number of rows to generate (default 10,000)
    seed : int   – Random seed for reproducibility

    Returns
    -------
    pd.DataFrame with 21 columns (20 features + target 'y')
    """
    np.random.seed(seed)

    jobs    = ['admin.', 'technician', 'services', 'management', 'retired',
               'blue-collar', 'student', 'entrepreneur', 'housemaid',
               'self-employed', 'unemployed']
    marry   = ['married', 'single', 'divorced']
    edu     = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
               'professional.course', 'university.degree', 'illiterate']
    months  = ['jan','feb','mar','apr','may','jun',
               'jul','aug','sep','oct','nov','dec']
    days    = ['mon','tue','wed','thu','fri']
    contact = ['cellular', 'telephone']

    age          = np.random.normal(40, 12, n).clip(18, 95).astype(int)
    job          = np.random.choice(jobs, n,
                       p=[0.25,0.17,0.09,0.10,0.07,0.10,0.04,0.04,0.03,0.04,0.07])
    marital      = np.random.choice(marry, n, p=[0.61, 0.28, 0.11])
    education    = np.random.choice(edu, n,
                       p=[0.10, 0.06, 0.15, 0.23, 0.13, 0.30, 0.03])
    default      = np.random.choice(['no','yes','unknown'], n, p=[0.79,0.02,0.19])
    housing      = np.random.choice(['yes','no','unknown'], n, p=[0.52,0.45,0.03])
    loan         = np.random.choice(['yes','no','unknown'], n, p=[0.16,0.81,0.03])
    contact_type = np.random.choice(contact, n, p=[0.64, 0.36])
    month        = np.random.choice(months, n)
    day_of_week  = np.random.choice(days, n)
    duration     = np.random.exponential(250, n).clip(0, 4918).astype(int)
    campaign     = np.random.poisson(2.5, n).clip(1, 56)
    pdays        = np.where(np.random.rand(n) < 0.85, 999,
                            np.random.randint(0, 28, n))
    previous     = np.where(pdays == 999, 0, np.random.randint(0, 7, n))
    poutcome     = np.where(pdays == 999, 'nonexistent',
                   np.random.choice(['failure','success'], n, p=[0.73, 0.27]))

    emp_var_rate   = np.random.normal(-0.1, 1.6, n).round(1)
    cons_price_idx = np.random.normal(93.6, 0.6, n).round(3)
    cons_conf_idx  = np.random.normal(-40.5, 4.6, n).round(1)
    euribor3m      = np.random.normal(3.6, 1.7, n).clip(0.6, 5.1).round(3)
    nr_employed    = np.random.choice([5099.1, 5191.0, 5228.1, 5008.7], n)

    # Logistic target generation (~8-11% positive rate)
    logit = (
        -1.8
        + 0.012  * (age - 40)
        + 0.5    * (job == 'student').astype(float)
        + 0.6    * (job == 'retired').astype(float)
        + 0.4    * (education == 'university.degree').astype(float)
        + 0.003  * duration
        - 0.10   * (campaign - 1)
        + 2.0    * (poutcome == 'success').astype(float)
        + 0.4    * (contact_type == 'cellular').astype(float)
        - 0.3    * (housing == 'yes').astype(float)
        - 0.2    * (loan == 'yes').astype(float)
        - 0.4    * emp_var_rate
        + 0.05   * cons_conf_idx
    )
    prob = 1 / (1 + np.exp(-logit))
    y    = np.where(np.random.rand(n) < prob, 'yes', 'no')

    df = pd.DataFrame({
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'housing': housing, 'loan': loan,
        'contact': contact_type, 'month': month, 'day_of_week': day_of_week,
        'duration': duration, 'campaign': campaign, 'pdays': pdays,
        'previous': previous, 'poutcome': poutcome,
        'emp_var_rate': emp_var_rate, 'cons_price_idx': cons_price_idx,
        'cons_conf_idx': cons_conf_idx, 'euribor3m': euribor3m,
        'nr_employed': nr_employed, 'y': y
    })
    return df


def load_uci_data(filepath: str) -> pd.DataFrame:
    """
    Load the real UCI Bank Marketing CSV file.

    Parameters
    ----------
    filepath : str – Path to bank-additional-full.csv (semicolon-separated)

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(filepath, sep=';')
    df.columns = df.columns.str.lower().str.strip()
    return df

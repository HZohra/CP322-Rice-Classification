"""
Reusable logistic regression models for rice classification.

Course: CP322 - Machine Learning
Author: Zohra Haidary
"""

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


RANDOM_STATE = 42

C_VALUES = [
    0.001, 0.01, 0.1, 1, 10, 100, 1000
]


def create_baseline_model(random_state=RANDOM_STATE):
    """Create unregularized logistic regression."""

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=np.inf,
            solver="lbfgs",
            max_iter=10000,
            tol=1e-8,
            random_state=random_state
        ))
    ])


def create_ridge_model(C=100, random_state=RANDOM_STATE):
    """Create L2-regularized logistic regression."""

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            l1_ratio=0.0,
            C=C,
            solver="lbfgs",
            max_iter=10000,
            tol=1e-8,
            random_state=random_state
        ))
    ])


def create_lasso_model(C=1, random_state=RANDOM_STATE):
    """Create L1-regularized logistic regression."""

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            l1_ratio=1.0,
            C=C,
            solver="saga",
            max_iter=50000,
            tol=1e-6,
            random_state=random_state
        ))
    ])


def create_all_models(
    ridge_C=100,
    lasso_C=1,
    random_state=RANDOM_STATE
):
    """Return all three logistic regression pipelines."""

    return {
        "Baseline": create_baseline_model(
            random_state=random_state
        ),

        "Ridge": create_ridge_model(
            C=ridge_C,
            random_state=random_state
        ),

        "Lasso": create_lasso_model(
            C=lasso_C,
            random_state=random_state
        )
    }
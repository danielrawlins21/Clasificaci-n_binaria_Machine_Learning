from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_external_split(y: pd.Series, *, test_size: float, random_state: int) -> pd.DataFrame:
    idx = np.arange(len(y))
    _, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return pd.DataFrame(
        {
            "row_idx": idx,
            "set": np.where(np.isin(idx, idx_test), "test", "train_val"),
        }
    )


def build_phase2_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    nominal_cols = [c for c in ["V1", "V3"] if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in nominal_cols]

    nominal_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("nominal", nominal_pipe, nominal_cols),
            ("numeric", numeric_pipe, numeric_cols),
        ],
        remainder="drop",
    )


def evaluate_cv_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv: StratifiedKFold,
    random_state: int,
) -> Dict[str, float]:
    model = Pipeline(
        steps=[
            ("prep", build_phase2_preprocessor(X)),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state, solver="lbfgs")),
        ]
    )
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "recall": "recall",
            "precision": "precision",
            "prauc": "average_precision",
        },
        error_score="raise",
    )

    return {
        "acc_mean": float(np.mean(scores["test_accuracy"])),
        "recall_mean": float(np.mean(scores["test_recall"])),
        "precision_mean": float(np.mean(scores["test_precision"])),
        "prauc_mean": float(np.mean(scores["test_prauc"])),
        "acc_std": float(np.std(scores["test_accuracy"], ddof=1)),
        "recall_std": float(np.std(scores["test_recall"], ddof=1)),
    }

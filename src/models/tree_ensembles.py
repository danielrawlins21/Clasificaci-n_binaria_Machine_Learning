from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict


@dataclass(frozen=True)
class EnsembleTuningResult:
    model_name: str
    search: RandomizedSearchCV
    cv_probabilities: np.ndarray


def tune_ensemble_model(
    *,
    model_name: str,
    X_train: Any,
    y_train: Any,
    cv: StratifiedKFold,
    random_state: int,
    n_iter: int,
    objective: str,
) -> EnsembleTuningResult:
    if model_name == "RF":
        estimator = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        param_distributions = {
            "n_estimators": [300, 500, 700],
            "max_depth": [None, 10, 14, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", 0.6, 0.8],
            "class_weight": [None, "balanced", "balanced_subsample"],
        }
    elif model_name == "ET":
        estimator = ExtraTreesClassifier(random_state=random_state, n_jobs=-1)
        param_distributions = {
            "n_estimators": [300, 500, 700],
            "max_depth": [None, 10, 14, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", 0.6, 0.8],
            "class_weight": [None, "balanced"],
        }
    else:
        raise ValueError("model_name debe ser 'RF' o 'ET'")

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=objective,
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
    )
    search.fit(X_train, y_train)

    cv_probabilities = cross_val_predict(
        clone(search.best_estimator_),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    return EnsembleTuningResult(
        model_name=model_name,
        search=search,
        cv_probabilities=cv_probabilities,
    )

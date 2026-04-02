from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ProjectConfig, get_default_config


def _intersect(existing: List[str], requested: Tuple[str, ...]) -> List[str]:
    req = set(requested)
    return [c for c in existing if c in req]


def infer_column_groups(
    X: pd.DataFrame,
    config: ProjectConfig | None = None,
) -> Dict[str, List[str]]:
    cfg = config or get_default_config()
    cols = X.columns.tolist()
    return {
        "nominal": _intersect(cols, cfg.nominal_columns),
        "binary": _intersect(cols, cfg.binary_columns),
        "discrete": _intersect(cols, cfg.discrete_columns),
        "continuous": _intersect(cols, cfg.continuous_columns),
    }


def build_linear_mlp_preprocessor(
    X: pd.DataFrame,
    config: ProjectConfig | None = None,
) -> ColumnTransformer:
    """Preprocesado para regresion logistica y MLP (con escalado)."""
    groups = infer_column_groups(X, config)

    nominal_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_cols = groups["binary"] + groups["discrete"] + groups["continuous"]
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("nominal", nominal_pipe, groups["nominal"]),
            ("numeric", numeric_pipe, numeric_cols),
        ],
        remainder="drop",
    )


def build_tree_preprocessor(
    X: pd.DataFrame,
    config: ProjectConfig | None = None,
) -> ColumnTransformer:
    """Preprocesado para arbol: sin escalado, con codificacion nominal."""
    groups = infer_column_groups(X, config)

    nominal_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_cols = groups["binary"] + groups["discrete"] + groups["continuous"]
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("nominal", nominal_pipe, groups["nominal"]),
            ("numeric", numeric_pipe, numeric_cols),
        ],
        remainder="drop",
    )

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from src.config import ProjectConfig, get_default_config


class DataSchemaError(ValueError):
    """Error de esquema del dataset."""


def read_raw_dataset(config: ProjectConfig | None = None) -> pd.DataFrame:
    cfg = config or get_default_config()
    if not cfg.data_file.exists():
        raise FileNotFoundError(f"No se encontro el dataset en: {cfg.data_file}")
    return pd.read_csv(cfg.data_file, na_values=list(cfg.missing_tokens))


def validate_expected_columns(df: pd.DataFrame, config: ProjectConfig | None = None) -> None:
    cfg = config or get_default_config()
    expected = set(cfg.expected_columns)
    current = set(df.columns)

    missing = sorted(expected - current)
    extra = sorted(current - expected)

    if missing or extra:
        raise DataSchemaError(
            "Esquema inesperado en dataset. "
            f"Faltan={missing if missing else []}; "
            f"Sobrantes={extra if extra else []}"
        )


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(4)
    out = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(dropna=True),
    })
    return out.sort_values(by="missing_count", ascending=False)


def build_model_base_dataset(
    df_raw: pd.DataFrame,
    config: ProjectConfig | None = None,
    drop_duplicates: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Aplica las decisiones estructurales previas al modelado."""
    cfg = config or get_default_config()
    df = df_raw.copy()

    n_rows_raw = len(df)
    duplicates_removed = 0
    if drop_duplicates:
        duplicates_removed = int(df.duplicated().sum())
        df = df.drop_duplicates().copy()

    cols_to_drop = [c for c in cfg.columns_to_drop_by_design if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    metadata = {
        "n_rows_raw": n_rows_raw,
        "n_rows_model_base": len(df),
        "duplicates_removed": duplicates_removed,
        "dropped_columns": cols_to_drop,
    }
    return df, metadata


def split_features_target(
    df: pd.DataFrame,
    config: ProjectConfig | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    cfg = config or get_default_config()
    if cfg.target_col not in df.columns:
        raise DataSchemaError(f"No existe la variable objetivo '{cfg.target_col}'")

    X = df.drop(columns=[cfg.target_col])
    y = df[cfg.target_col]
    return X, y

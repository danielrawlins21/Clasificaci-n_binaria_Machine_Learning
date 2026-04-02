from __future__ import annotations

import sys
from pathlib import Path

# Permite ejecutar tanto `python -m src.main` como `python .\src\main.py`.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_default_config
from src.data.load_data import (
    build_model_base_dataset,
    get_missing_summary,
    read_raw_dataset,
    split_features_target,
    validate_expected_columns,
)
from src.data.preprocessing import (
    build_linear_mlp_preprocessor,
    build_tree_preprocessor,
    infer_column_groups,
)
from src.utils.seeds import set_global_seed


def main() -> None:
    cfg = get_default_config()
    set_global_seed(cfg.random_state)

    df_raw = read_raw_dataset(cfg)
    validate_expected_columns(df_raw, cfg)

    df_model, metadata = build_model_base_dataset(df_raw, cfg, drop_duplicates=True)
    X, y = split_features_target(df_model, cfg)

    missing = get_missing_summary(df_raw)
    groups = infer_column_groups(X, cfg)

    _ = build_linear_mlp_preprocessor(X, cfg)
    _ = build_tree_preprocessor(X, cfg)

    print("Dataset bruto:", df_raw.shape)
    print("Dataset modelado:", df_model.shape)
    print("Metadatos base:", metadata)
    print("Target balance:")
    print(y.value_counts(normalize=True).sort_index())
    print("\nMissing top 5:")
    print(missing.head(5))
    print("\nGrupos de columnas:")
    print(groups)


if __name__ == "__main__":
    main()

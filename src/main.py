from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Permite ejecutar tanto `python -m src.main` como `python src/main.py`.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ProjectConfig, get_default_config
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
from src.evaluation.comparison import build_decisions_and_summaries, write_action_plan_status
from src.evaluation.metrics import create_external_split, evaluate_cv_metrics
from src.features.selection import (
    apply_candidate_feature,
    build_feature_eval_template,
    feature_candidates_catalog,
)
from src.utils.seeds import set_global_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline principal del proyecto.")
    parser.add_argument("--phase2", action="store_true", help="Ejecuta la Fase 2 industrializada.")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--recall-tol-acc", type=float, default=None)
    parser.add_argument("--precision-tol-rec", type=float, default=None)
    parser.add_argument("--prauc-tol-rec", type=float, default=None)
    parser.add_argument("--std-increase-limit", type=float, default=None)
    return parser.parse_args(argv)


def _build_config(project_root: str | None) -> ProjectConfig:
    if project_root:
        return ProjectConfig(project_root=Path(project_root).resolve())
    return get_default_config()


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _run_default_validation(cfg: ProjectConfig, random_state: int) -> None:
    set_global_seed(random_state)
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


def _run_phase2(cfg: ProjectConfig, args: argparse.Namespace) -> dict[str, Any]:
    random_state = cfg.random_state if args.random_state is None else args.random_state
    test_size = cfg.phase2_test_size if args.test_size is None else args.test_size
    n_splits = cfg.phase2_n_splits if args.n_splits is None else args.n_splits
    recall_tol_acc = (
        cfg.phase2_recall_tolerance_acc if args.recall_tol_acc is None else args.recall_tol_acc
    )
    precision_tol_rec = (
        cfg.phase2_precision_tolerance_rec if args.precision_tol_rec is None else args.precision_tol_rec
    )
    prauc_tol_rec = cfg.phase2_prauc_tolerance_rec if args.prauc_tol_rec is None else args.prauc_tol_rec
    std_increase_limit = (
        cfg.phase2_std_increase_limit if args.std_increase_limit is None else args.std_increase_limit
    )

    set_global_seed(random_state)
    cfg.outputs_tables_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_interim_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_processed_dir.mkdir(parents=True, exist_ok=True)

    df_raw = read_raw_dataset(cfg)
    validate_expected_columns(df_raw, cfg)
    df_model_base, _ = build_model_base_dataset(df_raw, cfg, drop_duplicates=True)
    X_model, y_model = split_features_target(df_model_base, cfg)

    split_df = create_external_split(y_model, test_size=test_size, random_state=random_state)
    split_df.to_csv(cfg.outputs_tables_dir / "fase2_split_indices.csv", index=False)

    split_config = {
        "random_state": random_state,
        "test_size": test_size,
        "n_splits_cv": n_splits,
        "dataset_shape": list(df_model_base.shape),
    }
    with open(cfg.outputs_tables_dir / "fase2_split_config.json", "w", encoding="utf-8") as f:
        json.dump(split_config, f, ensure_ascii=False, indent=2)

    feature_candidates = feature_candidates_catalog()
    feature_candidates.to_csv(cfg.outputs_tables_dir / "fase2_feature_candidates.csv", index=False)
    template_df = build_feature_eval_template(feature_candidates["feature"].tolist())
    template_df.to_csv(cfg.outputs_tables_dir / "fase2_feature_eval_template.csv", index=False)

    train_idx = split_df.loc[split_df["set"] == "train_val", "row_idx"].to_numpy()
    test_idx = split_df.loc[split_df["set"] == "test", "row_idx"].to_numpy()
    df_train_val = df_model_base.iloc[train_idx].reset_index(drop=True)
    df_test = df_model_base.iloc[test_idx].reset_index(drop=True)
    _write_parquet(df_model_base, cfg.data_interim_dir / "fase2_model_base.parquet")
    _write_parquet(df_train_val, cfg.data_interim_dir / "fase2_train_val.parquet")
    _write_parquet(df_test, cfg.data_interim_dir / "fase2_test.parquet")

    X_train_val = X_model.iloc[train_idx].reset_index(drop=True)
    y_train_val = y_model.iloc[train_idx].reset_index(drop=True)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    base_metrics = evaluate_cv_metrics(X_train_val, y_train_val, cv=cv, random_state=random_state)

    rows: list[dict[str, Any]] = []
    for candidate in feature_candidates["feature"]:
        X_ext = apply_candidate_feature(X_train_val, candidate)
        ext_metrics = evaluate_cv_metrics(X_ext, y_train_val, cv=cv, random_state=random_state)
        for objective in ("accuracy", "recall"):
            rows.append(
                {
                    "candidate": candidate,
                    "objective": objective,
                    "acc_base_mean": base_metrics["acc_mean"],
                    "acc_ext_mean": ext_metrics["acc_mean"],
                    "recall_base_mean": base_metrics["recall_mean"],
                    "recall_ext_mean": ext_metrics["recall_mean"],
                    "precision_base_mean": base_metrics["precision_mean"],
                    "precision_ext_mean": ext_metrics["precision_mean"],
                    "prauc_base_mean": base_metrics["prauc_mean"],
                    "prauc_ext_mean": ext_metrics["prauc_mean"],
                    "primary_std_base": base_metrics["acc_std"] if objective == "accuracy" else base_metrics["recall_std"],
                    "primary_std_ext": ext_metrics["acc_std"] if objective == "accuracy" else ext_metrics["recall_std"],
                }
            )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(cfg.outputs_tables_dir / "fase2_feature_eval_metrics.csv", index=False)

    decisions_df, summary_counts, summary_delta = build_decisions_and_summaries(
        metrics_df,
        recall_tolerance_acc=recall_tol_acc,
        precision_tolerance_rec=precision_tol_rec,
        prauc_tolerance_rec=prauc_tol_rec,
        std_increase_limit=std_increase_limit,
    )
    decisions_df.to_csv(cfg.outputs_tables_dir / "fase2_feature_eval_decisions.csv", index=False)
    summary_counts.to_csv(cfg.outputs_tables_dir / "fase2_feature_eval_summary_counts.csv", index=False)
    summary_delta.to_csv(cfg.outputs_tables_dir / "fase2_feature_eval_summary_delta.csv", index=False)

    accepted_features = sorted(
        set(decisions_df.loc[decisions_df["decision"] == "aceptar", "candidate"].tolist())
    )
    X_train_val_ready = X_train_val.copy()
    X_test_ready = X_model.iloc[test_idx].reset_index(drop=True)
    for candidate in accepted_features:
        X_train_val_ready = apply_candidate_feature(X_train_val_ready, candidate)
        X_test_ready = apply_candidate_feature(X_test_ready, candidate)
    df_train_val_ready = pd.concat([X_train_val_ready, y_train_val], axis=1)
    df_test_ready = pd.concat([X_test_ready, y_model.iloc[test_idx].reset_index(drop=True)], axis=1)
    _write_parquet(df_train_val_ready, cfg.data_processed_dir / "fase2_train_val_ready.parquet")
    _write_parquet(df_test_ready, cfg.data_processed_dir / "fase2_test_ready.parquet")

    selected_candidates = {
        "accuracy": sorted(
            decisions_df.loc[
                (decisions_df["objective"] == "accuracy") & (decisions_df["decision"] == "aceptar"),
                "candidate",
            ].unique().tolist()
        ),
        "recall": sorted(
            decisions_df.loc[
                (decisions_df["objective"] == "recall") & (decisions_df["decision"] == "aceptar"),
                "candidate",
            ].unique().tolist()
        ),
    }
    with open(cfg.data_processed_dir / "fase2_selected_candidates.json", "w", encoding="utf-8") as f:
        json.dump(selected_candidates, f, ensure_ascii=False, indent=2)

    action_plan = write_action_plan_status(cfg.outputs_tables_dir, cfg)
    artifact_paths = {name: str(cfg.outputs_tables_dir / name) for name in cfg.phase2_artifact_filenames}
    missing_artifacts = [name for name, path in artifact_paths.items() if not Path(path).exists()]
    if missing_artifacts:
        raise RuntimeError(f"No se generaron todos los artefactos de Fase 2: {missing_artifacts}")

    return {
        "split_config": split_config,
        "selected_candidates": selected_candidates,
        "action_plan": action_plan,
        "artifact_paths": artifact_paths,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _build_config(args.project_root)
    random_state = cfg.random_state if args.random_state is None else args.random_state

    if args.phase2:
        result = _run_phase2(cfg, args)
        print("Fase 2 completada.")
        print("Candidatas aceptadas:", result["selected_candidates"])
        return 0

    _run_default_validation(cfg, random_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

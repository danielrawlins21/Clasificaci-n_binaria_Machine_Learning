from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import ProjectConfig


PHASE3_SUMMARY_FILENAME = "fase3_model_selection_summary.json"


def export_phase3_artifacts(
    *,
    cfg: ProjectConfig,
    threshold_table_acc: pd.DataFrame,
    threshold_table_rec: pd.DataFrame,
    comparison_df: pd.DataFrame,
    top_features_accuracy: list[str],
    top_features_recall: list[str],
    best_params_accuracy: dict[str, Any],
    best_params_recall: dict[str, Any],
    threshold_accuracy: float,
    threshold_recall: float,
) -> None:
    cfg.outputs_tables_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_processed_dir.mkdir(parents=True, exist_ok=True)

    threshold_table_acc.to_csv(cfg.outputs_tables_dir / "fase3_threshold_search_accuracy.csv", index=False)
    threshold_table_rec.to_csv(cfg.outputs_tables_dir / "fase3_threshold_search_recall.csv", index=False)
    comparison_df.to_csv(cfg.outputs_tables_dir / "fase3_rfe_tree_mlp_comparison.csv", index=False)

    pd.DataFrame({"rfe_feature": top_features_accuracy}).to_csv(
        cfg.outputs_tables_dir / "fase3_rfe_selected_features.csv", index=False
    )
    pd.DataFrame({"tree_feature": top_features_recall}).to_csv(
        cfg.outputs_tables_dir / "fase3_tree_selected_features.csv", index=False
    )

    summary_json = {
        "rfe_features": top_features_accuracy,
        "tree_features": top_features_recall,
        "best_params_accuracy": best_params_accuracy,
        "best_params_recall": best_params_recall,
        "threshold_accuracy": float(threshold_accuracy),
        "threshold_recall": float(threshold_recall),
    }
    with open(cfg.data_processed_dir / PHASE3_SUMMARY_FILENAME, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)


def check_phase3_artifacts(cfg: ProjectConfig) -> list[str]:
    missing: list[str] = []
    for name in cfg.phase3_artifact_filenames:
        if not (cfg.outputs_tables_dir / name).exists():
            missing.append(name)
    summary = cfg.data_processed_dir / PHASE3_SUMMARY_FILENAME
    if not summary.exists():
        missing.append(str(Path("data/processed") / PHASE3_SUMMARY_FILENAME))
    return missing

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.config import get_default_config


def test_phase2_smoke_generates_expected_artifacts(tmp_path: Path) -> None:
    cfg = get_default_config()
    raw_src = cfg.data_file

    project_root = tmp_path / "project"
    (project_root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_src, project_root / "data" / "raw" / "BBDD_ML_TAREA.csv")

    cmd = [sys.executable, "-m", "src.main", "--phase2", "--project-root", str(project_root)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

    tables = project_root / "outputs" / "tables"
    expected = [
        "fase2_split_indices.csv",
        "fase2_split_config.json",
        "fase2_feature_candidates.csv",
        "fase2_feature_eval_template.csv",
        "fase2_feature_eval_metrics.csv",
        "fase2_feature_eval_decisions.csv",
        "fase2_feature_eval_summary_counts.csv",
        "fase2_feature_eval_summary_delta.csv",
        "fase2_action_plan_status.csv",
    ]
    for name in expected:
        assert (tables / name).exists(), f"Falta artefacto: {name}"

    metrics = pd.read_csv(tables / "fase2_feature_eval_metrics.csv")
    assert list(metrics.columns) == [
        "candidate",
        "objective",
        "acc_base_mean",
        "acc_ext_mean",
        "recall_base_mean",
        "recall_ext_mean",
        "precision_base_mean",
        "precision_ext_mean",
        "prauc_base_mean",
        "prauc_ext_mean",
        "primary_std_base",
        "primary_std_ext",
    ]
    decisions = pd.read_csv(tables / "fase2_feature_eval_decisions.csv")
    assert list(decisions.columns) == [
        "candidate",
        "objective",
        "acc_base_mean",
        "acc_ext_mean",
        "recall_base_mean",
        "recall_ext_mean",
        "precision_base_mean",
        "precision_ext_mean",
        "prauc_base_mean",
        "prauc_ext_mean",
        "primary_std_base",
        "primary_std_ext",
        "decision",
        "reason",
    ]
    action_plan = pd.read_csv(tables / "fase2_action_plan_status.csv")
    assert set(action_plan["status"].tolist()) == {"hecho"}

    assert (project_root / "data" / "interim" / "fase2_model_base.parquet").exists()
    assert (project_root / "data" / "interim" / "fase2_train_val.parquet").exists()
    assert (project_root / "data" / "interim" / "fase2_test.parquet").exists()
    assert (project_root / "data" / "processed" / "fase2_train_val_ready.parquet").exists()
    assert (project_root / "data" / "processed" / "fase2_test_ready.parquet").exists()
    assert (project_root / "data" / "processed" / "fase2_selected_candidates.json").exists()

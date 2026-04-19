from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.config import get_default_config


def test_phase3_smoke_generates_expected_artifacts(tmp_path: Path) -> None:
    cfg = get_default_config()
    raw_src = cfg.data_file

    project_root = tmp_path / "project"
    (project_root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_src, project_root / "data" / "raw" / "BBDD_ML_TAREA.csv")

    cmd = [sys.executable, "-m", "src.main", "--phase3", "--project-root", str(project_root)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

    tables = project_root / "outputs" / "tables"
    expected = [
        "fase3_threshold_search_accuracy.csv",
        "fase3_threshold_search_recall.csv",
        "fase3_rfe_tree_mlp_comparison.csv",
        "fase3_rfe_selected_features.csv",
        "fase3_tree_selected_features.csv",
    ]
    for name in expected:
        assert (tables / name).exists(), f"Falta artefacto: {name}"

    comparison = pd.read_csv(tables / "fase3_rfe_tree_mlp_comparison.csv")
    assert set(comparison["model"].tolist()) == {"RF_Accuracy", "ET_Recall"}

    processed = project_root / "data" / "processed"
    assert (processed / "fase3_model_selection_summary.json").exists()
    assert (processed / "fase3_reproducibility_report.json").exists()

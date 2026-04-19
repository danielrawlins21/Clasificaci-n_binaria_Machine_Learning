from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from src.config import ProjectConfig


def generate_reproducibility_report(
    *,
    cfg: ProjectConfig,
    metadata: dict[str, Any],
    train_size: int,
    test_size: int,
    model_name_accuracy: str,
    model_name_recall: str,
    threshold_accuracy: float,
    threshold_recall: float,
    checks: dict[str, bool],
) -> Path:
    report = {
        "fecha_ejecucion": pd.Timestamp.now().isoformat(),
        "versions": {
            "python": sys.version.split()[0],
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "configuracion_experimento": {
            "random_state": cfg.random_state,
            "test_size": cfg.phase2_test_size,
            "n_splits_cv": cfg.phase2_n_splits,
            "phase3_recall_min_precision": cfg.phase3_recall_min_precision,
            "phase3_recall_min_prauc": cfg.phase3_recall_min_prauc,
        },
        "datos_base": {
            "metadata": metadata,
            "split_train_val_test": {"train_val": int(train_size), "test": int(test_size)},
        },
        "modelos_entrenados": {
            "model_accuracy": model_name_accuracy,
            "model_recall": model_name_recall,
            "threshold_accuracy": float(threshold_accuracy),
            "threshold_recall": float(threshold_recall),
        },
        "checks": checks,
    }

    path = cfg.data_processed_dir / "fase3_reproducibility_report.json"
    cfg.data_processed_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path

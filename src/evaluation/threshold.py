from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics_at_threshold


@dataclass(frozen=True)
class ThresholdPolicy:
    objective: str
    min_precision: float | None = None
    min_prauc: float | None = None


@dataclass(frozen=True)
class ThresholdSelectionResult:
    best_threshold: float
    threshold_table: pd.DataFrame


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    policy: ThresholdPolicy,
    thresholds: np.ndarray,
) -> ThresholdSelectionResult:
    rows: list[dict[str, float | list[int]]] = []
    for threshold in thresholds:
        rows.append(compute_metrics_at_threshold(y_true, y_prob, float(threshold)))

    table = pd.DataFrame(rows)
    if policy.objective == "accuracy":
        idx = table["accuracy"].idxmax()
        return ThresholdSelectionResult(best_threshold=float(table.loc[idx, "threshold"]), threshold_table=table)

    if policy.objective == "recall":
        candidates = table.copy()
        if policy.min_precision is not None:
            candidates = candidates[candidates["precision"] >= policy.min_precision]
        if policy.min_prauc is not None:
            candidates = candidates[candidates["pr_auc"] >= policy.min_prauc]
        if candidates.empty:
            candidates = table
        idx = candidates["recall"].idxmax()
        return ThresholdSelectionResult(best_threshold=float(candidates.loc[idx, "threshold"]), threshold_table=table)

    raise ValueError("policy.objective debe ser 'accuracy' o 'recall'")

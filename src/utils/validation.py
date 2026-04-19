from __future__ import annotations

from pathlib import Path
from typing import Any


def validate_no_leakage_phase3(
    *,
    train_idx: Any,
    test_idx: Any,
    threshold_acc: float,
    threshold_rec: float,
    artifacts_to_check: list[Path],
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["train_test_disjoint"] = len(set(train_idx).intersection(set(test_idx))) == 0
    checks["threshold_acc_valid"] = 0.0 <= float(threshold_acc) <= 1.0
    checks["threshold_rec_valid"] = 0.0 <= float(threshold_rec) <= 1.0
    checks["artifacts_exist"] = all(path.exists() for path in artifacts_to_check)
    return checks

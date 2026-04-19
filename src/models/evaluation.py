from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.evaluation.metrics import compute_metrics_at_threshold


def evaluate_model_on_test(
    *,
    model: Any,
    X_test: Any,
    y_test: Any,
    threshold: float,
    model_label: str,
) -> Dict[str, Any]:
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics_at_threshold(np.asarray(y_test), y_prob, threshold)
    metrics["model"] = model_label
    return metrics

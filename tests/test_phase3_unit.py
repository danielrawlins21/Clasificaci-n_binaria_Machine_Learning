from __future__ import annotations

import numpy as np

from src.evaluation.guardrails import validate_guardrails_recall
from src.evaluation.threshold import ThresholdPolicy, select_threshold


def test_select_threshold_accuracy() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.7, 0.8, 0.4, 0.9])

    result = select_threshold(
        y_true=y_true,
        y_prob=y_prob,
        policy=ThresholdPolicy(objective="accuracy"),
        thresholds=np.array([0.3, 0.5, 0.7]),
    )

    assert 0.0 <= result.best_threshold <= 1.0
    assert not result.threshold_table.empty


def test_select_threshold_recall_guardrails() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])

    result = select_threshold(
        y_true=y_true,
        y_prob=y_prob,
        policy=ThresholdPolicy(objective="recall", min_precision=0.3, min_prauc=0.3),
        thresholds=np.array([0.1, 0.3, 0.5]),
    )

    assert 0.0 <= result.best_threshold <= 1.0


def test_validate_guardrails_recall() -> None:
    status = validate_guardrails_recall(
        precision=0.31,
        pr_auc=0.40,
        min_precision=0.30,
        min_prauc=0.35,
    )
    assert status.is_valid

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import create_external_split
from src.features.selection import apply_candidate_feature, decide_feature_candidate


def test_create_external_split_is_reproducible_and_disjoint() -> None:
    y = pd.Series(([0] * 80) + ([1] * 20))
    s1 = create_external_split(y, test_size=0.2, random_state=42)
    s2 = create_external_split(y, test_size=0.2, random_state=42)

    assert s1.equals(s2)
    assert (s1["set"] == "test").sum() == 20
    assert (s1["set"] == "train_val").sum() == 80

    test_idx = set(s1.loc[s1["set"] == "test", "row_idx"].tolist())
    train_idx = set(s1.loc[s1["set"] == "train_val", "row_idx"].tolist())
    assert test_idx.isdisjoint(train_idx)
    assert len(test_idx | train_idx) == len(y)


def test_apply_candidate_minutes_per_call_handles_zero_division() -> None:
    X = pd.DataFrame(
        {
            "V8": [1.0, 2.0],
            "V11": [1.0, 2.0],
            "V14": [1.0, 2.0],
            "V17": [1.0, 2.0],
            "V9": [0.0, 1.0],
            "V12": [0.0, 1.0],
            "V15": [0.0, 1.0],
            "V18": [0.0, 1.0],
            "V2": [2.0, 4.0],
            "V20": [1.0, 2.0],
            "V1": [1, 2],
            "V3": [100, 200],
            "V5": [0, 1],
            "V6": [1, 0],
            "V7": [1, 2],
        }
    )
    out = apply_candidate_feature(X, "minutes_per_call")
    assert "minutes_per_call" in out.columns
    assert np.isnan(out.loc[0, "minutes_per_call"])
    assert out.loc[1, "minutes_per_call"] == 2.0


def test_decide_feature_candidate_rejects_for_instability() -> None:
    row = pd.Series(
        {
            "objective": "accuracy",
            "acc_base_mean": 0.80,
            "acc_ext_mean": 0.82,
            "recall_base_mean": 0.40,
            "recall_ext_mean": 0.39,
            "precision_base_mean": 0.50,
            "precision_ext_mean": 0.50,
            "prauc_base_mean": 0.55,
            "prauc_ext_mean": 0.55,
            "primary_std_base": 0.01,
            "primary_std_ext": 0.05,
        }
    )
    decision = decide_feature_candidate(
        row,
        recall_tolerance_acc=0.02,
        precision_tolerance_rec=0.02,
        prauc_tolerance_rec=0.02,
        std_increase_limit=0.20,
    )
    assert decision["decision"] == "rechazar"
    assert "inestabilidad" in decision["reason"]

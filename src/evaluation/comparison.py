from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import ProjectConfig
from src.features.selection import decide_feature_candidate


def build_decisions_and_summaries(
    metrics_df: pd.DataFrame,
    *,
    recall_tolerance_acc: float,
    precision_tolerance_rec: float,
    prauc_tolerance_rec: float,
    std_increase_limit: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decisions_df = metrics_df.join(
        metrics_df.apply(
            decide_feature_candidate,
            axis=1,
            recall_tolerance_acc=recall_tolerance_acc,
            precision_tolerance_rec=precision_tolerance_rec,
            prauc_tolerance_rec=prauc_tolerance_rec,
            std_increase_limit=std_increase_limit,
        )
    )

    summary_counts = (
        decisions_df.groupby(["objective", "decision"])["candidate"].count().rename("n_candidates").reset_index()
    )

    summary_delta = decisions_df.assign(
        delta_acc=lambda d: d["acc_ext_mean"] - d["acc_base_mean"],
        delta_recall=lambda d: d["recall_ext_mean"] - d["recall_base_mean"],
        delta_precision=lambda d: d["precision_ext_mean"] - d["precision_base_mean"],
        delta_prauc=lambda d: d["prauc_ext_mean"] - d["prauc_base_mean"],
    )[
        [
            "candidate",
            "objective",
            "decision",
            "delta_acc",
            "delta_recall",
            "delta_precision",
            "delta_prauc",
        ]
    ]

    return decisions_df, summary_counts, summary_delta


def write_action_plan_status(outputs_tables_dir: Path, cfg: ProjectConfig) -> pd.DataFrame:
    action_plan = pd.DataFrame(
        [{"step": step, "owner": "script", "status": "pendiente"} for step in cfg.phase2_action_plan_steps]
    )
    status_map = {
        "Crear split externo estratificado y guardarlo": (outputs_tables_dir / "fase2_split_indices.csv").exists(),
        "Guardar configuracion formal del split/CV": (outputs_tables_dir / "fase2_split_config.json").exists(),
        "Calcular metricas CV base vs extendido por candidata": (outputs_tables_dir / "fase2_feature_eval_metrics.csv").exists(),
        "Aplicar regla de decision y exportar aceptar/rechazar": (outputs_tables_dir / "fase2_feature_eval_decisions.csv").exists(),
        "Generar resumen final para memoria": (outputs_tables_dir / "fase2_feature_eval_summary_counts.csv").exists(),
    }
    action_plan["status"] = action_plan["step"].map(lambda s: "hecho" if status_map.get(s, False) else "pendiente")
    action_plan.to_csv(outputs_tables_dir / "fase2_action_plan_status.csv", index=False)
    return action_plan

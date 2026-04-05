from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_METRIC_COLUMNS: tuple[str, ...] = (
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
)


def feature_candidates_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": "total_minutes",
                "definition": "V8 + V11 + V14 + V17",
                "motivation": "Resume la intensidad global de uso sin depender de la franja horaria.",
            },
            {
                "feature": "total_calls",
                "definition": "V9 + V12 + V15 + V18",
                "motivation": "Agrega el volumen total de contactos por franjas.",
            },
            {
                "feature": "minutes_per_call",
                "definition": "total_minutes / total_calls",
                "motivation": "Captura la intensidad media de cada contacto.",
            },
            {
                "feature": "international_share_minutes",
                "definition": "V17 / total_minutes",
                "motivation": "Mide la proporción de uso internacional dentro del total.",
            },
            {
                "feature": "log1p_v2",
                "definition": "log(1 + V2)",
                "motivation": "Reduce asimetría y estabiliza el efecto de la antigüedad.",
            },
            {
                "feature": "log1p_total_minutes",
                "definition": "log(1 + total_minutes)",
                "motivation": "Suaviza colas largas en una variable de uso agregada.",
            },
            {
                "feature": "customer_service_intensity",
                "definition": "V20",
                "motivation": "Mantiene la señal de insatisfacción observable en el número de llamadas al servicio.",
            },
        ]
    )


def build_feature_eval_template(candidates: Iterable[str]) -> pd.DataFrame:
    template = pd.MultiIndex.from_product(
        [list(candidates), ["accuracy", "recall"]],
        names=["candidate", "objective"],
    ).to_frame(index=False)
    cols = ["candidate", "objective", *REQUIRED_METRIC_COLUMNS]
    for col in cols:
        if col not in template.columns:
            template[col] = np.nan
    return template[cols]


def apply_candidate_feature(df_features: pd.DataFrame, candidate_name: str) -> pd.DataFrame:
    df = df_features.copy()
    total_minutes = df["V8"] + df["V11"] + df["V14"] + df["V17"]
    total_calls = df["V9"] + df["V12"] + df["V15"] + df["V18"]

    if candidate_name == "total_minutes":
        df["total_minutes"] = total_minutes
    elif candidate_name == "total_calls":
        df["total_calls"] = total_calls
    elif candidate_name == "minutes_per_call":
        df["minutes_per_call"] = np.where(total_calls > 0, total_minutes / total_calls, np.nan)
    elif candidate_name == "international_share_minutes":
        df["international_share_minutes"] = np.where(total_minutes > 0, df["V17"] / total_minutes, 0.0)
    elif candidate_name == "log1p_v2":
        df["log1p_v2"] = np.log1p(df["V2"].clip(lower=0))
    elif candidate_name == "log1p_total_minutes":
        df["log1p_total_minutes"] = np.log1p(total_minutes.clip(lower=0))
    elif candidate_name == "customer_service_intensity":
        df["customer_service_intensity"] = df["V20"]
    else:
        raise ValueError(f"Candidata no soportada: {candidate_name}")

    return df


def decide_feature_candidate(
    row: pd.Series,
    *,
    recall_tolerance_acc: float,
    precision_tolerance_rec: float,
    prauc_tolerance_rec: float,
    std_increase_limit: float,
) -> pd.Series:
    std_ratio = (row["primary_std_ext"] / row["primary_std_base"]) if row["primary_std_base"] > 0 else np.inf
    unstable = std_ratio > (1 + std_increase_limit)

    if row["objective"] == "accuracy":
        primary_ok = row["acc_ext_mean"] > row["acc_base_mean"]
        guardrail_ok = (row["recall_base_mean"] - row["recall_ext_mean"]) <= recall_tolerance_acc
    elif row["objective"] == "recall":
        primary_ok = row["recall_ext_mean"] > row["recall_base_mean"]
        guardrail_ok = (
            (row["precision_base_mean"] - row["precision_ext_mean"]) <= precision_tolerance_rec
            and (row["prauc_base_mean"] - row["prauc_ext_mean"]) <= prauc_tolerance_rec
        )
    else:
        return pd.Series({"decision": "pendiente", "reason": "objective debe ser accuracy o recall"})

    if primary_ok and guardrail_ok and not unstable:
        return pd.Series(
            {
                "decision": "aceptar",
                "reason": "mejora primaria, respeta guardrails y mantiene estabilidad",
            }
        )

    reasons: list[str] = []
    if not primary_ok:
        reasons.append("no mejora metrica primaria")
    if not guardrail_ok:
        reasons.append("viola guardrails secundarios")
    if unstable:
        reasons.append("incrementa inestabilidad entre pliegues")
    return pd.Series({"decision": "rechazar", "reason": "; ".join(reasons)})

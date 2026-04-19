from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailStatus:
    is_valid: bool
    violations: tuple[str, ...]


def validate_guardrails_recall(
    *,
    precision: float,
    pr_auc: float,
    min_precision: float,
    min_prauc: float,
) -> GuardrailStatus:
    violations: list[str] = []
    if precision < min_precision:
        violations.append(f"precision<{min_precision}")
    if pr_auc < min_prauc:
        violations.append(f"pr_auc<{min_prauc}")
    return GuardrailStatus(is_valid=not violations, violations=tuple(violations))

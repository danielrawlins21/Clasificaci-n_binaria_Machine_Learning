from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class ProjectConfig:
    """Configuracion centralizada del proyecto."""

    project_root: Path
    random_state: int = 42
    target_col: str = "Y"

    # Variables segun conocimiento de dominio del enunciado.
    id_columns: Tuple[str, ...] = ("V4",)
    nominal_columns: Tuple[str, ...] = ("V1", "V3")
    binary_columns: Tuple[str, ...] = ("V5", "V6")
    discrete_columns: Tuple[str, ...] = ("V2", "V7", "V9", "V12", "V15", "V18", "V20")
    continuous_columns: Tuple[str, ...] = ("V8", "V10", "V11", "V13", "V14", "V16", "V17", "V19")

    # Pares con redundancia alta (minutos, coste). Se mantiene minutos y se descarta coste.
    redundant_pairs: Tuple[Tuple[str, str], ...] = (
        ("V8", "V10"),
        ("V11", "V13"),
        ("V14", "V16"),
        ("V17", "V19"),
    )

    missing_tokens: Tuple[str, ...] = ("NA",)
    expected_columns: Tuple[str, ...] = tuple([f"V{i}" for i in range(1, 21)] + ["Y"])

    @property
    def data_raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def data_file(self) -> Path:
        return self.data_raw_dir / "BBDD_ML_TAREA.csv"

    @property
    def outputs_tables_dir(self) -> Path:
        return self.project_root / "outputs" / "tables"

    @property
    def columns_to_drop_by_design(self) -> List[str]:
        to_drop = list(self.id_columns)
        to_drop.extend(cost for _, cost in self.redundant_pairs)
        return sorted(set(to_drop))


def get_project_root_from_file(current_file: Path) -> Path:
    """Obtiene la raiz del proyecto subiendo dos niveles desde src/*.py."""
    return current_file.resolve().parents[1]


def get_default_config() -> ProjectConfig:
    project_root = get_project_root_from_file(Path(__file__))
    return ProjectConfig(project_root=project_root)

"""
Módulo de configuración centralizada del proyecto.

Este módulo define un único punto de verdad para:
- Rutas de archivos y directorios del proyecto
- Metadatos del dataset (columnas esperadas, tipos de variables)
- Criterios de depuración de datos (identificadores, redundancias)
- Parámetros de reproducibilidad (random_state)

Diseño de configuración:
- **Inmutable**: Los parámetros son inmutables (frozen=True) para evitar cambios accidentales
- **Centralizado**: Evita hardcoding de paths y nombres de columnas en múltiples archivos
- **Programático**: Propiedades derivadas se calculan dinámicamente (@property)
- **Debuggable**: ProjectConfig.columns_to_drop_by_design auto-resuelve todas las reglas

Uso típico:
    >>> cfg = get_default_config()
    >>> data_file = cfg.data_file  # Path completa a CSV
    >>> x, y = split_features_target(df_base, config=cfg)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class ProjectConfig:
    """
    Configuración centralizada e inmutable del proyecto.
    
    Contiene:
    - Metadatos obligatorios (raíz del proyecto, random_state)
    - Taxonomía de variables según dominio del problema
    - Criterios de depuración de datos (identificadores, redundancias)
    - Rutas derivadas dinámicamente (@property)
    
    Atributos de inicialización:
        project_root: Ruta raíz del proyecto (calculada auto en get_default_config)
        random_state: Semilla para reproducibilidad (defecto: 42)
        target_col: Nombre de variable objetivo (defecto: "Y")
    
    Taxonomía de variables (conocimiento de dominio):
        id_columns: Identificadores (V4 - número de teléfono)
        nominal_columns: Categóricas sin orden (V1, V3)
        binary_columns: Booleanas/binarias (V5, V6)
        discrete_columns: Numéricas contables (V2, V7, V9, V12, V15, V18, V20)
        continuous_columns: Numéricas continuas (V8, V10, V11, V13, V14, V16, V17, V19)
    
    Criterio de redundancia:
        redundant_pairs: Tupla de pares (minutos, coste) con r > 0.999.
                        Se mantiene minutos, se descarta coste para cada par.
    
    Tokens especiales:
        missing_tokens: Símbolos que representan valores faltantes en CSV ("NA")
        expected_columns: Lista de columnas esperadas en dataset bruto (V1-V20, Y)
    
    Note:
        - Frozen=True: Imposible modificar atributos después de instanciación
        - Todas las propiedades (@property) son derivadas y calculadas on-demand
        - Mutabilidad cero garantiza consistencia en todo el pipeline
    
    Example:
        >>> cfg = ProjectConfig(project_root=Path('/project'))
        >>> cfg.data_file
        Path('/project/data/raw/BBDD_ML_TAREA.csv')
        >>> cfg.columns_to_drop_by_design
        ['V10', 'V13', 'V16', 'V19', 'V4']  # Sorted: id + costes
    """

    project_root: Path
    random_state: int = 42
    target_col: str = "Y"
    phase2_test_size: float = 0.20
    phase2_n_splits: int = 5
    phase2_recall_tolerance_acc: float = 0.02
    phase2_precision_tolerance_rec: float = 0.02
    phase2_prauc_tolerance_rec: float = 0.02
    phase2_std_increase_limit: float = 0.20
    phase3_accuracy_threshold_grid: Tuple[float, ...] = tuple(round(0.10 + i * 0.01, 2) for i in range(81))
    phase3_recall_threshold_grid: Tuple[float, ...] = tuple(round(0.05 + i * 0.01, 2) for i in range(66))
    phase3_recall_min_precision: float = 0.28
    phase3_recall_min_prauc: float = 0.35
    phase3_rf_n_iter: int = 24
    phase3_et_n_iter: int = 24

    # Variables según conocimiento de dominio del enunciado
    id_columns: Tuple[str, ...] = ("V4",)
    nominal_columns: Tuple[str, ...] = ("V1", "V3")
    binary_columns: Tuple[str, ...] = ("V5", "V6")
    discrete_columns: Tuple[str, ...] = ("V2", "V7", "V9", "V12", "V15", "V18", "V20")
    continuous_columns: Tuple[str, ...] = ("V8", "V10", "V11", "V13", "V14", "V16", "V17", "V19")

    # Pares con redundancia alta (minutos vs. coste): r > 0.999 → colinealidad perfecta
    # Decisión: se mantiene minutos (interpretable), se descarta coste (redundante)
    redundant_pairs: Tuple[Tuple[str, str], ...] = (
        ("V8", "V10"),   # V8=minutos intl, V10=coste intl
        ("V11", "V13"),  # V11=minutos roaming, V13=coste roaming
        ("V14", "V16"),  # V14=minutos nocturnos, V16=coste nocturno
        ("V17", "V19"),  # V17=minutos especiales, V19=coste especial
    )

    missing_tokens: Tuple[str, ...] = ("NA",)
    expected_columns: Tuple[str, ...] = tuple([f"V{i}" for i in range(1, 21)] + ["Y"])
    phase2_artifact_filenames: Tuple[str, ...] = (
        "fase2_split_indices.csv",
        "fase2_split_config.json",
        "fase2_feature_candidates.csv",
        "fase2_feature_eval_template.csv",
        "fase2_feature_eval_metrics.csv",
        "fase2_feature_eval_decisions.csv",
        "fase2_feature_eval_summary_counts.csv",
        "fase2_feature_eval_summary_delta.csv",
        "fase2_action_plan_status.csv",
    )
    phase2_action_plan_steps: Tuple[str, ...] = (
        "Crear split externo estratificado y guardarlo",
        "Guardar configuracion formal del split/CV",
        "Calcular metricas CV base vs extendido por candidata",
        "Aplicar regla de decision y exportar aceptar/rechazar",
        "Generar resumen final para memoria",
    )
    phase3_artifact_filenames: Tuple[str, ...] = (
        "fase3_threshold_search_accuracy.csv",
        "fase3_threshold_search_recall.csv",
        "fase3_rfe_tree_mlp_comparison.csv",
        "fase3_rfe_selected_features.csv",
        "fase3_tree_selected_features.csv",
    )

    @property
    def data_raw_dir(self) -> Path:
        """
        Ruta del directorio de datos brutos.
        
        Returns:
            Path: {project_root}/data/raw/
                  Contiene el CSV original sin procesamiento (BBDD_ML_TAREA.csv)
        """
        return self.project_root / "data" / "raw"

    @property
    def data_file(self) -> Path:
        """
        Ruta completa al archivo de dataset bruto.
        
        Returns:
            Path: {project_root}/data/raw/BBDD_ML_TAREA.csv
                  Archivo CSV con 9200 filas x 21 columnas (V1-V20, Y)
        
        Nota:
            Esta ruta es utilizada por load_data.read_raw_dataset().
            Si el archivo no existe, raises FileNotFoundError.
        """
        return self.data_raw_dir / "BBDD_ML_TAREA.csv"

    @property
    def data_interim_dir(self) -> Path:
        return self.project_root / "data" / "interim"

    @property
    def data_processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def outputs_tables_dir(self) -> Path:
        """
        Ruta del directorio de salida para tablas y reportes.
        
        Returns:
            Path: {project_root}/outputs/tables/
                  Destino para archivos generados (CSVs, métricas, resultados)
        
        Uso típico:
            - fase2_split_indices.csv
            - fase2_split_config.json
            - fase2_feature_eval_template.csv
            - fase2_feature_eval_decisions.csv
        """
        return self.project_root / "outputs" / "tables"

    @property
    def columns_to_drop_by_design(self) -> List[str]:
        """
        Lista de columnas a excluir del modelado según decisiones metodológicas.
        
        Combina automáticamente:
        1. **Identificadores**: V4 (conocido como número telefónico)
        2. **Redundancias**: Segundo elemento de cada par (costes):
           - V10 (coste internacional, redundante con V8=minutos internacionales)
           - V13 (coste roaming, redundante con V11=minutos roaming)
           - V16 (coste nocturno, redundante con V14=minutos nocturnos)
           - V19 (coste especial, redundante con V17=minutos especiales)
        
        Returns:
            List[str]: Columnas a descartar, ordenadas alfabéticamente.
                       Típicamente: ['V10', 'V13', 'V16', 'V19', 'V4']
        
        Rationale:
            - Se eliminan identificadores (V4) que no tienen poder predictivo
            - Entre pares colineales (r > 0.999), se mantiene minutos (interpretable)
              y se descarta coste (derivado, no aporta información nueva)
            - Aplicado en build_model_base_dataset() para construir X,y
        
        Example:
            >>> cfg = get_default_config()
            >>> cfg.columns_to_drop_by_design
            ['V10', 'V13', 'V16', 'V19', 'V4']
        """
        to_drop = list(self.id_columns)
        to_drop.extend(cost for _, cost in self.redundant_pairs)
        return sorted(set(to_drop))


def get_project_root_from_file(current_file: Path) -> Path:
    """
    Deduce la raíz del proyecto a partir de la ruta de un archivo dentro de src/.
    
    Lógica:
    - Resuelve la ruta absoluta del archivo
    - Sube dos niveles de directorios (.parents[1])
      * .parents[0] → src/
      * .parents[1] → project_root/
    
    Args:
        current_file: Path a un archivo dentro del árbol src/
                      (típicamente: Path(__file__) desde src/config.py)
    
    Returns:
        Path: Raíz del proyecto (dos niveles arriba de src/)
    
    Example:
        >>> # Si current_file = /project/src/config.py
        >>> cfg_root = get_project_root_from_file(Path('/project/src/config.py'))
        >>> cfg_root
        Path('/project')
    """
    return current_file.resolve().parents[1]


def get_default_config() -> ProjectConfig:
    """
    Factory function para instanciar la configuración predeterminada del proyecto.
    
    Realiza:
    1. Deduce automáticamente project_root desde la ubicación de este archivo (src/config.py)
    2. Instancia ProjectConfig con todos los parámetros predefinidos
    3. Retorna objeto inmutable listo para usar en todo el proyecto
    
    Returns:
        ProjectConfig: Instancia de configuración del proyecto con:
                       - project_root: Detectada automáticamente
                       - random_state: 42 (para reproducibilidad)
                       - Todas las columnas, taxonomía, y criterios configurados
    
    Note:
        Función determinista: siempre retorna la misma configuración.
        Utilizada por defecto en load_data.py y preprocessing.py.
    
    Example:
        >>> cfg = get_default_config()
        >>> cfg.data_file.exists()
        True
        >>> cfg.outputs_tables_dir
        Path('...Master UCM/.../outputs/tables')
    """
    project_root = get_project_root_from_file(Path(__file__))
    return ProjectConfig(project_root=project_root)

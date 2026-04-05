"""
Módulo de carga y validación de datos.

Este módulo proporciona funciones para:
- Lectura del dataset bruto desde archivo CSV
- Validación de esquema esperado (columnas y tipos)
- Análisis de valores faltantes (missingness)
- Construcción del dataset base para modelado (depuración)
- Separación de características y variable objetivo

Flujo típico de uso:
    1. read_raw_dataset() → cargar datos brutos
    2. validate_expected_columns() → verificar esquema
    3. build_model_base_dataset() → aplicar depuración estructural
    4. split_features_target() → separar X, y para modelado

"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from src.config import ProjectConfig, get_default_config


class DataSchemaError(ValueError):
    """
    Excepción de validación de esquema del dataset.
    
    Se lanza cuando el dataset no coincide con el esquema esperado
    (columnas faltantes, sobrepuestas, o tipos incompatibles).
    """


def read_raw_dataset(config: ProjectConfig | None = None) -> pd.DataFrame:
    """
    Carga el dataset bruto desde archivo CSV.
    
    Realiza la lectura del CSV especificado en la configuración,
    interpretando los tokens de valores faltantes definidos en config.
    
    Args:
        config: Instancia de configuración del proyecto. Si es None,
                utiliza la configuración por defecto (get_default_config()).
    
    Returns:
        pd.DataFrame: Dataset bruto con todas las filas y columnas originales.
                      Los valores faltantes se reemplazan con NaN de pandas.
    
    Raises:
        FileNotFoundError: Si el archivo CSV no existe en la ruta especificada.
        
    Example:
        >>> df_raw = read_raw_dataset()
        >>> df_raw.shape
        (9200, 21)
    """
    cfg = config or get_default_config()
    if not cfg.data_file.exists():
        raise FileNotFoundError(f"No se encontro el dataset en: {cfg.data_file}")
    return pd.read_csv(cfg.data_file, na_values=list(cfg.missing_tokens))


def validate_expected_columns(df: pd.DataFrame, config: ProjectConfig | None = None) -> None:
    """
    Valida que el dataset cumpla con el esquema esperado.
    
    Verifica que:
    - Todas las columnas esperadas (según config) estén presentes
    - No haya columnas adicionales no permitidas
    
    Esta es una validación de contrato: asegura que el dataset 
    está listo para procesamiento posterior sin anomalías de esquema.
    
    Args:
        df: DataFrame a validar.
        config: Instancia de configuración con columnas esperadas.
                Si es None, usa configuración por defecto.
    
    Returns:
        None. No retorna valor; solo valida.
    
    Raises:
        DataSchemaError: Si faltan columnas, hay columnas sobrantes,
                        o el esquema no coincide con lo esperado.
                        Incluye detalles de las columnas problemáticas.
        
    Example:
        >>> validate_expected_columns(df_raw)  # Retorna silenciosamente si OK
        >>> # Si falla: DataSchemaError: Esquema inesperado en dataset...
    """
    cfg = config or get_default_config()
    expected = set(cfg.expected_columns)
    current = set(df.columns)

    missing = sorted(expected - current)
    extra = sorted(current - expected)

    if missing or extra:
        raise DataSchemaError(
            "Esquema inesperado en dataset. "
            f"Faltan={missing if missing else []}; "
            f"Sobrantes={extra if extra else []}"
        )


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen auditoría de valores faltantes por columna.
    
    Calcula para cada variable:
    - Conteo de valores faltantes (NaN)
    - Porcentaje de missingness respecto al total de filas
    - Tipo de dato de la columna
    - Número de valores únicos (sin contar NaN)
    
    Útil para:
    - Identificar columnas con problemas de calidad de datos
    - Decidir estrategias de imputación
    - Documentar la auditoría de datos
    
    Args:
        df: DataFrame del que se calculan estadísticas de missingness.
    
    Returns:
        pd.DataFrame: Tabla con índice = nombre de columna,
                      columnas = [missing_count, missing_pct, dtype, n_unique]
                      ordenada descendentemente por missing_count.
    
    Example:
        >>> missing_table = get_missing_summary(df_raw)
        >>> missing_table.head()
                    missing_count  missing_pct dtype  n_unique
        V3                       75       0.8152   int64       12
        V14                      72       0.7826   int64       45
    """
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(4)
    out = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(dropna=True),
    })
    return out.sort_values(by="missing_count", ascending=False)


def build_model_base_dataset(
    df_raw: pd.DataFrame,
    config: ProjectConfig | None = None,
    drop_duplicates: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Construye el dataset base para modelado aplicando depuración estructural.
    
    Aplica las decisiones metodológicas definidas en el protocolo experimental:
    
    1. Eliminación de duplicados (si drop_duplicates=True)
       - Se realiza ANTES de cualquier otra transformación para evitar
         fuga de información (data leakage) en la posterior estratificación.
    
    2. Exclusión de columnas por diseño
       - Elimina identificadores (ej: V4) y variables redundantes que
         fueron descartadas según criterios especificados en config.
       - Redundancia: se mantienen variables de minutos, se descartan costos
         (correlación > 0.999 con minutos).
    
    Args:
        df_raw: DataFrame bruto con todas las columnas originales.
        config: Configuración del proyecto con criterios de depuración.
                Si es None, usa configuración por defecto.
        drop_duplicates: Si True (por defecto), elimina filas duplicadas
                         basadas en todas las columnas.
    
    Returns:
        Tuple[pd.DataFrame, Dict]:
            - df: Dataset depurado listo para modelado
            - metadata: Diccionario con historial de transformaciones:
                * n_rows_raw: Número de filas originales
                * n_rows_model_base: Filas después de depuración
                * duplicates_removed: Cantidad de duplicados eliminados
                * dropped_columns: Lista de columnas excluidas
    
    Note:
        Los duplicados se identifican considerando TODAS las columnas.
        Las transformaciones se aplican en este orden: dedup → drop columns.
        
    Example:
        >>> df_base, meta = build_model_base_dataset(df_raw)
        >>> df_base.shape
        (3538, 16)
        >>> meta['duplicates_removed']
        5662
    """
    cfg = config or get_default_config()
    df = df_raw.copy()

    n_rows_raw = len(df)
    duplicates_removed = 0
    if drop_duplicates:
        duplicates_removed = int(df.duplicated().sum())
        df = df.drop_duplicates().copy()

    cols_to_drop = [c for c in cfg.columns_to_drop_by_design if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    metadata = {
        "n_rows_raw": n_rows_raw,
        "n_rows_model_base": len(df),
        "duplicates_removed": duplicates_removed,
        "dropped_columns": cols_to_drop,
    }
    return df, metadata


def split_features_target(
    df: pd.DataFrame,
    config: ProjectConfig | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa el dataset en matriz de características (X) y variable objetivo (y).
    
    Particiona el DataFrame en:
    - X: Todas las variables excepto la objetivo (características/predictores)
    - y: Variable objetivo (Y) para predicción de churn
    
    Uso típico en el flujo de modelado (después de build_model_base_dataset):
    
    Args:
        df: Dataset base ya depurado (salida de build_model_base_dataset).
        config: Configuración del proyecto con nombre de variable objetivo.
                Si es None, usa configuración por defecto.
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            - X: DataFrame con características (shape[1] = número de predictores)
            - y: Series con variable objetivo (valores 0 o 1 para clasificación)
    
    Raises:
        DataSchemaError: Si la variable objetivo no existe en df.columns.
        
    Example:
        >>> X, y = split_features_target(df_base)
        >>> X.shape
        (3538, 15)
        >>> y.value_counts()
        0    2827
        1     711
        Name: Y, dtype: int64
    """
    cfg = config or get_default_config()
    if cfg.target_col not in df.columns:
        raise DataSchemaError(f"No existe la variable objetivo '{cfg.target_col}'")

    X = df.drop(columns=[cfg.target_col])
    y = df[cfg.target_col]
    return X, y

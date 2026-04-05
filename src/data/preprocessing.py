"""
Módulo de preprocesamiento de características.

Constituye el puente entre el dataset base depurado y los modelos de machine learning.
Proporciona pipelines de transformación optimizadas según la familia de modelo:

1. **Mod
delos lineales y MLP**: Con escalado StandardScaler (requerido para convergencia)
2. **Modelos basados en árboles**: Sin escalado (no es necesario, solo codificación)

Características del preprocesamiento:
- Imputación inteligente: moda para nominales, mediana para numéricos
- One-Hot Encoding para variables categóricas/nominales
- Escalado Z-score para modelos sensibles a magnitud
- Manejo robusto de valores desconocidos en producción

Uso típico:
    >>> preprocessor = build_linear_mlp_preprocessor(X_train)
    >>> X_train_transformed = preprocessor.fit_transform(X_train)
    >>> X_test_transformed = preprocessor.transform(X_test)

"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ProjectConfig, get_default_config


def _intersect(existing: List[str], requested: Tuple[str, ...]) -> List[str]:
    """
    Calcula la intersección entre columnas existentes y solicitadas.
    
    Función auxiliar interna que preserva el orden de las columnas existentes
    mientras filtra por las solicitadas. Utilizada para mapear grupos de variables
    (nominal, binary, discrete, continuous) contra las columnas reales del DataFrame.
    
    Args:
        existing: Lista de nombres de columnas presentes en el DataFrame X.
        requested: Tupla de nombres de columnas esperadas (del config).
    
    Returns:
        List[str]: Subconjunto de `existing` que también aparecen en `requested`,
                   manteniendo el orden de `existing`.
    
    Example:
        >>> columns_in_data = ['V1', 'V2', 'V3']
        >>> configured_nominal = ('V1', 'V3', 'V5')  # V5 no en datos
        >>> _intersect(columns_in_data, configured_nominal)
        ['V1', 'V3']
    """
    req = set(requested)
    return [c for c in existing if c in req]


def infer_column_groups(
    X: pd.DataFrame,
    config: ProjectConfig | None = None,
) -> Dict[str, List[str]]:
    """
    Clasifica columnas del dataset en grupos semánticos.
    
    Mapea cada columna presente en X a su grupo de variable según la taxonomía
    definida en la configuración del proyecto: nominal, binaria, discreta, continua.
    
    Esta clasificación es fundamental para decidir:
    - Estrategia de imputación (moda vs mediana)
    - Transformación (OneHotEncoder vs escalado numérico)
    - Inclusión en pipeline
    
    Args:
        X: DataFrame de características (después de split_features_target).
        config: Configuración del proyecto con taxonomía de variables.
                Si es None, usa configuración por defecto.
    
    Returns:
        Dict[str, List[str]]: Diccionario con claves:
            - "nominal": Variables categóricas sin orden (ej: V1, V3)
            - "binary": Variables binarias booleanas (ej: V5, V6)
            - "discrete": Variables numéricas contables (ej: V2, V7)
            - "continuous": Variables numéricas continuas (ej: V8, V11)
            
            Cada lista contiene solo columnas que existen en X, en su orden original.
    
    Note:
        Si una columna del config no existe en X, se omite silenciosamente.
        Esto permite flexibilidad si algunas variables fueron descartadas previamente.
    
    Example:
        >>> X = pd.DataFrame({'V1': [1,2], 'V5': [0,1], 'V8': [1.5, 2.3]})
        >>> groups = infer_column_groups(X)
        >>> groups
        {'nominal': ['V1'], 'binary': ['V5'], 'discrete': [], 'continuous': ['V8']}
    """
    cfg = config or get_default_config()
    cols = X.columns.tolist()
    return {
        "nominal": _intersect(cols, cfg.nominal_columns),
        "binary": _intersect(cols, cfg.binary_columns),
        "discrete": _intersect(cols, cfg.discrete_columns),
        "continuous": _intersect(cols, cfg.continuous_columns),
    }


def build_linear_mlp_preprocessor(
    X: pd.DataFrame,
    config: ProjectConfig | None = None,
) -> ColumnTransformer:
    """
    Construye pipeline de preprocesamiento para modelos lineales y MLP.
    
    Optimizado para modelos sensibles a la escala de características:
    - Regresión Logística (para accuracy/recall con o sin threshold)
    - Multilayer Perceptron/MLP (requiere escalado para convergencia óptima)
    
    Estrategia de transformación:
    
    **Variables nominales**:
    1. Imputación: moda (valor más frecuente)
    2. Codificación: One-Hot Encoding (transforma v categorías → v-1 binarias)
    
    **Variables numéricas (binarias, discretas, continuas)**:
    1. Imputación: mediana (robusta a outliers)
    2. Escalado: StandardScaler Z-score (media=0, std=1)
    
    Args:
        X: DataFrame de características para ajustar grupos de variables.
        config: Configuración del proyecto con taxonomía de columnas.
                Si es None, usa configuración por defecto.
    
    Returns:
        sklearn.compose.ColumnTransformer: Pipeline compilado con:
            - Transformador "nominal" para variables categóricas
            - Transformador "numeric" para variables numéricas
            - remainder="drop" descarta columnas no nombradas
            
            Pronto para usar: .fit(X_train).transform(X_train)
    
    Note:
        El escalado es CRÍTICO para:
        - Convergencia de gradiente en MLP
        - Interpretación de coeficientes en regresión logística
        - Comparabilidad de pesos en redes neuronales
    
    Example:
        >>> from sklearn.model_selection import train_test_split
        >>> X_train, X_test = train_test_split(X, test_size=0.2)
        >>> preprocessor = build_linear_mlp_preprocessor(X_train)
        >>> X_train_scaled = preprocessor.fit_transform(X_train)
        >>> X_test_scaled = preprocessor.transform(X_test)  # Sin refit
        >>> X_train_scaled.shape
        (2825, 18)  # Después de One-Hot Encoding
    """
    groups = infer_column_groups(X, config)

    nominal_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_cols = groups["binary"] + groups["discrete"] + groups["continuous"]
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("nominal", nominal_pipe, groups["nominal"]),
            ("numeric", numeric_pipe, numeric_cols),
        ],
        remainder="drop",
    )


def build_tree_preprocessor(
    X: pd.DataFrame,
    config: ProjectConfig | None = None,
) -> ColumnTransformer:
    """
    Construye pipeline de preprocesamiento para modelos basados en árboles.
    
    Optimizado para modelos invariantes a escala de características:
    - Decision Trees (nodos dividen por umbrales, no magnitud)
    - Random Forests (ensembles de árboles)
    - Gradient Boosting (XGBoost, LightGBM)
    
    Diferencia clave vs build_linear_mlp_preprocessor:
    - **NO escalado StandardScaler**: Los árboles toman decisiones por comparación,
      no se benefician del escalado y consume menos memoria.
    - **Imputación idéntica**: moda (nominal) y mediana (numéricos)
    - **One-Hot Encoding**: Requerido para árboles sklearn nativos
    
    Estrategia de transformación:
    
    **Variables nominales**:
    1. Imputación: moda (valor más frecuente)
    2. Codificación: One-Hot Encoding (v categorías → v-1 binarias)
    
    **Variables numéricas (binarias, discretas, continuas)**:
    1. Imputación: mediana (robusta a outliers)
    2. SIN escalado (se mantiene escala original)
    
    Args:
        X: DataFrame de características para ajustar grupos de variables.
        config: Configuración del proyecto con taxonomía de columnas.
                Si es None, usa configuración por defecto.
    
    Returns:
        sklearn.compose.ColumnTransformer: Pipeline compilado con:
            - Transformador "nominal" para variables categóricas (con OneHot)
            - Transformador "numeric" para variables numéricas (sin escalado)
            - remainder="drop" descarta columnas no nombradas
            
            Pronto para usar: .fit(X_train).transform(X_train)
    
    Note:
        Aunque aparentemente idéntico a build_linear_mlp_preprocessor,
        la AUSENCIA de StandardScaler permite:
        - Interpretación más directa de umbrales en nodos
        - Menor consumo de memoria en ColinearTransformer
        - Consistencia con librerías XGBoost que aceptan sparse input
    
    Example:
        >>> X_train, X_test = train_test_split(X, test_size=0.2)
        >>> preprocessor = build_tree_preprocessor(X_train)
        >>> X_train_encoded = preprocessor.fit_transform(X_train)
        >>> X_test_encoded = preprocessor.transform(X_test)  # Sin refit
        >>> X_train_encoded.shape
        (2825, 18)  # Mismo shape que linear/MLP pero sin escalado
    """
    groups = infer_column_groups(X, config)

    nominal_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_cols = groups["binary"] + groups["discrete"] + groups["continuous"]
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("nominal", nominal_pipe, groups["nominal"]),
            ("numeric", numeric_pipe, numeric_cols),
        ],
        remainder="drop",
    )

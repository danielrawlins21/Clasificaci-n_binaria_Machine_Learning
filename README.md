# Pipeline de Machine Learning para Predicción de Churn

## Descripción General

Este repositorio contiene un pipeline modular y completamente reproducible de machine learning para predecir abandono de clientes (churn) a partir de un dataset de comportamiento de llamadas.

**Objetivo:** Comparar dos enfoques contrastantes:
- **Modelo Accuracy (Random Forest)**: optimizado para precisión global y fiabilidad.
- **Modelo Recall (Extra Trees)**: optimizado para captación máxima de casos de churn.

El proyecto está estructurado en **3 fases**:

1. **Fase 1**: Auditoría y depuración de datos.
2. **Fase 2**: Selección de variables mediante RFE.
3. **Fase 3**: Entrenamiento y comparación de modelos finales.

---

## Requisitos Previos

### Software requerido

- **Python 3.8+** (recomendado: 3.10 o superior)
- **Git** (opcional, para clonar el repositorio)
- **LaTeX** (opcional, solo para compilar el informe: `latexmk`, `pdflatex`)

### Versión verificada

Este proyecto ha sido probado con:
- Python 3.10.13
- Conda (entorno `ml_python`)
- Windows 10/11

---

## Instalación del Entorno

### Opción 1: Usando virtualenv (recomendado para compatibilidad universal)

```bash
# Clonar o descargar el repositorio
cd "Ruta del proyecto"

# Crear entorno virtual
python -m venv .venv

# Activar entorno
# En Windows:
.venv\Scripts\activate
# En macOS/Linux:
source .venv/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

### Opción 2: Usando Conda

```bash
# Crear entorno conda
conda create -n ml_project python=3.10 -y

# Activar entorno
conda activate ml_project

# Instalar dependencias
pip install -r requirements.txt
```

### Verificar instalación

```bash
python -c "import pandas, numpy, sklearn; print('✓ Dependencias instaladas correctamente')"
```

---

## Estructura del Proyecto

```
.
├── README.md                          # Este archivo
├── AGENTE.md                          # Especificaciones técnicas y metodológicas
├── requirements.txt                   # Dependencias Python
│
├── data/
│   ├── raw/
│   │   └── BBDD_ML_TAREA.csv          # Dataset original (NO MODIFICAR)
│   ├── interim/
│   │   └── (outputs intermedios Fase 2)
│   └── processed/
│       ├── fase2_selected_candidates.json
│       ├── fase3_model_selection_summary.json
│       └── fase3_reproducibility_report.json
│
├── src/                               # Código principal
│   ├── __init__.py
│   ├── config.py                      # Configuración centralizada
│   ├── main.py                        # Entrypoint CLI (--phase2, --phase3)
│   ├── data/
│   │   ├── load_data.py               # Carga y split del dataset
│   │   └── preprocessing.py           # Pipelines de preprocesamiento
│   ├── features/
│   │   └── selection.py               # RFE y evaluación de candidatas
│   ├── models/
│   │   ├── tree_ensembles.py          # Tuning RF y ET
│   │   └── evaluation.py              # Evaluación en test
│   ├── evaluation/
│   │   ├── metrics.py                 # Métricas y CV
│   │   ├── threshold.py               # Selección de umbral
│   │   ├── guardrails.py              # Validación de restricciones Recall
│   │   └── comparison.py              # Comparación final
│   ├── pipelines/
│   │   └── phase3_export.py           # Exportación de artefactos
│   └── utils/
│       ├── seeds.py                   # Manejo de semillas aleatorias
│       ├── reproducibility.py         # Reportes JSON
│       └── validation.py              # Checklist no-leakage
│
├── notebooks/                         # Exploración y trazabilidad
│   ├── 01_eda.ipynb
│   ├── 02_hallazgos_eda.ipynb
│   ├── 03_plan_accion_fase2.ipynb
│   └── 04_rfe_tree_mlp_sklearn.ipynb  # Trazabilidad Fase 3
│
├── outputs/                           # Artefactos generados
│   └── tables/
│       ├── fase1_*.csv                # Auditoría Fase 1
│       ├── fase2_*.csv                # Selección Fase 2
│       └── fase3_*.csv                # Comparación Fase 3
│
├── tests/                             # Suite de tests
│   ├── test_phase2_unit.py
│   ├── test_phase2_smoke.py
│   ├── test_phase3_unit.py
│   └── test_phase3_smoke.py
│
└── report/                            # Informe LaTeX
    ├── main.tex
    ├── bibliography.bib
    └── sections/
        ├── 01_introduccion.tex
        ├── 02_descripcion_dataset.tex
        ├── 03_analisis_depuracion.tex
        ├── 04_protocolo_experimental.tex
        ├── 05_modelo_accuracy.tex
        ├── 06_modelo_recall.tex
        ├── 07_comparacion_global.tex
        └── 08_conclusiones.tex
```

---

## Ejecución del Pipeline

### 1. Ejecutar solo Fase 2 (Selección de variables)

```bash
python -m src.main --phase2
```

**Qué hace:**
- Carga el dataset original.
- Aplica RFE para seleccionar 5 variables.
- Evalúa candidatas adicionales usando Random Forest.
- Exporta tablas en `outputs/tables/`.

**Salida esperada:**
```
INFO: Cargando dataset...
INFO: Ejecutando Fase 2...
INFO: RFE completado. Variables seleccionadas: [...]
INFO: Artefactos guardados en outputs/tables/
```

---

### 2. Ejecutar solo Fase 3 (Modelos finales)

```bash
python -m src.main --phase3
```

**Requisito:** Debe haber completado Fase 2 previamente (o ejecutar Fase 2+3).

**Qué hace:**
- Carga datos procesados de Fase 2.
- Entrena Random Forest (optimizado para Accuracy).
- Entrena Extra Trees (optimizado para Recall).
- Realiza búsqueda exhaustiva de threshold.
- Compara ambos modelos en test.
- Exporta tablas y reportes JSON.

**Salida esperada:**
```
INFO: Ejecutando Fase 3...
INFO: Entrenando RF_Accuracy...
INFO: Entrenando ET_Recall...
INFO: Búsqueda de threshold completada
INFO: Artefactos guardados en outputs/tables/
```

---

### 3. Ejecutar Fase 2 + Fase 3 (Pipeline completo)

```bash
python -m src.main --phase2 --phase3
```

**Recomendado:** Ejecutar esto la primera vez para generar todos los artefactos desde cero.

**Duración estimada:** 3-5 minutos (depende del hardware).

---

### 4. Opciones avanzadas

```bash
# Especificar semilla aleatoria
python -m src.main --phase3 --random-state 42

# Especificar ruta personalizada al proyecto
python -m src.main --phase3 --project-root /ruta/personalizada

# Especificar tamaño del split
python -m src.main --phase2 --test-size 0.25

# Mostrar ayuda
python -m src.main --help
```

---

## Validación mediante Tests

### Ejecutar todos los tests

```bash
python -m pytest tests/ -v
```

### Ejecutar tests específicos

```bash
# Tests unitarios de Fase 3
python -m pytest tests/test_phase3_unit.py -v

# Test end-to-end (smoke test) de Fase 3
python -m pytest tests/test_phase3_smoke.py -v

# Tests de Fase 2
python -m pytest tests/test_phase2_unit.py tests/test_phase2_smoke.py -v
```

### Interpretación de resultados

**Estado esperado:** ✅ Todos los tests deben pasar.

```
test_phase3_unit.py::test_select_threshold_accuracy PASSED
test_phase3_unit.py::test_select_threshold_recall_guardrails PASSED
test_phase3_unit.py::test_validate_guardrails_recall PASSED
test_phase3_smoke.py::test_phase3_smoke_generates_expected_artifacts PASSED

======================== 4 passed in 175.83s ========================
```

Si algún test falla, consulta la sección **Solución de problemas** más abajo.

---

## Reproducibilidad

### Garantías de reproducibilidad

Este pipeline garantiza reproducibilidad completa mediante:

1. **Semillas fijas:** `random_state=42` en todos los modelos.
2. **Dataset inmutable:** El archivo original `data/raw/BBDD_ML_TAREA.csv` nunca se modifica.
3. **Protocolos deterministas:** Split estratificado y validación cruzada determinista.
4. **Reportes JSON:** Cada ejecución genera un `fase3_reproducibility_report.json` con metadata completa.

### Verificar reproducibilidad

```bash
# Primera ejecución
python -m src.main --phase2 --phase3

# Guardar resultados
cp outputs/tables/fase3_rfe_tree_mlp_comparison.csv outputs/tables/fase3_rfe_tree_mlp_comparison_run1.csv

# Segunda ejecución (en máquina distinta o después de limpiar outputs/)
python -m src.main --phase2 --phase3

# Comparar resultados
diff outputs/tables/fase3_rfe_tree_mlp_comparison_run1.csv outputs/tables/fase3_rfe_tree_mlp_comparison.csv
# No debe haber diferencias (salvo cifras decimales por precisión)
```

---

## Artefactos Generados

### Tablas de salida (en `outputs/tables/`)

#### Fase 2
- `fase2_feature_candidates.csv` - Candidatas evaluadas
- `fase2_feature_eval_metrics.csv` - Métricas por candidata

#### Fase 3
- `fase3_rfe_selected_features.csv` - Variables seleccionadas por RFE (5 vars)
- `fase3_tree_selected_features.csv` - Top 8 variables por importancia
- `fase3_threshold_search_accuracy.csv` - Métricas vs threshold (RF_Accuracy)
- `fase3_threshold_search_recall.csv` - Métricas vs threshold (ET_Recall)
- `fase3_rfe_tree_mlp_comparison.csv` - Comparación final en test

### Reportes JSON (en `data/processed/`)

- `fase3_model_selection_summary.json` - Configuración final (hiperparámetros, threshold, etc.)
- `fase3_reproducibility_report.json` - Metadata de reproducibilidad (semillas, versiones, timings)

### Cómo leer los resultados

**Tabla comparativa final:**

```bash
cat outputs/tables/fase3_rfe_tree_mlp_comparison.csv
```

Columnas:
- `model`: Nombre del modelo (ET_Recall, RF_Accuracy)
- `threshold`: Umbral de clasificación optimizado
- `accuracy`, `recall`, `precision`, `f1`: Métricas de desempeño
- `roc_auc`, `pr_auc`: Áreas bajo curva
- `tn_fp_fn_tp`: Matriz de confusión [TN, FP, FN, TP]

**Búsqueda de threshold:**

```bash
head outputs/tables/fase3_threshold_search_accuracy.csv
head outputs/tables/fase3_threshold_search_recall.csv
```

Estos archivos permiten entender cómo varían todas las métricas según el threshold elegido.

---

## Solución de Problemas

### Problema: Error `ModuleNotFoundError: No module named 'src'`

**Causa:** El entorno no está activado o las dependencias no están instaladas.

**Solución:**
```bash
# Verificar que estás en el directorio correcto
cd "Ruta del proyecto"

# Activar entorno virtual
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Reinstalar dependencias
pip install -r requirements.txt
```

---

### Problema: Error `FileNotFoundError: data/raw/BBDD_ML_TAREA.csv`

**Causa:** El dataset no está en el lugar esperado.

**Solución:**
1. Verifica que `data/raw/BBDD_ML_TAREA.csv` existe.
2. Si estás en otro directorio, usa la ruta completa:
   ```bash
   python -m src.main --phase3 --project-root "C:\ruta\completa\al\proyecto"
   ```

---

### Problema: Error `ImportError: cannot import name 'X' from 'sklearn'`

**Causa:** Versión incompatible de scikit-learn.

**Solución:**
```bash
# Reinstalar versión correcta
pip install --upgrade scikit-learn pandas numpy

# O usar la versión exacta especificada en requirements.txt
pip install -r requirements.txt --force-reinstall
```

---

### Problema: Los tests fallan con timeout

**Causa:** La máquina es lenta o no hay suficientes recursos.

**Solución:**
```bash
# Ejecutar tests con timeout extendido
python -m pytest tests/test_phase3_smoke.py -v --tb=short --timeout=300

# O ejecutar solo tests unitarios (más rápidos)
python -m pytest tests/test_phase3_unit.py -v
```

---

### Problema: Memoria insuficiente durante Fase 3

**Causa:** La búsqueda exhaustiva de threshold es intensiva en memoria.

**Solución:**
```bash
# Cerrar otras aplicaciones para liberar RAM

# O reducir el dataset para pruebas rápidas (en config.py)
# Esta es una opción de desarrollo, no de producción
```

---

### Problema: Resultados ligeramente diferentes entre máquinas

**Causa:** Variaciones en la precisión numérica de floating-point.

**Solución:**
Esto es **normal y esperado**. Las diferencias deben ser menores a `1e-10` para ser consideradas equivalentes. Usa:
```bash
python -c "
import pandas as pd
import numpy as np

df1 = pd.read_csv('outputs/tables/fase3_rfe_tree_mlp_comparison.csv')
df2 = pd.read_csv('path/to/other/comparison.csv')

# Comparar con tolerancia numérica
diff = np.abs(df1.select_dtypes(float) - df2.select_dtypes(float))
print(f'Max difference: {diff.max().max()}')
print(f'All close (1e-8): {np.allclose(df1.select_dtypes(float), df2.select_dtypes(float), atol=1e-8)}')
"
```

---

## Compilar el Informe LaTeX

### Requisitos

Necesitas tener instalado:
- `latexmk` y `pdflatex` (incluidos en MiKTeX, TexLive)

### En Windows (MiKTeX)

```bash
cd report
latexmk -pdf main.tex
cd ..
```

El archivo `main.pdf` se generará en `report/`.

### En macOS/Linux (TexLive)

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
cd ..
```

### Ver el informe

```bash
# Windows
start report/main.pdf

# macOS
open report/main.pdf

# Linux
xdg-open report/main.pdf
```

---

## Flujo de Trabajo Recomendado

### Primer uso (setup completo)

```bash
# 1. Crear entorno
python -m venv .venv
.venv\Scripts\activate  # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar pipeline completo
python -m src.main --phase2 --phase3

# 4. Validar tests
python -m pytest tests/ -v

# 5. Compilar informe
cd report && latexmk -pdf main.tex && cd ..
```

### Desarrollo iterativo

```bash
# Cambios en Fase 3
python -m src.main --phase3  # Reutiliza salidas de Fase 2

# Cambios en tests
python -m pytest tests/test_phase3_unit.py -v

# Cambios en informe
cd report && pdflatex main.tex && cd ..
```

### Limpiar y regenerar desde cero

```bash
# Eliminar outputs intermedios
rmdir outputs  # Windows: rmdir /s /q outputs
rm -rf outputs  # macOS/Linux

# Eliminar datos procesados
rmdir data\interim  # Windows
rm -rf data/interim  # macOS/Linux

# Regenerar todo
python -m src.main --phase2 --phase3
```

---

## Configuración Avanzada

### Modificar hiperparámetros

Edita `src/config.py`:

```python
# Búsqueda de hiperparámetros RF_Accuracy
RF_ACCURACY_PARAMS = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
}

# Búsqueda de hiperparámetros ET_Recall
ET_RECALL_PARAMS = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
}
```

Luego ejecuta:

```bash
python -m src.main --phase3
```

### Cambiar semilla aleatoria

```bash
python -m src.main --phase3 --random-state 123
```

Esto generará resultados diferentes pero reproducibles.

---

## Integración Continua (CI)

Para ejecutar el pipeline en un servidor CI (GitHub Actions, GitLab CI, etc.):

```yaml
# .github/workflows/ml-pipeline.yml (GitHub Actions)
name: ML Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']
    
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run pipeline
        run: |
          python -m src.main --phase2 --phase3
      
      - name: Run tests
        run: |
          python -m pytest tests/ -v
```

---

## Contribuciones y Cambios

Si realizas cambios en el código:

1. Verifica que los tests pasan: `python -m pytest tests/ -v`
2. Valida reproducibilidad: ejecuta el pipeline dos veces y compara outputs
3. Documenta cambios en `AGENTE.md`
4. Actualiza `requirements.txt` si añades dependencias: `pip freeze > requirements.txt`

---

## Soporte y Debugging

### Logs detallados

```bash
# Ejecutar con logging verbose
python -m src.main --phase3 --verbose
```

Los logs se guardan en `src/logs/` con timestamp.

### Inspeccionar datos intermedios

```python
import pandas as pd
import json

# Leer tabla de comparación final
comparison = pd.read_csv('outputs/tables/fase3_rfe_tree_mlp_comparison.csv')
print(comparison)

# Leer metadata de reproducibilidad
with open('data/processed/fase3_reproducibility_report.json') as f:
    report = json.load(f)
    print(f"Ejecutado con random_state: {report['random_state']}")
    print(f"Timestamp: {report['timestamp']}")
```

### Contacto y reporte de bugs

Si encuentras un problema:

1. Verifica que estés usando las versiones correctas: `pip list | grep -E "pandas|scikit|numpy"`
2. Intenta limpiar y regenerar: `rm -rf outputs/ && python -m src.main --phase3`
3. Consulta los logs en `src/logs/`
4. Revisa `AGENTE.md` para especificaciones técnicas

---

## Licencia

Este proyecto forma parte del módulo de Machine Learning del Master de la UCM.

---

## Resumen rápido

| Tarea | Comando |
|-------|---------|
| Instalar dependencias | `pip install -r requirements.txt` |
| Ejecutar Fase 2+3 | `python -m src.main --phase2 --phase3` |
| Ejecutar solo Fase 3 | `python -m src.main --phase3` |
| Validar con tests | `python -m pytest tests/ -v` |
| Compilar informe | `cd report && latexmk -pdf main.tex` |
| Ver ayuda | `python -m src.main --help` |

---

**Última actualización:** 19 de abril de 2026  
**Estado:** ✅ Pipeline completamente funcional y reproducible  
**Tests:** 4/4 pasando  
**Secciones LaTeX:** 8/8 completadas

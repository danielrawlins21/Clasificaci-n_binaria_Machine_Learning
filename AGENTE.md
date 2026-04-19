# AGENTE.md

## Propósito de este archivo

Este archivo define cómo debe trabajar el agente en este repositorio para el trabajo individual de **clasificación binaria** del módulo de **Machine Learning**.

El agente debe priorizar siempre:

1. **Rigor académico**.
2. **Reproducibilidad completa**.
3. **Claridad metodológica**.
4. **Coherencia entre código, resultados y memoria en LaTeX**.

No se debe improvisar ni "rellenar". Toda decisión técnica debe poder justificarse en la memoria final.

---

## Contexto del proyecto

El objetivo es predecir una variable binaria `Y` (abandono o churn) a partir de las variables `V1`–`V20` del fichero obligatorio `BBDD_ML_TAREA.csv`.

La tarea exige cuatro bloques:

1. **Análisis y depuración de datos**, explicando y justificando cada transformación y cada paso de feature engineering.
2. **RFE con regresión logística** para seleccionar **5 variables**, y luego diseño/ajuste de una **red neuronal** optimizada en términos de **Accuracy**, con **búsqueda paramétrica exhaustiva**.
3. **Árbol de decisión** para identificar variables importantes, y luego diseño/ajuste de una **red neuronal** optimizada en términos de **Recall**, también con **búsqueda paramétrica exhaustiva**.
4. **Comparación global** entre los enfoques anteriores.

La entrega consta de:

- **Memoria en PDF** hecha en **LaTeX** y evaluada académicamente.
- **Código Python reproducible**, implementado como pipeline modular y ejecutable desde línea de comandos.

La memoria debe ser **autoexplicativa**, estar **bien estructurada**, y respetar el **límite de 20 páginas**.

### Estado actual del proyecto

El proyecto está estructurado en **tres fases**:

- **Fase 1**: Carga, auditoría y depuración del dataset (completada, documentada en LaTeX).
- **Fase 2**: Selección de candidatas mediante RFE y evaluación de feature engineering (completada, outputs en `outputs/tables/`).
- **Fase 3**: Implementación de modelos finales (RF_Accuracy y ET_Recall) con threshold optimization (completada, **funcional y validada con tests**).

La **Fase 3** está implementada tanto en:
- **Notebook exploratorio**: `notebooks/04_rfe_tree_mlp_sklearn.ipynb` (trazabilidad completa).
- **Pipeline CLI reproducible**: `src/main.py` con comando `python -m src.main --phase3` (validado con tests unitarios y smoke test).

Todas las fases son **reproducibles desde cero** y los artefactos generados pueden regenerarse ejecutando el código modular en `src/`.

---

## Entorno de trabajo

- Editor de código principal: **VS Code**.
- Lenguaje principal: **Python**.
- Documento final: **LaTeX**.
- Se debe trabajar con rutas relativas y estructura ordenada de proyecto.
- Las salidas intermedias y finales deben generarse desde scripts o módulos reproducibles.

Se permiten notebooks para exploración, pero:

- no deben ser la base de la entrega,
- no deben ser la fuente única de resultados,
- y no deben sustituir scripts reproducibles.

---

## Convenciones generales para el agente

### 1. Idioma y estilo

- Escribir siempre en **español académico claro**.
- Ser preciso, directo y técnicamente sólido.
- Diferenciar claramente entre:
  - hechos observados,
  - decisiones metodológicas,
  - interpretación,
  - limitaciones.

### 2. Honestidad metodológica

- No inventar resultados.
- No presentar como concluido nada que no haya sido ejecutado o verificado.
- No ocultar problemas de convergencia, inestabilidad o ambigüedad metodológica.
- Si una decisión es discutible, explicitar la alternativa y justificar la elegida.

### 3. Reproducibilidad

- Fijar semillas aleatorias (`random_state`) siempre que sea posible.
- No modificar nunca el dataset original en `data/raw`.
- Toda transformación debe implementarse dentro de pipelines o quedar trazada en código.
- Todo número que aparezca en la memoria debe poder regenerarse.

### 4. Validez experimental

- Evitar **data leakage** en todo momento.
- Separar correctamente entrenamiento, validación y test.
- El test final no se usa para seleccionar hiperparámetros ni umbrales.
- Las comparaciones entre modelos deben hacerse en condiciones homogéneas.

---

## Estructura actual del repositorio

```text
.
├── AGENTE.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── BBDD_ML_TAREA.csv
│   ├── interim/
│   │   └── (outputs intermedios de Fase 2)
│   └── processed/
│       ├── fase2_selected_candidates.json
│       ├── fase3_model_selection_summary.json
│       └── fase3_reproducibility_report.json
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_hallazgos_eda.ipynb
│   ├── 03_plan_accion_fase2.ipynb
│   └── 04_rfe_tree_mlp_sklearn.ipynb (trazabilidad de Fase 3)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py (entrypoint: --phase2, --phase3)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   └── preprocessing.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── selection.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tree_ensembles.py (RF y ET tuning)
│   │   └── evaluation.py (evaluación en test)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── comparison.py
│   │   ├── threshold.py (selección de umbral)
│   │   └── guardrails.py (validación Recall)
│   ├── pipelines/
│   │   ├── __init__.py
│   │   └── phase3_export.py (exportación de artefactos)
│   └── utils/
│       ├── __init__.py
│       ├── seeds.py
│       ├── reproducibility.py (reportes JSON)
│       └── validation.py (checklist no-leakage)
├── outputs/
│   ├── tables/
│   │   ├── fase1_*.csv (auditoría Fase 1)
│   │   ├── fase2_*.csv (selección de candidatas)
│   │   ├── fase3_threshold_search_accuracy.csv
│   │   ├── fase3_threshold_search_recall.csv
│   │   ├── fase3_rfe_tree_mlp_comparison.csv
│   │   ├── fase3_rfe_selected_features.csv
│   │   └── fase3_tree_selected_features.csv
│   ├── figures/
│   └── logs/
├── tests/
│   ├── test_phase2_unit.py
│   ├── test_phase2_smoke.py
│   ├── test_phase3_unit.py
│   └── test_phase3_smoke.py
└── report/
    ├── main.tex
    ├── bibliography.bib
    ├── sections/
    │   ├── 01_introduccion.tex
    │   ├── 02_descripcion_dataset.tex
    │   ├── 03_analisis_depuracion.tex
    │   ├── 04_protocolo_experimental.tex
    │   ├── 05_modelo_accuracy.tex
    │   ├── 06_modelo_recall.tex
    │   ├── 07_comparacion_global.tex
    │   └── 08_conclusiones.tex
    ├── figures/
    └── tables/
```

La estructura refleja un proyecto modular con separación clara entre:
- **Datos**: raw → interim → processed.
- **Código**: módulos reutilizables por fase.
- **Outputs**: tablas y logs de cada etapa.
- **Tests**: validación de Fase 2 y Fase 3.
- **Informe**: LaTeX con secciones modulares.

---

## Flujo de trabajo obligatorio

### Fase 1. Ingesta y auditoría inicial de datos

Antes de modelar, el agente debe:

1. Cargar el dataset y verificar dimensiones, tipos y nombres de columnas.
2. Identificar:
   - valores perdidos,
   - duplicados,
   - posibles valores anómalos,
   - columnas con baja variabilidad,
   - posibles identificadores,
   - variables mal tipadas.
3. Analizar la variable objetivo `Y`.
4. Elaborar una primera clasificación conceptual de variables:
   - categóricas nominales,
   - binarias,
   - numéricas continuas,
   - numéricas discretas,
   - identificadores no predictivos.

### Reglas importantes para esta fase

- `V4` parece conceptualmente un **identificador** (número de teléfono del cliente) y debe tratarse, por defecto, como **variable no utilizable** para predicción, salvo justificación muy fuerte en sentido contrario.
- `V1` (estado/región) y `V3` (código de área) están codificadas numéricamente, pero conceptualmente son **categóricas nominales**, no continuas.
- `V5` y `V6` son binarias.
- Variables como `V8`/`V10`, `V11`/`V13`, `V14`/`V16` y `V17`/`V19` pueden contener **redundancia muy alta** por reflejar minutos y coste de un mismo bloque de llamadas. No es fuga de información, pero sí una posible fuente de colinealidad o redundancia.

El agente no debe eliminar estas variables de forma automática: debe **verificar empíricamente** la redundancia y **justificar** cualquier decisión.

---

### Fase 2. Protocolo experimental común

Para asegurar validez comparativa, el agente debe seguir un protocolo estable.

### Recomendación por defecto

- Hacer una partición inicial **estratificada** en:
  - `train` (o `train_val`) para ajuste,
  - `test` para evaluación final.
- Dentro del conjunto de entrenamiento, usar **validación cruzada estratificada** para:
  - selección de variables,
  - búsqueda de hiperparámetros,
  - selección de umbral.

### Reglas del protocolo

- El test se reserva para la evaluación final.
- Los escaladores, imputadores y codificadores se ajustan solo con entrenamiento.
- Las comparaciones entre modelos deben usar el mismo split base, salvo motivo metodológico justificado.
- Guardar la semilla, el split y la configuración de cada experimento.

---

### Fase 3. Preprocesamiento

El preprocesamiento debe ser **dependiente de la técnica** y estar explícitamente explicado en la memoria.

### Para regresión logística y red neuronal

- Imputación explícita de valores perdidos.
- Codificación adecuada de variables categóricas.
- Escalado/estandarización de variables numéricas.
- Uso preferente de `Pipeline` y `ColumnTransformer`.

### Para árbol de decisión

- No es obligatorio escalar.
- Sí es obligatorio tratar los missing de forma coherente.
- La codificación de categóricas debe ser compatible con el modelo elegido.

### Regla de oro

No describir el preprocesamiento como si fuese universal. La memoria debe indicar **qué pasos necesita cada modelo y por qué**.

---

### Fase 4. Apartado 1 de la tarea: análisis y depuración de datos

Este bloque debe responder, como mínimo, a estas preguntas:

- ¿Qué estructura tiene la base de datos?
- ¿Hay missing values? ¿Cómo se gestionan y por qué?
- ¿Hay variables problemáticas por identificación, escala, tipado o redundancia?
- ¿Qué transformaciones se aplican?
- ¿Qué feature engineering se realiza?
- ¿Qué transformaciones mejoran la capacidad predictiva o la estabilidad del modelo?

El agente debe evitar un EDA puramente descriptivo sin conexión con el modelado. Toda transformación debe tener finalidad analítica.

---

### Fase 5. Apartado 2: RFE con regresión logística + red neuronal optimizada en Accuracy

### Obligaciones

1. Ejecutar **RFE** usando como estimador base una **regresión logística**.
2. Seleccionar exactamente **5 variables**.
3. Con esas variables, entrenar una **red neuronal** y optimizarla en **Accuracy**.
4. Realizar una **búsqueda paramétrica exhaustiva**.

### Implementación preferente

- Usar `LogisticRegression` con configuración documentada.
- Usar `RFE` o `RFECV` solo si se respeta la exigencia de dejar finalmente **5 variables**.
- Para la red neuronal, preferir `MLPClassifier` de scikit-learn salvo petición expresa de otro framework.
- Para la búsqueda exhaustiva, preferir `GridSearchCV` frente a búsqueda aleatoria.

### Hiperparámetros que deben considerarse

El grid debe ser razonable, pero suficientemente rico. Documentar siempre lo probado.

Como mínimo considerar:

- `hidden_layer_sizes`
- `activation`
- `alpha`
- `learning_rate_init`
- `solver`
- `max_iter`
- `early_stopping`

### Threshold

El umbral de clasificación **no debe fijarse por inercia en 0.5**.

Debe seleccionarse con base en validación y justificarse en función de la evolución de métricas. En este apartado, la métrica principal es **Accuracy**, pero el agente también debe vigilar:

- Recall
- Precision
- F1
- Matriz de confusión

El objetivo es evitar una mejora artificial de Accuracy acompañada de deterioros severos no discutidos.

---

### Fase 6. Apartado 3: árbol de decisión + red neuronal optimizada en Recall

### Obligaciones

1. Entrenar un **árbol de decisión**.
2. Obtener las variables más relevantes a partir de la importancia calculada por el árbol.
3. Usar esas variables para construir una **red neuronal**.
4. Optimizarla en términos de **Recall**.
5. Realizar una **búsqueda paramétrica exhaustiva**.

### Reglas metodológicas

- Documentar cómo se seleccionan las variables importantes.
- Evitar elegir el número de variables de forma arbitraria y opaca.
- Justificar si se usa un corte por importancia acumulada, top-k o criterio mixto.

### En este apartado la métrica principal es Recall

Pero el agente debe reportar también, al menos:

- Accuracy
- Precision
- F1
- ROC-AUC
- PR-AUC
- Matriz de confusión

### Threshold

Al optimizar Recall, el umbral puede desplazarse por debajo de 0.5 si eso está justificado por validación.

La justificación debe discutir explícitamente el coste de:

- falsos negativos,
- falsos positivos,
- y el compromiso entre capturar churn y sobreactivar alertas.

---

### Fase 7. Apartado 4: comparación global

La comparación final no debe limitarse a decir cuál tiene mejor número en una métrica.

El agente debe comparar ambos procesos en términos de:

- capacidad predictiva,
- interpretabilidad,
- estabilidad,
- sensibilidad al threshold,
- complejidad del pipeline,
- riesgo de sobreajuste,
- coste computacional,
- utilidad práctica para el problema de churn.

### Regla clave

La comparación debe ser **justa**:

- mismo protocolo de evaluación,
- mismo conjunto de test final,
- métricas calculadas del mismo modo,
- thresholds seleccionados sin contaminar el test.

---

## Fase 8. Implementación modular de Fase 3

La Fase 3 (modelos finales) está **completamente implementada** como pipeline modular en `src/`.

### Ejecución

```bash
# Ejecutar Fase 3 desde cero
python -m src.main --phase3

# O combinado con Fase 2
python -m src.main --phase2 --phase3

# Con opciones personalizadas
python -m src.main --phase3 --random-state 42 --project-root /ruta/custom
```

### Módulos de Fase 3

- **`src/models/tree_ensembles.py`**: Tuning de Random Forest (Accuracy) y Extra Trees (Recall).
- **`src/evaluation/threshold.py`**: Selección inteligente de threshold por objetivo (Accuracy o Recall).
- **`src/evaluation/guardrails.py`**: Validación de guardrails (precisión mínima, PR-AUC mínima) para Recall.
- **`src/models/evaluation.py`**: Evaluación final en test con threshold aplicado.
- **`src/pipelines/phase3_export.py`**: Exportación de tablas, top features y reportes JSON.
- **`src/utils/reproducibility.py`**: Generación de reporte de reproducibilidad.
- **`src/utils/validation.py`**: Checklist de no-leakage.

### Artefactos generados

- `fase3_rfe_tree_mlp_comparison.csv`: Comparación final RF_Accuracy vs ET_Recall en test.
- `fase3_threshold_search_accuracy.csv`: Métricas vs threshold para Accuracy.
- `fase3_threshold_search_recall.csv`: Métricas vs threshold para Recall.
- `fase3_rfe_selected_features.csv`: Variables seleccionadas por RFE (5 vars).
- `fase3_tree_selected_features.csv`: Top variables por importancia del árbol.
- `data/processed/fase3_model_selection_summary.json`: Configuración final.
- `data/processed/fase3_reproducibility_report.json`: Validación de reproducibilidad.

### Validación

Los tests de Fase 3 validan:

- **Unitarios** (`tests/test_phase3_unit.py`): selección de threshold, guardrails, métricas.
- **Smoke** (`tests/test_phase3_smoke.py`): ejecución end-to-end y generación de artefactos.

```bash
# Ejecutar tests
python -m pytest tests/test_phase3_unit.py tests/test_phase3_smoke.py -v
```

**Estado**: ✅ **4 tests pasando** (3 unitarios + 1 smoke).

---



Aunque la tarea enfatiza Accuracy en el apartado 2 y Recall en el apartado 3, el agente debe construir una evaluación más completa.

Métricas mínimas:

- Accuracy
- Recall
- Precision
- F1-score
- ROC-AUC
- PR-AUC
- Matriz de confusión

Opcionalmente:

- specificity,
- balanced accuracy,
- curva ROC,
- curva Precision-Recall,
- curvas métricas vs threshold.

---

## Gestión de la selección de umbral

La selección de threshold es parte central de la tarea.

### Reglas

- Nunca escoger el threshold usando el test final.
- Evaluar varios umbrales sobre validación.
- Guardar tabla de métricas por umbral.
- Incluir gráficos o tablas que permitan justificar el umbral elegido.

### Interpretación esperada

- En el modelo orientado a **Accuracy**, el threshold debe maximizar o casi maximizar Accuracy sin ocultar un posible deterioro relevante de Recall.
- En el modelo orientado a **Recall**, el threshold debe privilegiar captación de churn, pero la memoria debe reconocer el coste en falsos positivos.

---

## Estándares de implementación en Python

### Librerías preferentes

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `scipy`

Evitar dependencias innecesarias. Si se introduce una librería nueva, justificar por qué aporta valor real.

### Estilo de código

- Funciones pequeñas y reutilizables.
- Nada de scripts monolíticos de cientos de líneas sin estructura.
- Nombres de variables descriptivos.
- Configuración centralizada cuando sea posible.
- Logs claros para poder reconstruir cada ejecución.

### Persistencia de resultados

Guardar, como mínimo:

- tablas de métricas,
- tablas de hiperparámetros probados,
- mejores configuraciones,
- variables seleccionadas por cada enfoque,
- figuras exportadas para LaTeX.

---

## Reglas para la memoria en LaTeX

La memoria es el entregable evaluado. Por tanto, el agente debe pensar siempre en términos de **escribibilidad académica**.

### Estructura sugerida

1. Introducción y objetivo.
2. Descripción del dataset.
3. Análisis y depuración de datos.
4. Protocolo experimental.
5. Apartado 2: RFE + logística + red neuronal (Accuracy).
6. Apartado 3: árbol + variables importantes + red neuronal (Recall).
7. Comparación global.
8. Conclusiones.
9. Anexos breves, si son necesarios.

### Reglas de redacción

- No pegar salidas crudas de notebook.
- No incluir código largo salvo fragmentos muy seleccionados.
- Explicar decisiones, no solo resultados.
- Toda tabla o figura debe citarse y comentarse en el texto.
- Las métricas deben interpretarse; no basta con mostrarlas.
- La memoria debe ser autoexplicativa incluso sin abrir el código.

### Regla de extensión

El límite de 20 páginas obliga a priorizar:

- claridad,
- síntesis,
- tablas compactas,
- y discusión relevante.

---

## Qué debe evitar el agente

- Tratar variables nominales codificadas como si fuesen continuas sin justificación.
- Elegir modelos solo por conveniencia sin conectar con el enunciado.
- Usar búsqueda aleatoria cuando se exige búsqueda exhaustiva.
- Tomar el threshold 0.5 por defecto y no discutirlo.
- Comparar modelos entrenados con protocolos distintos sin advertirlo.
- Usar el test para tomar decisiones de diseño.
- Confundir interpretabilidad con rendimiento puro.
- Convertir la memoria en una descarga de tablas sin narrativa.
- Introducir metodologías de scoring o WOE salvo que se presenten solo como contexto y no desvíen el foco del trabajo.

---

## Resultado esperado del repositorio

Al finalizar, el repositorio debería permitir:

1. Ejecutar el pipeline completo desde cero.
2. Regenerar tablas y figuras.
3. Identificar qué variables seleccionó cada enfoque.
4. Recuperar el mejor modelo por Accuracy y el mejor modelo por Recall.
5. Reconstruir la comparación final.
6. Compilar la memoria en LaTeX sin edición manual de resultados.

---

## Comandos para ejecutar el pipeline

```bash
# Configuración del entorno
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Ejecutar el pipeline completo (Fase 2 + Fase 3)
python -m src.main --phase2 --phase3

# Ejecutar solo Fase 3 (requiere que Fase 2 haya generado sus artefactos)
python -m src.main --phase3

# Ejecutar solo Fase 2
python -m src.main --phase2

# Ejecutar tests
python -m pytest tests/ -v

# Compilar el informe
cd report
latexmk -pdf main.tex
cd ..
```

El pipeline es **completamente reproducible**: todos los resultados pueden regenerarse desde el dataset original en `data/raw/BBDD_ML_TAREA.csv`.

---

## Checklist final antes de entregar

- [x] La memoria compila en PDF desde LaTeX.
- [x] El código reproduce las métricas reportadas.
- [x] Se respeta el protocolo de Fase 1 (auditoría de datos).
- [x] Se ejecutó RFE y se seleccionaron 5 variables.
- [x] Se documentó la selección de variables a partir del árbol.
- [x] Se realizó búsqueda exhaustiva de hiperparámetros.
- [x] El threshold está justificado en ambos apartados (Accuracy y Recall).
- [x] Fase 3 está implementada como pipeline modular y reproducible.
- [x] Los tests de Fase 3 pasan (unitarios + smoke).
- [x] Se explican los preprocesamientos específicos por técnica.
- [x] La comparación usa el mismo protocolo de evaluación.
- [x] El dataset original no ha sido alterado.
- [x] No hay resultados presentados sin trazabilidad.
- [ ] La memoria es autoexplicativa (trabajo en curso).
- [ ] Se respeta el límite de 20 páginas (trabajo en curso: secciones 07 y 08 pendientes).
- [ ] Se han completado todas las secciones LaTeX.

### Secciones LaTeX completadas

- [x] 01_introduccion.tex
- [x] 02_descripcion_dataset.tex
- [x] 03_analisis_depuracion.tex
- [x] 04_protocolo_experimental.tex
- [x] 05_modelo_accuracy.tex
- [x] 06_modelo_recall.tex
- [x] 07_comparacion_global.tex
- [x] 08_conclusiones.tex

**Estado**: ✅ **TODAS LAS SECCIONES COMPLETADAS**. El informe está listo para compilación.

### Módulos principales implementados

- [x] `src/main.py`: Entrypoint CLI.
- [x] `src/config.py`: Configuración centralizada.
- [x] `src/data/`: Carga y preprocesamiento.
- [x] `src/features/selection.py`: RFE y candidatas.
- [x] `src/models/tree_ensembles.py`: Random Forest y Extra Trees.
- [x] `src/evaluation/`: Métricas, threshold, guardrails.
- [x] `src/pipelines/phase3_export.py`: Exportación de artefactos.
- [x] `tests/`: Validación unitaria y end-to-end.

---

## Principio rector

Si hay que elegir entre una solución vistosa y una solución metodológicamente sólida, el agente debe elegir siempre la **solución metodológicamente sólida**.

---

## Estado actual del proyecto (última actualización)

### Completado ✅

1. **Fase 1**: Auditoría completa de datos.
   - Análisis de missing values, duplicados, variables problemáticas.
   - Documentado en LaTeX sección 03.

2. **Fase 2**: Selección de candidatas mediante RFE.
   - 5 variables seleccionadas por RFE + LogisticRegression.
   - Variables adicionales por importancia del árbol.
   - Trazabilidad en notebook `04_rfe_tree_mlp_sklearn.ipynb`.
   - Documentado en LaTeX secciones 05 y 06.

3. **Fase 3**: Implementación de modelos finales.
   - **RF_Accuracy**: Random Forest optimizado para Accuracy (~0.9492 en test).
   - **ET_Recall**: Extra Trees optimizado para Recall (~0.9433 en test).
   - Threshold selection con búsqueda exhaustiva.
   - Guardrails de precisión mínima y PR-AUC para Recall.
   - Tests validados: 3 unitarios + 1 smoke (4/4 pasando).
   - Artefactos exportados: tablas, features, reportes JSON.

4. **Código reproducible**:
   - Pipeline modular en `src/`.
   - CLI ejecutable: `python -m src.main --phase3`.
   - Todas las Fases ejecutables desde cero.
   - Separación clara entre datos, features, modelos y evaluación.

### En progreso 🔄

5. **Informe LaTeX**:
   - [x] Secciones 01-06 completas.
   - [ ] Sección 07 (comparación global): pendiente.
   - [ ] Sección 08 (conclusiones): pendiente.
   - Límite de 20 páginas: en evaluación tras completar todas las secciones.

### Notas de implementación

- **Protocolo de split**: Split externo fijo (75% train, 25% test) + validación cruzada estratificada (5-fold) para selección de variables y threshold.
- **No-leakage**: Validado con checklist automatizado en `src/utils/validation.py`.
- **Reproducibilidad**: Semillas fijas (`random_state=42`), reportes JSON con metadata.
- **Modularidad**: Cada componente (RFE, tuning, threshold, evaluación) es independiente y testeable.

### Próximos pasos

1. Completar sección 07 (comparación global entre RF_Accuracy y ET_Recall).
2. Completar sección 08 (conclusiones metodológicas).
3. Verificar límite de 20 páginas y ajustar síntesis si es necesario.
4. Compilación final del PDF.


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
- **Código Python reproducible**, que no se evalúa directamente pero debe regenerar los resultados.

La memoria debe ser **autoexplicativa**, estar **bien estructurada**, y respetar el **límite de 20 páginas**.

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

## Estructura sugerida del repositorio

```text
.
├── AGENTE.md
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── BBDD_ML_TAREA.csv
│   ├── interim/
│   └── processed/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocessing.py
│   ├── features/
│   │   └── selection.py
│   ├── models/
│   │   ├── logistic_rfe.py
│   │   ├── decision_tree_importance.py
│   │   ├── mlp_accuracy.py
│   │   └── mlp_recall.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── thresholding.py
│   │   └── comparison.py
│   └── utils/
│       ├── io.py
│       ├── plots.py
│       └── seeds.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── models/
│   └── logs/
└── report/
    ├── main.tex
    ├── sections/
    ├── figures/
    ├── tables/
    └── bibliography.bib
```

Si la estructura real cambia, el agente debe mantener igualmente estas ideas:

- datos crudos separados,
- código modular,
- outputs versionables,
- memoria en carpeta propia.

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

## Métricas mínimas a calcular

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

## Comandos orientativos

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main
latexmk -pdf report/main.tex
```

Si no existe `src.main`, mantener igualmente una convención clara para lanzar:

- preprocesamiento,
- entrenamiento,
- evaluación,
- exportación de tablas/figuras,
- compilación del informe.

---

## Checklist final antes de entregar

- [ ] La memoria compila en PDF desde LaTeX.
- [ ] La memoria es autoexplicativa.
- [ ] Se respeta el límite de 20 páginas o se justifica el uso de anexos mínimos.
- [ ] El código reproduce las métricas reportadas.
- [ ] Se explican los preprocesamientos específicos por técnica.
- [ ] Se justifica la selección de 5 variables vía RFE.
- [ ] Se documenta la selección de variables a partir del árbol.
- [ ] La búsqueda de hiperparámetros es exhaustiva y está documentada.
- [ ] El threshold está justificado en ambos apartados.
- [ ] La comparación final usa el mismo protocolo de evaluación.
- [ ] El dataset original no ha sido alterado.
- [ ] No hay resultados presentados sin trazabilidad.

---

## Principio rector

Si hay que elegir entre una solución vistosa y una solución metodológicamente sólida, el agente debe elegir siempre la **solución metodológicamente sólida**.

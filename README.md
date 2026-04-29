# PyCrim Experiments — Predicción de Sentencias Penales en Paraguay

> **Evaluación Comparativa de Algoritmos Clásicos y Modelos Transformer para la Predicción de Sentencias de la Sala Penal de la Corte Suprema de Justicia de Paraguay**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: Zenodo](https://img.shields.io/badge/Dataset-Zenodo-1D91C0.svg)](https://doi.org/10.5281/zenodo.14373749)

---

## 📄 Publicación

Este repositorio contiene el código fuente de los experimentos del siguiente trabajo académico:

> **Ferreira, F., Franco, B., Amarilla Closs, C., Gómez-Adorno, H., & Rojas Moreno, R.**  
> *"Predicción de Sentencias Penales en Paraguay: Evaluación Comparativa de Algoritmos Clásicos y Modelos Transformer."*  
> Facultad Politécnica, Universidad Nacional de Asunción, San Lorenzo, Paraguay.

**Corpus utilizado:**  
Gómez Adorno, H., Vázquez Noguera, J. L., Amarila Closs, C., & Vázquez-Cerrillo, J. (2024). *Dataset of the criminal chamber cases from the Supreme Court of Justice of Paraguay*. Zenodo. [https://doi.org/10.5281/zenodo.14373749](https://doi.org/10.5281/zenodo.14373749)

---

## 📋 Resumen

Este trabajo presenta la **primera evaluación comparativa formal** entre algoritmos clásicos de aprendizaje supervisado y modelos Transformer preentrenados para la predicción de sentencias penales en Paraguay, utilizando el corpus público **PyCrim** (5.000 resoluciones, 2011–2023).

### Familias de modelos evaluadas

| Familia | Modelos |
|---|---|
| **Algoritmos Clásicos** | Logistic Regression (LR), Support Vector Machines (SVM), Random Forests (RF), Naive Bayes (NB) |
| **Modelos Transformer** | BETO, mBERT, XLM-RoBERTa (con y sin fine-tuning) |

### Protocolo experimental (simétrico para ambas familias)

- Partición estratificada **70/15/15** con semilla fija (`seed=42`)
- Compensación de desbalanceo mediante **pesos balanceados**
- Optimización del **umbral de decisión** sobre el conjunto de validación
- Métrica de evaluación principal: **F1-macro**

### Resultados principales

| Modelo | Configuración | F1-macro (Test) |
|---|---|---|
| **BETO** (fine-tuning, lr=5e-6) | Protocolo optimizado | **82.91%** |
| XLM-RoBERTa (fine-tuning, lr=1e-5) | Protocolo optimizado | 82.42% |
| Logistic Regression | 4-gramas, TF, sin stopwords, stemming | 82.80% |
| SVM *(seleccionado por validación)* | 4-gramas, TF-IDF, sin stopwords, stemming | 78.78% |
| mBERT (fine-tuning, lr=1e-5) | Protocolo optimizado | 77.87% |

> **Nota metodológica:** SVM fue seleccionado como representante clásico según el criterio de validación pre-establecido (F1-macro en val = 81.92%). LR obtuvo mayor F1 en prueba (82.80%), reduciendo la diferencia con BETO a 0.11 puntos bajo el protocolo simétrico adoptado.

---

## 🗂️ Estructura del repositorio

```
pycrim-transformers/
├── experiments/
│   ├── EXP-ALGORITMOS CLASICOS/    # Experimentos con algoritmos clásicos
│   └── EXP-TRANSFORMER/             # Experimentos con modelos Transformer
├── algoritmos_clasicos_70_15_15_busqueda64config.py   # Búsqueda exhaustiva (64 configs)
├── algoritmos_clasicos_evaluacion_final_test.py        # Evaluación final en test
└── template_transformers_optimizado.py                 # Template Transformer (protocolo optimizado)
```

---

## 🔬 Descripción de los experimentos

### 1. Algoritmos Clásicos (`algoritmos_clasicos_*`)

**`algoritmos_clasicos_70_15_15_busqueda64config.py`**

Realiza una búsqueda exhaustiva sobre **64 combinaciones de hiperparámetros** (producto cartesiano):

| Factor | Valores |
|---|---|
| N-gramas | unigrama, bigrama, trigrama, 4-grama |
| Stop words | con / sin eliminación |
| Stemming | con / sin (`SnowballStemmer` para español) |
| min_df | {2, 3} |
| Vectorización | TF / TF-IDF |

- Entrenamiento sobre el **70% (train)**
- Selección por **F1-macro en validación**
- El mejor modelo se evalúa **una única vez** sobre el conjunto de prueba

**`algoritmos_clasicos_evaluacion_final_test.py`**

Evaluación final con el umbral de decisión alineado con el protocolo Transformer: `np.arange(0.50, 0.601, 0.005)`.

---

### 2. Modelos Transformer (`template_transformers_optimizado.py`)

Implementa el **protocolo optimizado** para los tres modelos evaluados:

| Modelo | Vocabulario | Tipo |
|---|---|---|
| [BETO](https://github.com/dccuchile/beto) | WordPiece | Monolingüe (español) |
| [mBERT](https://huggingface.co/bert-base-multilingual-cased) | WordPiece | Multilingüe (100+ idiomas) |
| [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base) | SentencePiece | Multilingüe (100 idiomas, 2.5 TB) |

**Características del protocolo optimizado:**
- Fragmentación con **sliding window** (512 tokens, desplazamiento 384, solapamiento 128) para documentos extensos (promedio 2.496 tokens)
- Agregación por **promedio de probabilidades** entre fragmentos
- Optimizador: **AdamW** (weight decay=0.01, batch_size=16)
- Learning rates evaluados: `3e-5`, `2e-5`, `1e-5`, `5e-6`
- Warmup: 300/500 pasos; máximo 7 épocas
- **Early stopping** (paciencia=2) sobre F1-macro en validación
- Búsqueda de umbral óptimo: `np.arange(0.20, 0.61, 0.01)`

---

## 🚀 Reproducibilidad

### Requisitos

```bash
# Python 3.8+
pip install torch transformers scikit-learn pandas numpy nltk matplotlib
```

### Datos

El corpus PyCrim está disponible públicamente en Zenodo:

```
https://doi.org/10.5281/zenodo.14373749
```

Descargar `PyCrim_dataset.zip` y colocar en Google Drive en la ruta configurada en los scripts, o ajustar `ZIP_PATH` según su entorno local.

### Ejecución

**Experimentos clásicos (búsqueda de configuración):**
```bash
python algoritmos_clasicos_70_15_15_busqueda64config.py
```

**Evaluación final en test (algoritmos clásicos):**
```bash
python algoritmos_clasicos_evaluacion_final_test.py
```

**Experimentos Transformer:**
```bash
python template_transformers_optimizado.py
```

> Los scripts están originalmente adaptados para Google Colab con Google Drive. Para ejecución local, ajustar las rutas de `ZIP_PATH` y `OUTPUT_DIR`.

---

## 📊 Hallazgos clave

1. **BETO supera a XLM-RoBERTa** (82.91% vs 82.42% F1-macro) a pesar de su menor capacidad paramétrica, lo que sugiere que la especialización monolingüe en español puede ser ventajosa en dominios jurídicos con terminología formulaica.

2. **Sensibilidad al protocolo:** BETO mejora de 80.40% a 82.91% de F1-macro al pasar del protocolo base al optimizado, evidenciando que los Transformer requieren mayor calibración en corpus de tamaño reducido.

3. **Ventaja marginal sobre los clásicos:** Bajo el protocolo simétrico adoptado, la diferencia entre BETO y LR es de solo 0.11 puntos de F1-macro en el conjunto de prueba.

4. **Errores con diferente costo práctico:** BETO detecta el 74.83% de los casos "Hace lugar" (vs 55.24% de SVM), reduciendo a casi la mitad los falsos negativos, lo que es especialmente relevante en el contexto judicial.

---

## 🔭 Trabajo futuro

- **Pruebas estadísticas de significancia** (Bootstrap, McNemar) sobre las diferencias observadas en prueba
- **Modelos para documentos largos:** Longformer, Lawformer (sin necesidad de fragmentación)
- **Extensión a otras salas** de la CSJ (Civil, Constitucional) y otras jurisdicciones hispanohablantes
- **Interpretabilidad avanzada** con SHAP (SHapley Additive exPlanations)

---

## 📚 Referencias principales

- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL-HLT.
- Cañete et al. (2020). *BETO: Spanish BERT.* PML4DC @ ICLR.
- Conneau et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* ACL.
- Chalkidis et al. (2020). *LEGAL-BERT.* Findings of EMNLP.
- Saito & Rehmsmeier (2015). *The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets.* PLOS ONE.

---

## ✍️ Autores

- **Blanca Franco** 
- **Fátima Ferreira** 
- **Cristian Amarilla Closs** 
- **Helena Gómez-Adorno** 
- **Romina Rojas Moreno**

---

## 📄 Licencia

Este repositorio se distribuye bajo la licencia MIT. Consulte el archivo [LICENSE](LICENSE) para más detalles.

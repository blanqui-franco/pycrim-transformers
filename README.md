# PyCrim Experiments — Predicción de Sentencias Penales en Paraguay

> **Evaluación Comparativa de Algoritmos Clásicos y Modelos Transformer para la Predicción de Sentencias de la Sala Penal de la Corte Suprema de Justicia de Paraguay**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: Zenodo](https://img.shields.io/badge/Dataset-Zenodo-1D91C0.svg)](https://doi.org/10.5281/zenodo.14373749)

---

## Resumen

Primera evaluación comparativa formal entre algoritmos clásicos de aprendizaje supervisado y modelos Transformer preentrenados para la predicción de sentencias penales en Paraguay, sobre el corpus público **PyCrim** (5.000 resoluciones, 2011–2023).

**Corpus:** Gómez Adorno, H. et al. (2024). *Dataset of the criminal chamber cases from the Supreme Court of Justice of Paraguay*. Zenodo. [https://doi.org/10.5281/zenodo.14373749](https://doi.org/10.5281/zenodo.14373749)

---

## Modelos evaluados

| Familia | Modelos |
|---|---|
| **Clásicos** | Logistic Regression, SVM, Random Forests, Naive Bayes |
| **Transformer** | BETO, mBERT, XLM-RoBERTa Legal Spanish LongFormer (con y sin fine-tuning) |

**Protocolo experimental (simétrico):** partición estratificada 70/15/15, compensación de desbalanceo por pesos balanceados, optimización de umbral sobre validación, métrica principal: F1-macro.

---

## Resultados principales

| Modelo | Configuración | F1-macro (Test) |
|---|---|---|
| **BETO** (fine-tuning, lr=5e-6) | Protocolo optimizado | **82.91%** |
| Logistic Regression | 4-gramas, TF, sin stopwords, stemming | 82.80% |
| XLM-RoBERTa (fine-tuning, lr=1e-5) | Protocolo optimizado | 82.42% |
| SVM *(seleccionado por validación)* | 4-gramas, TF-IDF, sin stopwords, stemming | 78.78% |
| mBERT (fine-tuning, lr=1e-5) | Protocolo optimizado | 77.87% |

> SVM fue seleccionado como representante clásico según criterio de validación pre-establecido (F1-macro val = 81.92%). LR obtuvo mayor F1 en prueba (82.80%), reduciendo la diferencia con BETO a 0.11 puntos.

---

## Estructura del repositorio

```
pycrim-transformers/
├── experiments/
│   ├── EXP-ALGORITMOS CLASICOS/
│   └── EXP-TRANSFORMER/
├── algoritmos_clasicos_70_15_15_busqueda64config.py
├── algoritmos_clasicos_evaluacion_final_test.py
└── template_transformers_optimizado.py
```

---

## Reproducibilidad

### Requisitos

```bash
pip install torch transformers scikit-learn pandas numpy nltk matplotlib
```

### Datos

Descargar `PyCrim_dataset.zip` desde [https://doi.org/10.5281/zenodo.14373749](https://doi.org/10.5281/zenodo.14373749) y ajustar `ZIP_PATH` en los scripts.

### Ejecución

```bash
# Búsqueda de configuración (clásicos)
python algoritmos_clasicos_70_15_15_busqueda64config.py

# Evaluación final en test (clásicos)
python algoritmos_clasicos_evaluacion_final_test.py

# Experimentos Transformer
python template_transformers_optimizado.py
```

> Los scripts están adaptados para Google Colab con Google Drive. Para ejecución local, ajustar `ZIP_PATH` y `OUTPUT_DIR`.

---

## Hallazgos clave

1. **BETO supera a XLM-RoBERTa** (82.91% vs 82.42%) pese a su menor capacidad paramétrica, sugiriendo ventajas de la especialización monolingüe en dominios jurídicos.
2. **Ventaja marginal sobre los clásicos:** la diferencia entre BETO y LR es de solo 0.11 puntos de F1-macro bajo el protocolo simétrico.
3. **Sensibilidad al protocolo:** BETO mejora de 80.40% a 82.91% al pasar del protocolo base al optimizado.
4. **Falsos negativos reducidos:** BETO detecta el 74.83% de los casos "Hace lugar" vs 55.24% de SVM, lo cual es relevante en el contexto judicial.

---

## Licencia

Distribuido bajo la licencia MIT. Consulte el archivo [LICENSE](LICENSE) para más detalles.

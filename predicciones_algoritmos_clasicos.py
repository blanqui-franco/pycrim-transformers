# DESCARGAMOS Y CREAMOS EL DATAFRAME
import zipfile
import io
import numpy as np
import pandas as pd
import csv
import os
import sys

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

archivozip = "PrediccionSentencias/Corpus/corpus.txt.zip"
with zipfile.ZipFile(archivozip, "r") as zipref:
    datos = {nombre: zipref.read(nombre) for nombre in zipref.namelist()}
datos = io.BytesIO(datos["corpus_final_corregido.txt"])
data = pd.read_csv(datos, sep="\t")
print(data.shape)

# SEPARAMOS LOS DATOS DE ENTRENAMIENTO Y PRUEBA
nltk.download("punkt")
X_train, X_test, y_train, y_test = train_test_split(
    data["Contenido Txt"],
    data["Resultado binario de la acción"],
    random_state=0
)

# DESCARGAR STOP WORDS Y STEMMING
nltk.download("stopwords")

# Función de stemming
stemmer = SnowballStemmer("spanish")
spanish_stopwords = stopwords.words("spanish")

# Aplicar stemming a las stopwords
spanish_stopwords_stemmed = [stemmer.stem(word) for word in spanish_stopwords]


# Función de stemming de los tokens
def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]


# Función para tokenizar y aplicar stemming con stopwords
def tokenize_and_stem_stop(text):
    sw = stopwords.words("spanish")
    tokens = nltk.word_tokenize(text.lower())
    tokens = [w for w in tokens if w not in sw]
    return stem_tokens(tokens)


# Función para tokenizar y aplicar stemming sin stopwords
def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    return stem_tokens(tokens)


# VECTORIZACIÓN. EJECUTAR MODELOS. ALMACENAR RESULTADOS.
def vectorizar(X_train_var, X_test_var, y_train_var, y_test_var,
               param0, param1, param2, param3, param4):
    """
    Args:
        X_train_var: datos de entrenamiento
        X_test_var:  datos de prueba
        y_train_var: etiquetas de entrenamiento
        y_test_var:  etiquetas de prueba
        param0: ngramas 1,2,3,4
        param1: stopwords 0,1
        param2: stemming 0,1
        param3: min_df 2,3
        param4: CountVectorizer=0 / TfidfVectorizer=1
    Returns:
        Una lista con los resultados de los modelos
    """
    token_pattern_var = r"(?u)\b\w+\b"  # equivalente a r(?u)\b\w+\b del original
    stop_word_var = None

    if param1 == 0 and param2 == 0:
        # sin sw y sin stemming
        tokenizer_var = None
    elif param1 == 1 and param2 == 0:
        # con sw y sin stemming
        tokenizer_var = None
        stop_word_var = spanish_stopwords
    elif param1 == 0 and param2 == 1:
        # sin sw y con stemming
        tokenizer_var = tokenize_and_stem
        token_pattern_var = None
    else:
        # con sw y con stemming
        tokenizer_var = tokenize_and_stem_stop
        token_pattern_var = None

    params_dict = {
        "ngram_range": (param0, param0),
        "stop_words": stop_word_var,
        "tokenizer": tokenizer_var,
        "token_pattern": token_pattern_var,
        "min_df": param3,
    }

    if param4 == 0:
        vect = CountVectorizer(**params_dict)
    else:
        vect = TfidfVectorizer(**params_dict)

    X_train_vectorized = vect.fit_transform(X_train_var)
    X_test_vectorized = vect.transform(X_test_var)

    r1 = naives_multinomial(X_train_vectorized, X_test_vectorized, y_train_var, y_test_var)
    r2 = logistic_regression(X_train_vectorized, X_test_vectorized, y_train_var, y_test_var)
    r3 = random_forest(X_train_vectorized, X_test_vectorized, y_train_var, y_test_var)
    r4 = support_vector_machine(X_train_vectorized, X_test_vectorized, y_train_var, y_test_var)

    return [r1, r2, r3, r4]


def naives_multinomial(X_train_vectorized, X_test_vectorized, y_train, y_test):
    # Calcular las probabilidades de clase
    class_counts = pd.Series(y_train).value_counts()
    class_prior = class_counts / class_counts.sum()
    # Entrenamiento del modelo con probabilidades de clase
    clf = MultinomialNB(alpha=0.1, class_prior=class_prior.tolist())
    clf.fit(X_train_vectorized, y_train)
    # Predicciones
    predictions = clf.predict(X_test_vectorized)
    return metricas(y_test, predictions)


def logistic_regression(X_train_vectorized, X_test_vectorized, y_train, y_test):
    # Ajuste del modelo de Regresión Logística
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_vectorized, y_train)
    # Predicciones
    predictions = clf.predict(X_test_vectorized)
    return metricas(y_test, predictions)


def random_forest(X_train_vectorized, X_test_vectorized, y_train, y_test):
    # Ajuste del modelo de Random Forest
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    clf.fit(X_train_vectorized, y_train)
    # Predicciones sobre el conjunto de prueba
    predictions = clf.predict(X_test_vectorized)
    return metricas(y_test, predictions)


def support_vector_machine(X_train_vectorized, X_test_vectorized, y_train, y_test):
    # Escalar los datos (with_mean=False para matrices dispersas)
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train_vectorized)
    X_test_scaled = scaler.transform(X_test_vectorized)
    # Ajuste del modelo SVM con un kernel lineal
    clf = SVC(kernel="linear", max_iter=-1)
    clf.fit(X_train_scaled, y_train)
    # Predicciones sobre el conjunto de prueba
    predictions = clf.predict(X_test_scaled)
    return metricas(y_test, predictions)


def metricas(y_test, predictions):
    scores = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="macro", zero_division=1),
        "recall": recall_score(y_test, predictions, average="macro"),
        "f1_score": f1_score(y_test, predictions, average="macro"),
    }
    return scores


# ITERAMOS PARÁMETROS PARA GENERAR VECTORES Y REALIZAR PREDICCIONES
# Definir la ruta base
base_path = "/content/gdrive/MyDrive/PrediccionSentencias/Códigos/Cristian/Predicción/"
df_indices = pd.read_csv(os.path.join(base_path, "parametros4.csv"), sep=",")

# Verificar si el archivo de resultados ya existe
resultado_file = os.path.join(base_path, "resultados4.csv")
if os.path.exists(resultado_file):
    df_resultados = pd.read_csv(resultado_file)
else:
    # Definir las columnas del DataFrame
    columnas = [
        "ng", "sw", "st", "min", "Tfidf",
        "nv_a", "nv_p", "nv_r", "nv_f1",
        "lg_a", "lg_p", "lg_r", "lg_f1",
        "rf_a", "rf_p", "rf_r", "rf_f1",
        "svm_a", "svm_p", "svm_r", "svm_f1",
    ]
    # Crear el DataFrame vacío
    df_resultados = pd.DataFrame(columns=columnas)

# Iterar sobre el archivo de índices
for index, row in df_indices.iterrows():
    # Verificar si el registro ya tiene un resultado
    already_computed = (
        (df_resultados["ng"] == row["ngram"]) &
        (df_resultados["sw"] == row["sw"]) &
        (df_resultados["st"] == row["st"]) &
        (df_resultados["min"] == row["min"]) &
        (df_resultados["Tfidf"] == row["Tfidf"])
    ).any()

    if not already_computed:
        param0 = row["ngram"]
        param1 = row["sw"]
        param2 = row["st"]
        param3 = row["min"]
        param4 = row["Tfidf"]

        # Ejecutar la función con los parámetros actuales
        resultado = vectorizar(X_train, X_test, y_train, y_test,
                               param0, param1, param2, param3, param4)
        r1, r2, r3, r4 = resultado

        # Crear una lista con los resultados del registro actual
        nuevo_registro = pd.DataFrame([[
            param0, param1, param2, param3, param4,
            r1["accuracy"], r1["precision"], r1["recall"], r1["f1_score"],
            r2["accuracy"], r2["precision"], r2["recall"], r2["f1_score"],
            r3["accuracy"], r3["precision"], r3["recall"], r3["f1_score"],
            r4["accuracy"], r4["precision"], r4["recall"], r4["f1_score"],
        ]], columns=df_resultados.columns)

        # Concatenar el nuevo registro con el DataFrame de resultados
        df_resultados = pd.concat([df_resultados, nuevo_registro], ignore_index=True)

        # Guardar el DataFrame actualizado en cada iteración
        df_resultados.to_csv(resultado_file, index=False)
        print(resultado)
        print(f"Resultado para la fila {index} almacenado.")

print("Proceso completado. Todos los resultados han sido almacenados.")
import os
import pickle
import numpy as np
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# ── Configuración de base de datos ──────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "NuevaContraseña",
    "database": "algoritmico-ia"
}

DATOS_DIR = os.path.join(os.path.dirname(__file__), "datos-ia")
os.makedirs(DATOS_DIR, exist_ok=True)

MODELO_PATH       = os.path.join(DATOS_DIR, "modelo.pkl")
VECTORIZADOR_PATH = os.path.join(DATOS_DIR, "vectorizacion.pkl")


def obtener_datos():
    """Trae todas las preguntas y respuestas de la base de datos."""
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, pregunta, respuesta, categoria FROM hechos_verificados")
    filas  = cursor.fetchall()
    cursor.close()
    conn.close()
    return filas


def entrenar(filas):
    """
    Entrena un clasificador TF-IDF + Regresión Logística.
    Cada fila de la BD se convierte en un texto combinado (pregunta + respuesta)
    y se le asigna su propio ID como etiqueta. En consulta, buscamos el ID
    más probable y devolvemos su respuesta desde la BD.
    """
    if len(filas) < 2:
        print("[ADVERTENCIA] Se necesitan al menos 2 registros para entrenar.")
        return None, None

    textos    = [f"{f['pregunta']} {f['respuesta']}" for f in filas]
    etiquetas = [f["id"] for f in filas]

    vectorizador = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )

    modelo = LogisticRegression(
        max_iter=500,
        C=1.0,
        solver="lbfgs"
    )

    X = vectorizador.fit_transform(textos)
    modelo.fit(X, etiquetas)

    # Guardar artefactos
    with open(VECTORIZADOR_PATH, "wb") as f:
        pickle.dump(vectorizador, f)
    with open(MODELO_PATH, "wb") as f:
        pickle.dump(modelo, f)

    print(f"[OK] Modelo entrenado con {len(filas)} registros.")
    print(f"     → {VECTORIZADOR_PATH}")
    print(f"     → {MODELO_PATH}")
    return vectorizador, modelo


def graficar_distribucion(filas):
    """Genera un gráfico de barras con la cantidad de registros por categoría."""
    categorias = {}
    for f in filas:
        cat = f.get("categoria") or "sin categoría"
        categorias[cat] = categorias.get(cat, 0) + 1

    nombres = list(categorias.keys())
    valores = list(categorias.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    colores = plt.cm.Blues(np.linspace(0.4, 0.85, len(nombres)))
    bars = ax.barh(nombres, valores, color=colores, edgecolor="white", height=0.5)

    for bar, val in zip(bars, valores):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=10, color="#333")

    ax.set_xlabel("Cantidad de registros")
    ax.set_title("Distribución del dataset por categoría", fontsize=13, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(valores) + 2)

    out = os.path.join(DATOS_DIR, "distribucion.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[OK] Gráfico guardado en {out}")


if __name__ == "__main__":
    print("=== ENTRENAMIENTO INICIAL ===")
    filas = obtener_datos()
    if not filas:
        print("[ERROR] La base de datos no tiene registros. Inserta datos primero.")
    else:
        print(f"[INFO] {len(filas)} registros encontrados.")
        entrenar(filas)
        graficar_distribucion(filas)
        print("=== LISTO ===")
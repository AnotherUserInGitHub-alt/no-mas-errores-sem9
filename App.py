import os
import pickle
import mysql.connector
from flask import Flask, render_template, request, jsonify

from entrenar import obtener_datos, entrenar, DB_CONFIG

app = Flask(__name__)

DATOS_DIR         = os.path.join(os.path.dirname(__file__), "datos-ia")
MODELO_PATH       = os.path.join(DATOS_DIR, "modelo.pkl")
VECTORIZADOR_PATH = os.path.join(DATOS_DIR, "vectorizacion.pkl")

UMBRAL_CONFIANZA = 0.30


# ── Helpers ──────────────────────────────────────────────────────────────────

def cargar_modelo():
    if not os.path.exists(MODELO_PATH) or not os.path.exists(VECTORIZADOR_PATH):
        return None, None
    with open(VECTORIZADOR_PATH, "rb") as f:
        vectorizador = pickle.load(f)
    with open(MODELO_PATH, "rb") as f:
        modelo = pickle.load(f)
    return vectorizador, modelo


def buscar_en_bd_por_id(id_hecho):
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT pregunta, respuesta, categoria, fuente FROM hechos_verificados WHERE id = %s",
        (int(id_hecho),)
    )
    fila = cursor.fetchone()
    cursor.close()
    conn.close()
    return fila


def buscar_por_keywords(texto):
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT pregunta, respuesta, categoria, fuente,
                   MATCH(pregunta, respuesta) AGAINST (%s IN NATURAL LANGUAGE MODE) AS score
            FROM hechos_verificados
            WHERE MATCH(pregunta, respuesta) AGAINST (%s IN NATURAL LANGUAGE MODE)
            ORDER BY score DESC
            LIMIT 1
            """,
            (texto, texto)
        )
        fila = cursor.fetchone()
    except mysql.connector.Error:
        palabras = texto.strip().split()
        condiciones = " OR ".join(
            ["pregunta LIKE %s OR respuesta LIKE %s"] * len(palabras)
        )
        valores = []
        for p in palabras:
            valores += [f"%{p}%", f"%{p}%"]
        cursor.execute(
            f"SELECT pregunta, respuesta, categoria, fuente FROM hechos_verificados WHERE {condiciones} LIMIT 1",
            valores
        )
        fila = cursor.fetchone()
    cursor.close()
    conn.close()
    return fila


def consultar(texto):
    vectorizador, modelo = cargar_modelo()
    fuente_respuesta = None
    resultado = None

    if vectorizador and modelo:
        X = vectorizador.transform([texto])
        probas = modelo.predict_proba(X)[0]
        confianza = float(probas.max())
        id_predicho = modelo.classes_[probas.argmax()]

        if confianza >= UMBRAL_CONFIANZA:
            fila = buscar_en_bd_por_id(id_predicho)
            if fila:
                resultado = fila
                fuente_respuesta = "modelo"

    if resultado is None:
        fila = buscar_por_keywords(texto)
        if fila:
            resultado = fila
            fuente_respuesta = "keywords"

    return resultado, fuente_respuesta


# ── Rutas ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/consultar", methods=["POST"])
def ruta_consultar():
    datos = request.get_json(silent=True) or {}
    texto = datos.get("consulta", "").strip()

    if not texto:
        return jsonify({"error": "La consulta no puede estar vacía."}), 400

    resultado, fuente = consultar(texto)

    if resultado:
        return jsonify({
            "encontrado": True,
            "pregunta":   resultado["pregunta"],
            "respuesta":  resultado["respuesta"],
            "categoria":  resultado.get("categoria", ""),
            "fuente":     resultado.get("fuente", ""),
            "via":        fuente
        })
    else:
        return jsonify({
            "encontrado": False,
            "mensaje": (
                "No tengo información verificada sobre eso. "
                "Puedes agregar el hecho en el formulario de abajo."
            )
        })


@app.route("/agregar", methods=["POST"])
def ruta_agregar():
    datos     = request.get_json(silent=True) or {}
    pregunta  = datos.get("pregunta", "").strip()
    respuesta = datos.get("respuesta", "").strip()
    categoria = datos.get("categoria", "general").strip()
    fuente    = datos.get("fuente", "").strip()

    if not pregunta or not respuesta:
        return jsonify({"error": "Pregunta y respuesta son obligatorias."}), 400

    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO hechos_verificados (pregunta, respuesta, categoria, fuente) VALUES (%s, %s, %s, %s)",
        (pregunta, respuesta, categoria, fuente)
    )
    conn.commit()
    nuevo_id = cursor.lastrowid
    cursor.close()
    conn.close()

    filas = obtener_datos()
    entrenar(filas)

    return jsonify({
        "ok":      True,
        "mensaje": f"Hecho #{nuevo_id} agregado y modelo re-entrenado con {len(filas)} registros.",
        "total":   len(filas)
    })


@app.route("/listar")
def ruta_listar():
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, pregunta, categoria, fuente FROM hechos_verificados ORDER BY id DESC")
    filas  = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template("listar.html", filas=filas, mensaje=None)


@app.route("/editar/<int:id>", methods=["GET", "POST"])
def ruta_editar(id):
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)

    if request.method == "POST":
        pregunta  = request.form.get("pregunta", "").strip()
        respuesta = request.form.get("respuesta", "").strip()
        categoria = request.form.get("categoria", "general").strip()
        fuente    = request.form.get("fuente", "").strip()

        cursor.execute(
            "UPDATE hechos_verificados SET pregunta=%s, respuesta=%s, categoria=%s, fuente=%s WHERE id=%s",
            (pregunta, respuesta, categoria, fuente, id)
        )
        conn.commit()
        cursor.close()
        conn.close()

        filas = obtener_datos()
        entrenar(filas)

        return render_template("editar.html", fila=None, mensaje="Registro actualizado y modelo re-entrenado.")

    cursor.execute("SELECT * FROM hechos_verificados WHERE id=%s", (id,))
    fila = cursor.fetchone()
    cursor.close()
    conn.close()
    return render_template("editar.html", fila=fila, mensaje=None)


@app.route("/eliminar/<int:id>", methods=["POST"])
def ruta_eliminar(id):
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM hechos_verificados WHERE id=%s", (id,))
    conn.commit()
    cursor.close()
    conn.close()

    filas = obtener_datos()
    entrenar(filas)

    conn2   = mysql.connector.connect(**DB_CONFIG)
    cursor2 = conn2.cursor(dictionary=True)
    cursor2.execute("SELECT id, pregunta, categoria, fuente FROM hechos_verificados ORDER BY id DESC")
    filas_actualizadas = cursor2.fetchall()
    cursor2.close()
    conn2.close()

    return render_template("listar.html", filas=filas_actualizadas, mensaje=f"Registro #{id} eliminado y modelo re-entrenado.")


@app.route("/stats")
def ruta_stats():
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT categoria, COUNT(*) as total FROM hechos_verificados GROUP BY categoria ORDER BY total DESC"
    )
    categorias = cursor.fetchall()
    cursor.execute("SELECT COUNT(*) as total FROM hechos_verificados")
    total = cursor.fetchone()["total"]
    cursor.close()
    conn.close()
    return jsonify({"total": total, "categorias": categorias})


if __name__ == "__main__":
    app.run(debug=True)
"""
app.py — Servidor Flask
Sistema de Clasificación de Tickets de Soporte
Universidad Rafael Landívar · IA 2026

Dataset: Bitext Customer Support LLM Chatbot (11 categorías)

Endpoints:
    GET  /            → Interfaz web principal
    POST /classify    → Clasificar un ticket
    GET  /metrics     → Métricas del modelo (K-Folds)
    GET  /tickets     → Historial de tickets clasificados
    POST /retrain     → Reentrenar el modelo
"""

import os
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from naive_bayes import load_model, train_and_evaluate, NaiveBayesClassifier

app = Flask(__name__, static_folder="static", template_folder="templates")

# ══════════════════════════════════════════════════════════════
#  INICIALIZACIÓN — cargar o entrenar modelo al arrancar
# ══════════════════════════════════════════════════════════════

MODEL_PATH   = "model.pkl"
METRICS_PATH = "metrics.json"


def get_model() -> NaiveBayesClassifier:
    """
    Carga el modelo desde disco si existe.
    Si no, entrena uno nuevo con el dataset sintético.
    """
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    print("[!] model.pkl no encontrado. Entrenando modelo automáticamente...")
    model, _ = train_and_evaluate(save_path=MODEL_PATH)
    return model


model = get_model()

# Cargar métricas guardadas (si existen)
metrics_summary = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        metrics_summary = json.load(f)

# Historial de tickets en memoria (últimos 200)
ticket_history = []


# ══════════════════════════════════════════════════════════════
#  RUTAS
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Sirve la interfaz web principal."""
    return send_from_directory("templates", "index.html")


@app.route("/classify", methods=["POST"])
def classify():
    """
    Clasifica un ticket de soporte.

    Request JSON:
        {
            "subject":     "Texto del asunto (opcional)",
            "description": "Descripción del problema"
        }

    Response JSON:
        {
            "ticket_id":     "TKT-XXXXXX",
            "subject":       "...",
            "description":   "...",
            "category":      "BILLING",
            "probabilities": {"BILLING": 82.3, "PAYMENT": 10.1, ...},
            "timestamp":     "2026-04-15 10:22:00"
        }
    """
    data        = request.get_json(force=True)
    subject     = data.get("subject",     "").strip()
    description = data.get("description", "").strip()

    if not description and not subject:
        return jsonify({"error": "Se requiere al menos una descripción."}), 400

    full_text = f"{subject} {description}".strip()

    try:
        category, probs = model.predict_text(full_text)
    except Exception as e:
        return jsonify({"error": f"Error en clasificación: {str(e)}"}), 500

    ticket = {
        "ticket_id":     f"TKT-{uuid.uuid4().hex[:6].upper()}",
        "subject":       subject,
        "description":   description,
        "category":      category,
        "probabilities": probs,
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    ticket_history.append(ticket)
    # Mantener solo los últimos 200 tickets en memoria
    if len(ticket_history) > 200:
        ticket_history.pop(0)

    return jsonify(ticket)


@app.route("/metrics")
def get_metrics():
    """
    Retorna las métricas de evaluación del modelo.
    Incluye resultados de K-Folds, accuracy, macro F1,
    métricas por clase y matriz de confusión.
    """
    return jsonify(metrics_summary)


@app.route("/tickets")
def get_tickets():
    """Retorna el historial de los últimos 50 tickets clasificados."""
    return jsonify(ticket_history[-50:])


@app.route("/retrain", methods=["POST"])
def retrain():
    """
    Reentrenar el modelo.

    Request JSON (opcional):
        { "csv_path": "ruta/al/bitext_dataset.csv" }

    Si no se especifica csv_path, usa el dataset sintético.
    """
    global model, metrics_summary
    data     = request.get_json(force=True) or {}
    csv_path = data.get("csv_path")

    try:
        model, summary = train_and_evaluate(
            dataset_path=csv_path,
            save_path=MODEL_PATH
        )
        metrics_summary = summary
        return jsonify({
            "status":   "Modelo reentrenado exitosamente",
            "accuracy": summary["avg_accuracy"],
            "macro_f1": summary["avg_macro_f1"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  HelpDesk AI — Servidor Flask")
    print("  Universidad Rafael Landívar · IA 2026")
    print("=" * 50)
    print("  URL: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)

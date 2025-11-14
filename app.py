"""
===============================================================
  AgroScan / SaccharumVision - Backend Flask
  Sistema de análisis de enfermedades en caña de azúcar usando IA
===============================================================
"""

# ------------------------------
# Importaciones principales
# ------------------------------
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import logging
import os
import sys

# ------------------------------
# Configuración de rutas del proyecto
# (permite importar módulos desde utils/)
# ------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------
# Importación del gestor de modelos
# ------------------------------
try:
    from utils.model_manager import ModelManager
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Advertencia: No se pudo importar ModelManager: {e}")
    MODEL_AVAILABLE = False

# ------------------------------
# Configuración de la aplicación Flask
# ------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "agroscan_key_2024"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

# ------------------------------
# Configuración del sistema de logs
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Crear carpetas necesarias
# ------------------------------
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

# ------------------------------
# Extensiones permitidas
# ------------------------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

# ------------------------------
# Configuración de modelos disponibles
# ------------------------------
MODELS_CONFIG = {
    "ResNet50": {
        "path": "models/ResNet50/ResNet50_latest.keras",
        "classes": "models/ResNet50/classes_latest.json",
        "size": (224, 224),
        "description": "Equilibrado y confiable"
    },
    "EfficientNetB0": {
        "path": "models/EfficientNetB0/EfficientNetB0_latest.keras",
        "classes": "models/EfficientNetB0/classes_latest.json",
        "size": (256, 256),
        "description": "Eficiente y preciso"
    },
    "MobileNetV2": {
        "path": "models/MobileNetV2/MobileNetV2_latest.keras",
        "classes": "models/MobileNetV2/classes_classes.json",
        "size": (256, 256),
        "description": "Rápido y ligero"
    }
}

# ------------------------------
# Cache de modelos cargados
# ------------------------------
loaded_models = {}

# ------------------------------
# Función para obtener o cargar un modelo
# ------------------------------
def get_model_manager(model_name="ResNet50"):
    """
    Devuelve el gestor de un modelo ya cargado o lo carga si no existe en caché.
    """
    if model_name not in MODELS_CONFIG:
        logger.warning(f"Modelo {model_name} no existe. Usando ResNet50 por defecto.")
        model_name = "ResNet50"

    # Si ya está cargado, devolverlo
    if model_name in loaded_models:
        logger.info(f"♻️ Usando modelo {model_name} desde caché")
        return loaded_models[model_name]

    # Cargar modelo nuevo
    try:
        config = MODELS_CONFIG[model_name]
        logger.info(f"📦 Cargando modelo {model_name}...")

        model_manager = ModelManager(
            model_path=config["path"],
            classes_path=config["classes"],
            img_size=config["size"],
            model_type=model_name  # NUEVO: Pasar tipo de modelo
        )

        loaded_models[model_name] = model_manager
        logger.info(f"✅ Modelo {model_name} cargado correctamente")

        return model_manager

    except Exception as e:
        logger.error(f"❌ Error al cargar modelo {model_name}: {e}")
        return None

# ------------------------------
# Inicializar modelo por defecto
# ------------------------------
model_manager = get_model_manager("ResNet50") if MODEL_AVAILABLE else None

# ------------------------------
# Utilidad: validar extensión de archivo
# ------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ==============================================================
#                   RUTAS DE PÁGINAS (HTML)
# ==============================================================

@app.route("/")
def index(): return render_template("index.html")

@app.route("/camera")
def camera(): return render_template("camera.html")

@app.route("/history")
def history(): return render_template("history.html")

@app.route("/history_details")
def history_details(): return render_template("history_details.html")

@app.route("/results")
def results(): return render_template("results.html")

@app.route("/settings")
def settings(): return render_template("settings.html")

@app.route("/terms")
def terms(): return render_template("terms.html")


# ==============================================================
#                       ENDPOINT: PREDICCIÓN
# ==============================================================

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Endpoint para analizar imágenes y predecir enfermedades.
    Acepta archivos enviados como 'image' o 'file' en FormData.
    """
    try:
        # Obtener archivo enviado
        file = request.files.get('image') or request.files.get('file')

        if not file or file.filename == "":
            return jsonify({"success": False, "error": "No se envió una imagen válida"}), 400

        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Formato no permitido"}), 400

        # Guardar imagen con timestamp
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        logger.info(f"📸 Imagen guardada: {filename}")

        # Leer parámetros extra
        use_tta = request.form.get('use_tta', 'false').lower() == 'true'
        selected_model = request.form.get('model', 'ResNet50')

        # Obtener modelo seleccionado
        current_model = get_model_manager(selected_model)

        # Si el modelo está disponible → analizar imagen
        if current_model:
            try:
                logger.info(f"🔍 Ejecutando predicción con {selected_model} (TTA={use_tta})")

                result = current_model.improved_predict(
                    image_path=filepath,
                    use_tta=use_tta,
                    threshold=0.50,
                    num_augmentations=8 if use_tta else 0
                )

                if result and result.get('status') in ['success', 'warning']:
                    return jsonify({
                        "success": True,
                        "filename": filename,
                        "timestamp": timestamp,
                        "prediction": {
                            "class": result['class'],
                            "confidence": result['confidence'],
                            "probability": result.get('probability', result['confidence'] / 100),
                            "method": result.get('method', "Predicción estándar"),
                            "top_3": result.get('top_3', []),
                            "all_probabilities": result.get('probabilities', {})
                        },
                        "status": result['status'],
                        "tta_used": use_tta,
                        "model_used": selected_model
                    })

                return jsonify({"success": False, "error": result.get("message")}), 500

            except Exception as e:
                logger.error(f"❌ Error en la predicción: {e}")
                return jsonify({"success": False, "error": "Error analizando imagen"}), 500

        # Si no hay modelo, ejecutar modo simulación
        mock_predictions = [
            {"class": "Healthy", "confidence": 85.5, "probability": 0.855},
            {"class": "Mosaic", "confidence": 10.2, "probability": 0.102},
            {"class": "Rust", "confidence": 4.3, "probability": 0.043},
        ]

        return jsonify({
            "success": True,
            "filename": filename,
            "prediction": mock_predictions[0],
            "status": "success",
            "message": "Simulación realizada (modelo no cargado)"
        })

    except Exception as e:
        logger.error(f"❌ Error interno: {e}")
        return jsonify({"success": False, "error": "Error interno del servidor"}), 500


# ==============================================================
#                    ENDPOINTS AUXILIARES
# ==============================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Devuelve archivos almacenados en /uploads."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/static/<path:filename>")
def static_files(filename):
    """Devuelve archivos estáticos."""
    return send_from_directory("static", filename)

@app.route("/api/health")
def health():
    """Estado general del servidor."""
    return jsonify({
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_manager is not None,
        "models_available": list(MODELS_CONFIG.keys()),
        "version": "1.0.0"
    })

@app.route("/api/models")
def available_models():
    """Lista completa de modelos disponibles."""
    models_info = {
        name: {
            "name": name,
            "description": cfg["description"],
            "loaded": name in loaded_models,
            "available": os.path.exists(cfg["path"])
        }
        for name, cfg in MODELS_CONFIG.items()
    }

    return jsonify({
        "success": True,
        "models": models_info,
        "default": "ResNet50"
    })

@app.route("/api/info")
def app_info():
    """Información del sistema y sus capacidades."""
    return jsonify({
        "name": "AgroScan / SaccharumVision",
        "description": "Sistema de análisis de enfermedades en plantas usando IA",
        "version": "1.0.0",
        "model_loaded": model_manager is not None,
        "endpoints": {
            "pages": ["/", "/camera", "/history", "/results"],
            "api": ["/api/predict", "/api/health", "/api/info"]
        }
    })


# ==============================================================
#                     MANEJO DE ERRORES
# ==============================================================

@app.errorhandler(404)
def not_found(error):
    return render_template("index.html"), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Error 500: {error}")
    return jsonify({"error": "Error interno del servidor"}), 500


# ==============================================================
#                   EJECUCIÓN DEL SERVIDOR
# ==============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Iniciando AgroScan / SaccharumVision")
    print("=" * 60)
    print(f"📂 Uploads: {app.config['UPLOAD_FOLDER']}")
    print(f"🤖 Modelos disponibles:")
    for model_name, config in MODELS_CONFIG.items():
        status = "✅" if os.path.exists(config["path"]) else "❌"
        print(f"  {status} {model_name} — {config['description']}")
    print("🌐 http://localhost:5000")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)

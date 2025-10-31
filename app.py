"""
🍃 SaccharumVision - Aplicación Flask
====================================

Aplicación web Flask para análisis de imágenes de caña de azúcar
usando modelos de Deep Learning.

Autor: Sistema de Visión Agrónoma
Fecha: 2025
"""

import os
import json
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import logging

# Importar configuración y utilidades
from config.config import config, validate_paths
from utils.model_manager import ModelManager

# ============================
# INICIALIZACIÓN DE LA APP
# ============================
app = Flask(__name__)

# Cargar configuración (desarrollo por defecto)
env = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[env])

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================
# INICIALIZACIÓN DEL MODELO
# ============================
model_manager = None

def init_model():
    """
    Inicializa el modelo de predicción
    """
    global model_manager
    try:
        logger.info("🔄 Inicializando modelo de predicción...")
        model_manager = ModelManager(
            model_path=app.config['MODEL_PATH'],
            classes_path=app.config['CLASSES_PATH'],
            img_size=app.config['IMG_SIZE']
        )
        logger.info("✅ Modelo cargado exitosamente")
        return True
    except Exception as e:
        logger.error(f"❌ Error al cargar el modelo: {str(e)}")
        return False

# ============================
# FUNCIONES AUXILIARES
# ============================
def allowed_file(filename):
    """
    Verifica si la extensión del archivo es permitida
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ============================
# RUTAS DE LA APLICACIÓN
# ============================

@app.route('/')
def index():
    """
    Ruta principal - Muestra la página de inicio
    """
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    """
    Ruta de análisis - Muestra la página de análisis de imágenes
    """
    return render_template('analyze.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint para realizar predicciones usando TTA
    
    Espera:
    - POST request con una imagen en 'file'
    - Parámetros opcionales:
        * use_tta: (opcional) usar Test Time Augmentation (default: True)
        * threshold: (opcional) umbral de confianza (default: 0.70)
        * num_augmentations: (opcional) número de aumentaciones (default: 5)
    
    Retorna:
    - JSON con la predicción mejorada usando TTA
    """
    try:
        # Verificar que se envió un archivo
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No se envió ningún archivo'
            }), 400
        
        file = request.files['file']
        
        # Verificar que el archivo tiene nombre
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'El archivo no tiene nombre'
            }), 400
        
        # Verificar que el modelo está cargado
        if model_manager is None:
            return jsonify({
                'success': False,
                'error': 'El modelo no está cargado. Por favor reinicie el servidor.'
            }), 500
        
        # Verificar que el archivo es válido
        if file and allowed_file(file.filename):
            # Guardar el archivo de forma segura
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Obtener parámetros opcionales (por defecto usa TTA)
            use_tta = request.form.get('use_tta', 'true').lower() == 'true'
            threshold = float(request.form.get('threshold', 0.70))
            num_augmentations = int(request.form.get('num_augmentations', 5))
            
            logger.info(f"📷 Procesando imagen con TTA: {filename}")
            logger.info(f"   • TTA: {use_tta}, Threshold: {threshold}, Augmentations: {num_augmentations}")
            
            # Realizar predicción mejorada con TTA
            prediction = model_manager.improved_predict(
                image_path=filepath,
                use_tta=use_tta,
                threshold=threshold,
                num_augmentations=num_augmentations
            )
            
            # Preparar respuesta
            response = {
                'success': prediction['status'] in ['success', 'warning'],
                'status': prediction['status'],
                'filename': filename,
                'image_url': url_for('static', filename=f'../uploads/{filename}'),
                'prediction': {
                    'class': prediction.get('class', 'Unknown'),
                    'confidence': float(prediction.get('confidence', 0)),
                    'probability': float(prediction.get('probability', 0)),
                    'method': prediction.get('method', 'TTA'),
                    'message': prediction.get('message', ''),
                    'all_probabilities': {
                        k: float(v) for k, v in prediction.get('probabilities', {}).items()
                    },
                    'top_3': prediction.get('top_3', [])
                }
            }
            
            if prediction['status'] == 'success':
                logger.info(f"✅ Predicción TTA: {prediction['class']} ({prediction['confidence']:.2f}%)")
            elif prediction['status'] == 'warning':
                logger.warning(f"⚠️ Predicción con baja confianza: {prediction['class']} ({prediction['confidence']:.2f}%)")
            
            return jsonify(response), 200
        
        else:
            return jsonify({
                'success': False,
                'error': f'Tipo de archivo no permitido. Use: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
            }), 400
    
    except Exception as e:
        logger.error(f"❌ Error en predicción: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error al procesar la imagen: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """
    Endpoint de salud - Verifica que la API esté funcionando
    """
    model_loaded = model_manager is not None
    
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded,
        'version': '1.0.0'
    }), 200 if model_loaded else 503

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """
    Endpoint para obtener las clases disponibles
    """
    if model_manager is None:
        return jsonify({
            'success': False,
            'error': 'El modelo no está cargado'
        }), 500
    
    return jsonify({
        'success': True,
        'classes': model_manager.get_classes()
    }), 200

@app.route('/api/predict-improved', methods=['POST'])
def predict_improved():
    """
    API endpoint para realizar predicciones mejoradas con TTA
    
    Espera:
    - POST request con:
        * 'file': imagen a analizar
        * 'use_tta': (opcional) usar Test Time Augmentation (default: True)
        * 'threshold': (opcional) umbral de confianza (default: 0.70)
        * 'num_augmentations': (opcional) número de aumentaciones (default: 5)
    
    Retorna:
    - JSON con la predicción mejorada, top 3 y estado
    """
    try:
        # Verificar que se envió un archivo
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No se envió ningún archivo'
            }), 400
        
        file = request.files['file']
        
        # Verificar que el archivo tiene nombre
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'El archivo no tiene nombre'
            }), 400
        
        # Verificar que el modelo está cargado
        if model_manager is None:
            return jsonify({
                'success': False,
                'error': 'El modelo no está cargado. Por favor reinicie el servidor.'
            }), 500
        
        # Verificar que el archivo es válido
        if file and allowed_file(file.filename):
            # Guardar el archivo de forma segura
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Obtener parámetros opcionales
            use_tta = request.form.get('use_tta', 'true').lower() == 'true'
            threshold = float(request.form.get('threshold', 0.70))
            num_augmentations = int(request.form.get('num_augmentations', 5))
            
            logger.info(f"📷 Procesando imagen con predicción mejorada: {filename}")
            logger.info(f"   • TTA: {use_tta}, Threshold: {threshold}, Augmentations: {num_augmentations}")
            
            # Realizar predicción mejorada
            prediction = model_manager.improved_predict(
                image_path=filepath,
                use_tta=use_tta,
                threshold=threshold,
                num_augmentations=num_augmentations
            )
            
            # Preparar respuesta
            response = {
                'success': prediction['status'] in ['success', 'warning'],
                'status': prediction['status'],
                'filename': filename,
                'image_url': url_for('static', filename=f'../uploads/{filename}'),
                'prediction': {
                    'class': prediction.get('class', 'Unknown'),
                    'confidence': float(prediction.get('confidence', 0)),
                    'probability': float(prediction.get('probability', 0)),
                    'method': prediction.get('method', 'Unknown'),
                    'message': prediction.get('message', ''),
                    'all_probabilities': {
                        k: float(v) for k, v in prediction.get('probabilities', {}).items()
                    },
                    'top_3': prediction.get('top_3', [])
                }
            }
            
            if prediction['status'] == 'success':
                logger.info(f"✅ Predicción mejorada: {prediction['class']} ({prediction['confidence']:.2f}%)")
            elif prediction['status'] == 'warning':
                logger.warning(f"⚠️ Predicción con baja confianza: {prediction['class']} ({prediction['confidence']:.2f}%)")
            
            return jsonify(response), 200
        
        else:
            return jsonify({
                'success': False,
                'error': f'Tipo de archivo no permitido. Use: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
            }), 400
    
    except Exception as e:
        logger.error(f"❌ Error en predicción mejorada: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error al procesar la imagen: {str(e)}'
        }), 500

@app.route('/api/predict-top3', methods=['POST'])
def predict_top3():
    """
    API endpoint para obtener las 3 mejores predicciones
    
    Espera:
    - POST request con una imagen en 'file'
    
    Retorna:
    - JSON con el top 3 de predicciones
    """
    try:
        # Verificar que se envió un archivo
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No se envió ningún archivo'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'El archivo no tiene nombre'
            }), 400
        
        if model_manager is None:
            return jsonify({
                'success': False,
                'error': 'El modelo no está cargado'
            }), 500
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            logger.info(f"📷 Obteniendo Top-3 para: {filename}")
            
            # Obtener top 3
            top_3 = model_manager.get_top_3_predictions(filepath)
            
            if top_3 is None:
                return jsonify({
                    'success': False,
                    'error': 'Error al procesar la imagen'
                }), 500
            
            return jsonify({
                'success': True,
                'filename': filename,
                'image_url': url_for('static', filename=f'../uploads/{filename}'),
                'top_3': top_3
            }), 200
        
        else:
            return jsonify({
                'success': False,
                'error': f'Tipo de archivo no permitido'
            }), 400
    
    except Exception as e:
        logger.error(f"❌ Error en top-3: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================
# MANEJO DE ERRORES
# ============================

@app.errorhandler(404)
def not_found(error):
    """Manejo de error 404"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Manejo de error 500"""
    logger.error(f"Error 500: {str(error)}")
    return render_template('500.html'), 500

@app.errorhandler(413)
def too_large(error):
    """Manejo de archivos muy grandes"""
    return jsonify({
        'success': False,
        'error': 'El archivo es demasiado grande. Máximo 16MB'
    }), 413

# ============================
# PUNTO DE ENTRADA
# ============================

if __name__ == '__main__':
    # Validar rutas críticas
    logger.info("🔍 Validando rutas del proyecto...")
    path_errors = validate_paths()
    
    if path_errors:
        logger.error("❌ Errores encontrados:")
        for error in path_errors:
            logger.error(f"  - {error}")
        logger.error("Por favor corrija los errores antes de continuar")
    else:
        logger.info("✅ Todas las rutas validadas correctamente")
        
        # Inicializar modelo
        if init_model():
            logger.info(f"🚀 Iniciando servidor en http://{app.config['HOST']}:{app.config['PORT']}")
            logger.info("📊 Endpoints disponibles:")
            logger.info("  - GET  /                     → Página principal")
            logger.info("  - GET  /analyze              → Página de análisis")
            logger.info("  - POST /api/predict          → ⭐ Predicción con TTA (MEJORADA)")
            logger.info("  - POST /api/predict-improved → Predicción mejorada con TTA (alternativa)")
            logger.info("  - POST /api/predict-top3     → Top 3 predicciones")
            logger.info("  - GET  /api/health           → Estado del servidor")
            logger.info("  - GET  /api/classes          → Clases disponibles")
            logger.info("")
            logger.info("✨ TTA (Test Time Augmentation) activado por defecto para mayor precisión")
            
            # Iniciar servidor
            app.run(
                host=app.config['HOST'],
                port=app.config['PORT'],
                debug=app.config['DEBUG']
            )
        else:
            logger.error("❌ No se pudo inicializar el modelo. Saliendo...")

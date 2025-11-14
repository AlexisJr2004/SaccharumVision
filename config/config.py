"""
🍃 SaccharumVision - Configuración
==================================

Archivo centralizado de configuración para la aplicación SaccharumVision.
Incluye rutas, constantes del modelo, configuración del servidor y validación
de directorios críticos.
"""

import os
import logging

# ============================
# RUTAS Y ARCHIVOS DEL MODELO
# ============================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

IMG_SIZE = (224, 224)  # Tamaño requerido por el modelo
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'resnet50_latest.keras')
CLASSES_PATH = os.path.join(BASE_DIR, 'models', 'classes_latest.json')

# Clases por defecto si no se encuentra archivo
DEFAULT_CLASSES = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

# ============================
# CONFIGURACIÓN DE FLASK
# ============================
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
SECRET_KEY = 'saccharum_vision_secret_key_2024'

# ============================
# CONFIGURACIÓN DEL SERVIDOR
# ============================
DEBUG_MODE = True
HOST = '0.0.0.0'
PORT = 5000

# ============================
# CONFIGURACIÓN DE LOGGING
# ============================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '[%(asctime)s] %(levelname)s: %(message)s'
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# ============================
# VALIDACIÓN DE RUTAS CRÍTICAS
# ============================
def validate_paths():
    """Valida que existan las rutas críticas del sistema."""
    errors = []

    # Verificar modelo
    if not os.path.exists(MODEL_PATH):
        errors.append(f"Modelo no encontrado: {MODEL_PATH}")

    # Verificar directorio de modelos
    model_dir = os.path.dirname(MODEL_PATH)
    if not os.path.exists(model_dir):
        errors.append(f"Directorio de modelos no encontrado: {model_dir}")

    # Verificar / crear directorio de uploads
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    except Exception as e:
        errors.append(f"No se pudo crear el directorio de uploads: {e}")

    return errors

# ============================
# CLASES DE CONFIGURACIÓN
# ============================
class Config:
    """Configuración base utilizada por todos los entornos."""

    # App
    SECRET_KEY = SECRET_KEY
    UPLOAD_FOLDER = UPLOAD_FOLDER
    MAX_CONTENT_LENGTH = MAX_CONTENT_LENGTH

    # Modelo
    MODEL_PATH = MODEL_PATH
    CLASSES_PATH = CLASSES_PATH
    IMG_SIZE = IMG_SIZE
    DEFAULT_CLASSES = DEFAULT_CLASSES

    # Archivos permitidos
    ALLOWED_EXTENSIONS = ALLOWED_EXTENSIONS

    @staticmethod
    def init_app(app):
        """Inicializaciones específicas opcionales."""
        pass

class DevelopmentConfig(Config):
    """Configuración utilizada durante desarrollo."""
    DEBUG = True
    HOST = HOST
    PORT = PORT

class ProductionConfig(Config):
    """Configuración específica para producción."""
    DEBUG = False
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', 5000))

    @classmethod
    def init_app(cls, app):
        super().init_app(app)

        # Crear directorio de logs si no existe
        os.makedirs(LOG_DIR, exist_ok=True)

        # Configurar logging rotativo
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            os.path.join(LOG_DIR, 'saccharumvision.log'),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=10
        )

        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        file_handler.setLevel(logging.INFO)

        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('SaccharumVision iniciado en producción')

# ============================
# MAPEO DE CONFIGURACIONES
# ============================
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

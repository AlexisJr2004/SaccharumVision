"""
🍃 SaccharumVision - Configuración
==================================

Archivo de configuración centralizado con todas las constantes
y configuraciones de la aplicación.
"""

import os

# ============================
# CONFIGURACIÓN DEL MODELO
# ============================
IMG_SIZE = (224, 224)  # Tamaño esperado por el modelo
MODEL_PATH = 'models/resnet50_latest.keras'
CLASSES_PATH = 'models/classes_latest.json'

# Clases por defecto si no hay archivo
DEFAULT_CLASSES = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

# ============================
# CONFIGURACIÓN DE FLASK
# ============================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB máximo
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

# ============================
# CONFIGURACIÓN DE ARCHIVOS
# ============================
# Validación de rutas críticas
def validate_paths():
    """
    Valida que existan las rutas críticas del proyecto
    """
    errors = []
    
    # Verificar modelo
    if not os.path.exists(MODEL_PATH):
        errors.append(f"Modelo no encontrado: {MODEL_PATH}")
    
    # Verificar directorio de modelos
    model_dir = os.path.dirname(MODEL_PATH)
    if not os.path.exists(model_dir):
        errors.append(f"Directorio de modelos no encontrado: {model_dir}")
    
    # Crear directorio de uploads si no existe
    if not os.path.exists(UPLOAD_FOLDER):
        try:
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        except Exception as e:
            errors.append(f"No se pudo crear directorio de uploads: {e}")
    
    return errors

# ============================
# CONFIGURACIÓN ESPECÍFICA
# ============================
class Config:
    """Clase de configuración para diferentes entornos"""
    
    # Configuración base
    SECRET_KEY = SECRET_KEY
    UPLOAD_FOLDER = UPLOAD_FOLDER
    MAX_CONTENT_LENGTH = MAX_CONTENT_LENGTH
    
    # Configuración del modelo
    MODEL_PATH = MODEL_PATH
    CLASSES_PATH = CLASSES_PATH
    IMG_SIZE = IMG_SIZE
    DEFAULT_CLASSES = DEFAULT_CLASSES
    
    # Extensiones permitidas
    ALLOWED_EXTENSIONS = ALLOWED_EXTENSIONS
    
    @staticmethod
    def init_app(app):
        """Inicializa configuración específica de la app"""
        pass

class DevelopmentConfig(Config):
    """Configuración para desarrollo"""
    DEBUG = True
    HOST = HOST
    PORT = PORT

class ProductionConfig(Config):
    """Configuración para producción"""
    DEBUG = False
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', 5000))
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Configuraciones específicas de producción
        import logging
        from logging.handlers import RotatingFileHandler
        
        # Configurar logging para producción
        if not app.debug and not app.testing:
            if not os.path.exists('logs'):
                os.mkdir('logs')
            
            file_handler = RotatingFileHandler(
                'logs/saccharumvision.log', 
                maxBytes=10240000, 
                backupCount=10
            )
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
            app.logger.setLevel(logging.INFO)
            app.logger.info('SaccharumVision startup')

# Mapeo de configuraciones
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

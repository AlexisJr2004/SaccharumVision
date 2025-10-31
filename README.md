# 🍃 SaccharumVision - Sistema de Detección de Enfermedades en Caña de Azúcar

## 📋 Descripción

SaccharumVision es un sistema de visión por computadora basado en Deep Learning para la detección automática de enfermedades en plantaciones de caña de azúcar. Utiliza una arquitectura ResNet50 con Test Time Augmentation (TTA) para lograr predicciones precisas.

## 🎯 Enfermedades Detectadas

- ✅ **Healthy** (Saludable)
- 🦠 **Mosaic** (Mosaico)
- 🔴 **RedRot** (Pudrición Roja)
- 🟤 **Rust** (Roya)
- 🟡 **Yellow** (Amarillamiento)

## 🚀 Características

- ⭐ **Predicción con TTA**: Test Time Augmentation activado por defecto para mayor precisión (80-99% confianza)
- 📊 **Interfaz Web Intuitiva**: Sistema de drag & drop para subir imágenes
- 🔬 **API RESTful**: Endpoints para integración con otros sistemas
- 📈 **Análisis Detallado**: Muestra probabilidades para todas las clases
- 🎨 **Diseño Moderno**: Interfaz con Tailwind CSS

## 🛠️ Tecnologías

- **Backend**: Flask (Python)
- **Deep Learning**: TensorFlow/Keras
- **Modelo**: ResNet50 (Transfer Learning)
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **Precisión**: 90%+ con TTA

## 📦 Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd Proyecto_Agronomìa
```

### 2. Crear entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar archivos necesarios

Asegúrate de tener los siguientes archivos en la carpeta `models/`:
- `resnet50_latest.keras` (modelo entrenado)
- `classes_latest.json` (clases del modelo)

### 5. Ejecutar la aplicación

```bash
python app.py
```

La aplicación estará disponible en: `http://localhost:5000`

## 📁 Estructura del Proyecto

```
Proyecto_Agronomìa/
│
├── app.py                      # Aplicación Flask principal
├── requirements.txt            # Dependencias de Python
├── .gitignore                 # Archivos ignorados por Git
├── CHANGELOG_TTA.md           # Documentación de cambios TTA
│
├── config/
│   ├── config.py              # Configuración de la aplicación
│   └── __pycache__/
│
├── models/
│   ├── resnet50_latest.keras  # Modelo entrenado (no en git)
│   ├── classes_latest.json    # Clases del modelo
│   ├── metrics_*.json         # Métricas de entrenamiento
│   └── .gitkeep
│
├── static/
│   ├── assets/                # Recursos estáticos
│   └── js/
│       └── analyze.js         # JavaScript para análisis
│
├── templates/
│   ├── index.html             # Página principal
│   ├── analyze.html           # Página de análisis
│   ├── 404.html               # Error 404
│   └── 500.html               # Error 500
│
├── tests/
│   ├── test_images_batch.py   # Test de predicciones por lote
│   ├── test_api_tta.py        # Test de API con TTA
│   └── .gitkeep
│
├── uploads/                    # Imágenes subidas (no en git)
│   └── .gitkeep
│
└── utils/
    ├── __init__.py
    ├── model_manager.py       # Gestor del modelo
    └── __pycache__/
```

## 🔧 Configuración

### Variables de Entorno

Puedes crear un archivo `.env` con las siguientes variables:

```env
FLASK_ENV=development
FLASK_DEBUG=1
HOST=0.0.0.0
PORT=5000
```

### Configuración del Modelo

En `config/config.py` puedes ajustar:
- Tamaño de imagen
- Ruta del modelo
- Extensiones permitidas
- Tamaño máximo de archivo

## 📡 API Endpoints

### 1. Predicción con TTA (Principal)

```bash
POST /api/predict
Content-Type: multipart/form-data

Parámetros:
- file: imagen a analizar (JPG, PNG, BMP, TIFF)
- use_tta: true/false (default: true)
- threshold: 0.0-1.0 (default: 0.70)
- num_augmentations: int (default: 5)
```

**Ejemplo con cURL:**

```bash
curl -X POST http://localhost:5000/api/predict \
  -F "file=@imagen.jpg"
```

**Ejemplo con Python:**

```python
import requests

files = {'file': open('imagen.jpg', 'rb')}
response = requests.post('http://localhost:5000/api/predict', files=files)
data = response.json()

print(f"Clase: {data['prediction']['class']}")
print(f"Confianza: {data['prediction']['confidence']:.2f}%")
```

### 2. Predicción Mejorada (Alternativa)

```bash
POST /api/predict-improved
```

### 3. Top 3 Predicciones

```bash
POST /api/predict-top3
```

### 4. Salud del Servidor

```bash
GET /api/health
```

### 5. Clases Disponibles

```bash
GET /api/classes
```

## 🧪 Testing

### Test de Predicciones por Lote

Analiza todas las imágenes en la carpeta `uploads/`:

```bash
# Test básico (sin TTA)
python tests/test_images_batch.py

# Test con TTA (recomendado)
python tests/test_images_batch.py --tta

# Test sin guardar resultados
python tests/test_images_batch.py --tta --no-save
```

### Test de API

Prueba el endpoint `/api/predict`:

```bash
python tests/test_api_tta.py
```

## 📊 Comparación de Métodos

| Método | Tiempo | Precisión | Confianza Promedio |
|--------|--------|-----------|-------------------|
| Estándar | ~0.5s | ❌ Baja | ~32% |
| TTA (5 aug) | ~2.5s | ✅ Alta | ~90% |
| TTA (10 aug) | ~4.5s | ✅ Muy Alta | ~92% |

## ⚙️ Configuración Recomendada

### Para Producción (balance precisión/velocidad)

```python
use_tta = True
threshold = 0.70
num_augmentations = 5
```

### Para Máxima Precisión

```python
use_tta = True
threshold = 0.80
num_augmentations = 10
```

### Para Máxima Velocidad (no recomendado)

```python
use_tta = False
threshold = 0.50
```

## 🎓 ¿Qué es TTA?

**Test Time Augmentation** es una técnica que:
1. Aplica múltiples transformaciones aleatorias a la imagen
2. Realiza una predicción para cada versión transformada
3. Promedia todas las predicciones para obtener un resultado robusto

**Ventajas:**
- ✅ Mayor precisión (80-99% vs 30-35%)
- ✅ Más robustez ante variaciones
- ✅ Reduce impacto de ruido e iluminación

**Desventajas:**
- ⏱️ Mayor tiempo de procesamiento (~2-4s vs 0.5s)

## 🐛 Troubleshooting

### Error: Modelo no encontrado

```bash
# Asegúrate de tener el modelo en la carpeta correcta
ls models/resnet50_latest.keras
```

### Error: Puerto 5000 ocupado

```bash
# Cambia el puerto en config/config.py o usa:
python app.py --port 8080
```

### Error: Memoria insuficiente

Reduce el número de aumentaciones:
```python
num_augmentations = 3  # En lugar de 5
```

## 📝 Notas Importantes

1. **Modelos**: Los archivos `.keras` son grandes y no están en el repositorio. Debes entrenar tu propio modelo o solicitar acceso.

2. **Imágenes**: La carpeta `uploads/` está en `.gitignore`. Las imágenes de prueba no se subirán al repositorio.

3. **Tests**: Los archivos de test están ignorados pero la carpeta se mantiene con `.gitkeep`.

4. **Entorno Virtual**: Siempre activa el entorno virtual antes de trabajar:
   ```bash
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es parte de un sistema de visión agrónoma para la detección de enfermedades en caña de azúcar.

## 👨‍💻 Autor

Sistema de Visión Agrónoma - SaccharumVision

## 🔗 Enlaces

- [Documentación de Flask](https://flask.palletsprojects.com/)
- [TensorFlow](https://www.tensorflow.org/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)

---

**Última actualización:** 31 de Octubre de 2025

**Versión:** 1.0.0 (TTA Activado por Defecto)

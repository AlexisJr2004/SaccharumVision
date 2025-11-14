# 🍃 SaccharumVision - Sistema de Detección de Enfermedades en Caña de Azúcar

## 📋 Descripción

SaccharumVision es un sistema de visión por computadora basado en Deep Learning para la detección automática de enfermedades en caña de azúcar. Implementa tres arquitecturas CNN (ResNet50, EfficientNetB0 y MobileNetV2) con una interfaz web moderna para análisis en tiempo real.

## 🎯 Enfermedades Detectadas

- ✅ **Healthy** (Saludable)
- 🦠 **Mosaic** (Mosaico)
- 🔴 **RedRot** (Pudrición Roja)
- 🟤 **Rust** (Roya)
- 🟡 **Yellow** (Amarillamiento)

## 🚀 Características

- 🤖 **Múltiples Modelos**: ResNet50, EfficientNetB0 y MobileNetV2
- 📸 **Análisis en Tiempo Real**: Cámara web integrada y carga de archivos
- 📊 **Historial de Análisis**: Seguimiento completo de predicciones
- ⚙️ **Configuración Flexible**: Ajuste de umbrales y selección de modelos
- 🎨 **Interfaz Moderna**: Diseño responsive con Tailwind CSS
- 🔬 **API RESTful**: Endpoints para integración con otros sistemas

## 🛠️ Tecnologías

- **Backend**: Flask 3.1.0
- **Deep Learning**: TensorFlow 2.18.0 / Keras 3.8.0
- **Modelos**: ResNet50, EfficientNetB0, MobileNetV2
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **Procesamiento**: Pillow, NumPy

## 📦 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/AlexisJr2004/SaccharumVision.git
cd SaccharumVision
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

### 4. Configurar modelos

Asegúrate de tener la estructura de modelos:
```
models/
├── ResNet50/
│   ├── ResNet50_latest.keras
│   └── classes_latest.json
├── EfficientNetB0/
│   ├── EfficientNetB0_latest.keras
│   └── classes_latest.json
└── MobileNetV2/
    ├── MobileNetV2_latest.keras
    └── classes_classes.json
```

> **Nota**: Los modelos están gestionados con Git LFS debido a su tamaño.

### 5. Ejecutar la aplicación

```bash
python app.py
```

La aplicación estará disponible en: `http://localhost:5000`

## 📁 Estructura del Proyecto

```
SaccharumVision/
│
├── app.py                      # Aplicación Flask principal
├── requirements.txt            # Dependencias de Python
├── .gitignore                 # Archivos ignorados por Git
├── .gitattributes             # Configuración Git LFS
│
├── config/
│   └── config.py              # Configuración de la aplicación
│
├── models/                     # Modelos entrenados (Git LFS)
│   ├── ResNet50/
│   ├── EfficientNetB0/
│   └── MobileNetV2/
│
├── static/
│   ├── assets/                # Recursos estáticos
│   └── js/
│       └── analyze.js         # JavaScript para análisis
│
├── templates/                  # Plantillas HTML
│   ├── base.html              # Plantilla base
│   ├── index.html             # Página principal
│   ├── camera.html            # Captura desde cámara
│   ├── results.html           # Resultados de análisis
│   ├── history.html           # Historial de análisis
│   ├── history_details.html   # Detalles de análisis
│   ├── settings.html          # Configuración
│   └── terms.html             # Términos y condiciones
│
├── tests/
│   └── test_comparison_models.py  # Comparación de modelos
│
├── uploads/                    # Imágenes subidas (ignorado)
│
└── utils/
    ├── __init__.py
    └── model_manager.py       # Gestor de modelos
```

## 🔧 Configuración

### Modelos Disponibles

| Modelo | Tamaño Entrada | Características |
|--------|---------------|-----------------|
| **ResNet50** | 224×224 | Equilibrado y confiable |
| **EfficientNetB0** | 256×256 | Eficiente y preciso |
| **MobileNetV2** | 256×256 | Rápido y ligero |

### Variables de Entorno

Opcionalmente, crea un archivo `.env`:

```env
FLASK_ENV=development
FLASK_DEBUG=1
HOST=0.0.0.0
PORT=5000
```

## 📡 API Endpoints

### 1. Analizar Imagen

```bash
POST /analyze
Content-Type: multipart/form-data

Parámetros:
- file: imagen a analizar
- model: ResNet50 | EfficientNetB0 | MobileNetV2 (opcional)
```

**Ejemplo con cURL:**

```bash
curl -X POST http://localhost:5000/analyze \
  -F "file=@imagen.jpg" \
  -F "model=ResNet50"
```

**Ejemplo con Python:**

```python
import requests

files = {'file': open('imagen.jpg', 'rb')}
data = {'model': 'ResNet50'}
response = requests.post('http://localhost:5000/analyze', files=files, data=data)
result = response.json()

print(f"Clase: {result['prediction']}")
print(f"Confianza: {result['confidence']:.2f}%")
```

### 2. Obtener Modelos Disponibles

```bash
GET /api/models
```

### 3. Historial de Análisis

```bash
GET /history
```

### 4. Detalles de Análisis

```bash
GET /history/<analysis_id>
```

## 🧪 Testing

### Comparación de Modelos

```bash
python tests/test_comparison_models.py
```

Este test evalúa el rendimiento de los tres modelos con las mismas imágenes.

## 🎯 Uso de la Aplicación

### Interfaz Web

1. **Página Principal** (`/`): Sube imágenes o usa la cámara
2. **Configuración** (`/settings`): Selecciona el modelo y ajusta umbrales
3. **Historial** (`/history`): Revisa análisis anteriores
4. **Resultados** (`/results`): Visualiza predicciones detalladas

### Análisis desde Cámara

1. Accede a `/camera`
2. Permite el acceso a la cámara
3. Captura la imagen de la hoja
4. Analiza automáticamente

## 🐛 Solución de Problemas

### Modelo no encontrado

```bash
# Verifica que los modelos existen
ls models/ResNet50/ResNet50_latest.keras
```

Si faltan, asegúrate de haber clonado correctamente con Git LFS:

```bash
git lfs pull
```

### Puerto ocupado

```bash
# Usa un puerto diferente
python app.py --port 8080
```

### Error de importación

```bash
# Reinstala las dependencias
pip install -r requirements.txt --force-reinstall
```

## 📝 Notas Importantes

- **Git LFS**: Los modelos `.keras` se gestionan con Git LFS debido a su tamaño (>100MB)
- **Uploads**: La carpeta `uploads/` está en `.gitignore` y no se sincroniza
- **Cache**: Los modelos se cargan en memoria para mayor velocidad
- **Formatos**: Soporta PNG, JPG, JPEG, GIF, BMP, WEBP

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -m 'Añadir nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 👨‍💻 Autor

**Alexis Jr** - [AlexisJr2004](https://github.com/AlexisJr2004)

## 🔗 Enlaces Útiles

- [Documentación Flask](https://flask.palletsprojects.com/)
- [TensorFlow](https://www.tensorflow.org/)
- [Git LFS](https://git-lfs.github.com/)

---

⭐ Si este proyecto te resultó útil, considera darle una estrella en GitHub

**Última actualización:** Noviembre 2025

/**
 * 🍃 SaccharumVision - Análisis de Imágenes
 * ==========================================
 * JavaScript para manejo de subida, previsualización
 * y análisis de imágenes mediante el modelo backend.
 */

// ============================================================
//                      ELEMENTOS DEL DOM
// ============================================================

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('dropzone-file');
const uploadContent = document.getElementById('uploadContent');
const previewImage = document.getElementById('previewImage');
const actionButtons = document.getElementById('actionButtons');

const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');

const initialState = document.getElementById('initialState');
const loadingState = document.getElementById('loadingState');
const resultsSection = document.getElementById('resultsSection');
const errorState = document.getElementById('errorState');

const predictedClass = document.getElementById('predictedClass');
const confidence = document.getElementById('confidence');
const probabilitiesList = document.getElementById('probabilitiesList');
const errorMessage = document.getElementById('errorMessage');

// Archivo seleccionado
let selectedFile = null;

// ============================================================
//                      INFORMACIÓN DE ENFERMEDADES
// ============================================================

const diseaseInfoData = {
    'Healthy': {
        name: 'Saludable',
        icon: '✅',
        description: 'La planta se encuentra en perfecto estado de salud.',
        recommendation: 'Mantén prácticas de cultivo actuales.'
    },
    'Mosaic': {
        name: 'Mosaico',
        icon: '🦠',
        description: 'Enfermedad viral que genera manchas en hojas.',
        recommendation: 'Elimina plantas infectadas y controla vectores.'
    },
    'RedRot': {
        name: 'Pudrición Roja',
        icon: '🔴',
        description: 'Enfermedad fúngica severa del tallo.',
        recommendation: 'Mejorar drenaje y usar fungicidas.'
    },
    'Rust': {
        name: 'Roya',
        icon: '🟤',
        description: 'Pústulas marrón-rojizas en hojas.',
        recommendation: 'Aplicar fungicidas sistémicos.'
    },
    'Yellow': {
        name: 'Amarillamiento',
        icon: '🟡',
        description: 'Estrés o deficiencia nutricional.',
        recommendation: 'Revisar nutrición, riego y pH.'
    }
};

// ============================================================
//                    EVENTOS PRINCIPALES
// ============================================================

// Prevenir comportamiento por defecto en drag & drop
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event =>
    uploadArea.addEventListener(event, preventDefaults, false)
);
document.body.addEventListener('dragenter', preventDefaults, false);
document.body.addEventListener('dragover', preventDefaults, false);

// Efectos visuales del área de subida
['dragenter', 'dragover'].forEach(event =>
    uploadArea.addEventListener(event, () => uploadArea.classList.add('dragover'))
);

['dragleave', 'drop'].forEach(event =>
    uploadArea.addEventListener(event, () => uploadArea.classList.remove('dragover'))
);

// Evento al arrastrar archivo
uploadArea.addEventListener('drop', (e) => {
    if (e.dataTransfer.files.length > 0)
        handleFile(e.dataTransfer.files[0]);
});

// Evento de selección manual
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0)
        handleFile(e.target.files[0]);
});

// Botones principales
analyzeBtn.addEventListener('click', analyzeImage);
clearBtn.addEventListener('click', clearAll);

// ============================================================
//                   FUNCIONES PRINCIPALES
// ============================================================

/**
 * Maneja el archivo subido por el usuario.
 */
function handleFile(file) {

    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'];
    if (!validTypes.includes(file.type)) {
        showError('Tipo de archivo no soportado. Use JPG, PNG, BMP o TIFF.');
        return;
    }

    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('El archivo es demasiado grande. Máximo 16MB.');
        return;
    }

    selectedFile = file;

    // Mostrar preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;

        uploadContent.classList.add('hidden');
        previewImage.classList.remove('hidden');

        actionButtons.classList.remove('hidden');
        actionButtons.classList.add('flex');

        hideAllStates();
        initialState.classList.remove('hidden');
    };

    reader.readAsDataURL(file);
}

/**
 * Envía la imagen al backend para análisis.
 */
async function analyzeImage() {
    if (!selectedFile) {
        showError('Por favor selecciona una imagen primero.');
        return;
    }

    hideAllStates();
    loadingState.classList.remove('hidden');
    analyzeBtn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success)
            displayResults(data.prediction);
        else
            showError(data.error || 'Error desconocido al analizar la imagen.');

    } catch (err) {
        console.error('Error:', err);
        showError('Error de conexión. Intenta de nuevo.');
    } finally {
        analyzeBtn.disabled = false;
    }
}

/**
 * Muestra los resultados entregados por el modelo.
 */
function displayResults(prediction) {
    hideAllStates();
    resultsSection.classList.remove('hidden');

    const classInfo = diseaseInfoData[prediction.class] || {
        name: prediction.class,
        icon: '❓'
    };

    predictedClass.textContent = `${classInfo.icon} ${classInfo.name}`;

    const confidenceValue = prediction.confidence || (prediction.probability * 100);
    confidence.textContent = `${confidenceValue.toFixed(1)}%`;

    probabilitiesList.innerHTML = '';

    for (const [className, prob] of Object.entries(prediction.all_probabilities)) {
        const info = diseaseInfoData[className] || { name: className, icon: '❓' };
        const pct = (prob * 100).toFixed(1);

        const item = document.createElement('div');
        item.className = 'mb-3';
        item.innerHTML = `
            <div class="flex justify-between items-center mb-1">
                <span class="text-sm">${info.icon} ${info.name}</span>
                <span class="text-sm font-semibold">${pct}%</span>
            </div>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${pct}%"></div>
            </div>
        `;

        probabilitiesList.appendChild(item);
    }
}

// ============================================================
//                       MANEJO DE ESTADOS
// ============================================================

function showError(message) {
    hideAllStates();
    errorState.classList.remove('hidden');
    errorMessage.textContent = message;
}

function hideAllStates() {
    initialState.classList.add('hidden');
    loadingState.classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorState.classList.add('hidden');
}

/**
 * Limpia la interfaz para un nuevo análisis.
 */
function clearAll() {
    selectedFile = null;
    fileInput.value = '';

    previewImage.classList.add('hidden');
    uploadContent.classList.remove('hidden');

    actionButtons.classList.add('hidden');
    actionButtons.classList.remove('flex');

    hideAllStates();
    initialState.classList.remove('hidden');
}

// ============================================================
//                        INICIALIZACIÓN
// ============================================================

console.log('🍃 SaccharumVision - Sistema de análisis cargado');

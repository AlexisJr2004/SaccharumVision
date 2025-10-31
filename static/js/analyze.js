/**
 * 🍃 SaccharumVision - Análisis de Imágenes
 * ==========================================
 * 
 * JavaScript para manejar la subida de imágenes y análisis con el modelo
 */

// Elementos del DOM
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

// Variable para almacenar el archivo seleccionado
let selectedFile = null;

// Información de enfermedades
const diseaseInfoData = {
    'Healthy': {
        name: 'Saludable',
        icon: '✅',
        description: 'La planta se encuentra en perfecto estado de salud. No se detectaron signos de enfermedades.',
        recommendation: 'Mantén las prácticas de cultivo actuales y continúa con el monitoreo regular.'
    },
    'Mosaic': {
        name: 'Mosaico',
        icon: '🦠',
        description: 'Enfermedad viral que causa manchas claras y oscuras en las hojas, reduciendo la fotosíntesis.',
        recommendation: 'Elimina las plantas infectadas, controla vectores (áfidos) y usa variedades resistentes.'
    },
    'RedRot': {
        name: 'Pudrición Roja',
        icon: '🔴',
        description: 'Enfermedad fúngica severa que causa pudrición roja interna en el tallo.',
        recommendation: 'Mejora el drenaje, aplica fungicidas y usa variedades resistentes. Destruye residuos infectados.'
    },
    'Rust': {
        name: 'Roya',
        icon: '🟤',
        description: 'Enfermedad fúngica que causa pústulas de color marrón-rojizo en las hojas.',
        recommendation: 'Aplica fungicidas sistémicos, mejora la ventilación y usa variedades resistentes.'
    },
    'Yellow': {
        name: 'Amarillamiento',
        icon: '🟡',
        description: 'Síntomas de estrés o deficiencias nutricionales que causan amarillamiento de hojas.',
        recommendation: 'Verifica nutrición, riego y pH del suelo. Puede requerir fertilización adecuada.'
    }
};

// =============================
// Event Listeners
// =============================

// Prevenir comportamiento por defecto en drag & drop
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Efectos visuales en drag & drop
['dragenter', 'dragover'].forEach(eventName => {
    uploadArea.addEventListener(eventName, () => {
        uploadArea.classList.add('dragover');
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, () => {
        uploadArea.classList.remove('dragover');
    }, false);
});

// Manejar drop de archivo
uploadArea.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}, false);

// Manejar selección de archivo
fileInput.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Botón de analizar
analyzeBtn.addEventListener('click', analyzeImage);

// Botón de limpiar
clearBtn.addEventListener('click', clearAll);

// =============================
// Funciones principales
// =============================

/**
 * Maneja el archivo seleccionado
 */
function handleFile(file) {
    // Validar tipo de archivo
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'];
    if (!validTypes.includes(file.type)) {
        showError('Tipo de archivo no soportado. Use JPG, PNG, BMP o TIFF.');
        return;
    }

    // Validar tamaño (16MB)
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
 * Analiza la imagen con el modelo
 */
async function analyzeImage() {
    if (!selectedFile) {
        showError('Por favor selecciona una imagen primero.');
        return;
    }

    // Mostrar estado de carga
    hideAllStates();
    loadingState.classList.remove('hidden');
    analyzeBtn.disabled = true;

    try {
        // Crear FormData
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Enviar petición
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data.prediction);
        } else {
            showError(data.error || 'Error desconocido al analizar la imagen.');
        }

    } catch (error) {
        console.error('Error:', error);
        showError('Error de conexión. Por favor intenta de nuevo.');
    } finally {
        analyzeBtn.disabled = false;
    }
}

/**
 * Muestra los resultados del análisis
 */
function displayResults(prediction) {
    hideAllStates();
    resultsSection.classList.remove('hidden');

    // Clase predicha y confianza
    const classInfo = diseaseInfoData[prediction.class] || {
        name: prediction.class,
        icon: '❓'
    };

    predictedClass.textContent = `${classInfo.icon} ${classInfo.name}`;
    
    // Usar el campo correcto de confianza (ya viene en porcentaje desde TTA)
    const confidenceValue = prediction.confidence || (prediction.probability * 100);
    confidence.textContent = `${confidenceValue.toFixed(1)}%`;

    // Lista de probabilidades
    probabilitiesList.innerHTML = '';
    for (const [className, probability] of Object.entries(prediction.all_probabilities)) {
        const info = diseaseInfoData[className] || { name: className, icon: '❓' };
        const percentage = (probability * 100).toFixed(1);
        
        const item = document.createElement('div');
        item.className = 'mb-3';
        item.innerHTML = `
            <div class="flex justify-between items-center mb-1">
                <span class="text-sm">${info.icon} ${info.name}</span>
                <span class="text-sm font-semibold">${percentage}%</span>
            </div>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${percentage}%"></div>
            </div>
        `;
        probabilitiesList.appendChild(item);
    }
}

/**
 * Muestra un error
 */
function showError(message) {
    hideAllStates();
    errorState.classList.remove('hidden');
    errorMessage.textContent = message;
}

/**
 * Oculta todos los estados
 */
function hideAllStates() {
    initialState.classList.add('hidden');
    loadingState.classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorState.classList.add('hidden');
}

/**
 * Limpia todo
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

// =============================
// Inicialización
// =============================

console.log('🍃 SaccharumVision - Sistema de análisis cargado');

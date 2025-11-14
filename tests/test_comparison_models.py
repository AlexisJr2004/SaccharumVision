"""
🍃 SaccharumVision - Test de Comparación de Modelos
===================================================

Script para comparar la precisión de los 3 modelos con y sin TTA
"""

import sys
import os
import time
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_manager import ModelManager
from PIL import Image

# ============================
# CONFIGURACIÓN
# ============================

# Ruta de la imagen de prueba
TEST_IMAGE = r"c:\Users\duran\OneDrive\Escritorio\pp\tests\rust (104).jpeg"

# Configuración de modelos
MODELS_CONFIG = {
    'ResNet50': {
        'model_path': 'models/ResNet50/resnet50_latest.keras',
        'classes_path': 'models/ResNet50/classes_latest.json',
        'img_size': (224, 224),
        'description': 'Equilibrado y confiable'
    },
    'EfficientNetB0': {
        'model_path': 'models/EfficientNetB0/EfficientNetB0_latest.keras',
        'classes_path': 'models/EfficientNetB0/classes_latest.json',
        'img_size': (256, 256),
        'description': 'Eficiente y preciso'
    },
    'MobileNetV2': {
        'model_path': 'models/MobileNetV2/MobileNetV2_latest.keras',
        'classes_path': 'models/MobileNetV2/classes_classes.json',
        'img_size': (256, 256),
        'description': 'Rápido y ligero'
    }
}

# Información de enfermedades
DISEASE_INFO = {
    'Healthy': {'name': 'Saludable', 'icon': '✅'},
    'Mosaic': {'name': 'Mosaico', 'icon': '🦠'},
    'RedRot': {'name': 'Pudrición Roja', 'icon': '🔴'},
    'Rust': {'name': 'Roya', 'icon': '🟤'},
    'Yellow': {'name': 'Amarillamiento', 'icon': '🟡'}
}

# ============================
# FUNCIONES
# ============================

def print_header(text):
    """Imprime un encabezado formateado"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_subheader(text):
    """Imprime un subencabezado"""
    print(f"\n{text}")
    print("-"*70)

def format_probabilities(probabilities):
    """Formatea las probabilidades para impresión"""
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    result = []
    for class_name, prob in sorted_probs:
        info = DISEASE_INFO.get(class_name, {'name': class_name, 'icon': '❓'})
        percentage = prob * 100
        bar_length = int(percentage / 2)  # Escala de 50 caracteres
        bar = '█' * bar_length + '░' * (50 - bar_length)
        result.append(f"  {info['icon']} {info['name']:20s} {percentage:6.2f}% {bar}")
    return '\n'.join(result)

def test_model(model_name, config, image_path, use_tta=False):
    """
    Prueba un modelo específico
    
    Args:
        model_name: Nombre del modelo
        config: Configuración del modelo
        image_path: Ruta de la imagen
        use_tta: Si se debe usar TTA
        
    Returns:
        dict con resultados
    """
    try:
        print(f"\n🔄 Cargando modelo {model_name}...")
        
        # Cargar modelo
        start_load = time.time()
        manager = ModelManager(
            model_path=config['model_path'],
            classes_path=config['classes_path'],
            img_size=config['img_size'],
            model_type=model_name  # NUEVO: Pasar tipo de modelo para preprocesamiento correcto
        )
        load_time = time.time() - start_load
        
        print(f"✅ Modelo cargado en {load_time:.2f}s")
        
        # Hacer predicción
        print(f"📷 Procesando imagen...")
        start_pred = time.time()
        if use_tta:
            print("🔬 Aplicando TTA (8 augmentaciones)...")
            result = manager.improved_predict(image_path, use_tta=True, num_augmentations=8)
        else:
            print("⚡ Predicción directa (sin TTA)...")
            result = manager.predict_single_direct(image_path)
        pred_time = time.time() - start_pred
        
        # Preparar resultados
        if use_tta:
            # improved_predict devuelve formato diferente
            predicted_class = result.get('class', 'Unknown')
            confidence = result.get('confidence', 0)
            all_probs = result.get('probabilities', {})
        else:
            # predict_single_direct devuelve formato más simple
            predicted_class = result.get('class', 'Unknown')
            confidence = result.get('confidence', 0)
            all_probs = result.get('probabilities', {})
        
        info = DISEASE_INFO.get(predicted_class, {'name': predicted_class, 'icon': '❓'})
        
        print(f"\n✨ Resultado:")
        print(f"  Clase predicha: {info['icon']} {info['name']}")
        print(f"  Confianza: {confidence:.2f}%")
        print(f"  Tiempo de predicción: {pred_time:.2f}s")
        
        print(f"\n📊 Distribución de probabilidades:")
        print(format_probabilities(all_probs))
        
        return {
            'success': True,
            'model': model_name,
            'use_tta': use_tta,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probs,
            'load_time': load_time,
            'prediction_time': pred_time,
            'total_time': load_time + pred_time
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            'success': False,
            'model': model_name,
            'use_tta': use_tta,
            'error': str(e)
        }

def print_comparison_table(results):
    """Imprime tabla comparativa de resultados"""
    print_header("📊 TABLA COMPARATIVA DE RESULTADOS")
    
    # Sin TTA
    print_subheader("⚡ SIN TTA (Predicción Directa)")
    print(f"{'Modelo':<20} {'Predicción':<20} {'Confianza':<12} {'Tiempo':<12}")
    print("-"*70)
    
    for result in results:
        if not result['use_tta'] and result['success']:
            info = DISEASE_INFO.get(result['predicted_class'], {'name': result['predicted_class'], 'icon': '❓'})
            pred_str = f"{info['icon']} {info['name']}"
            print(f"{result['model']:<20} {pred_str:<20} {result['confidence']:>6.2f}%     {result['prediction_time']:>6.2f}s")
    
    # Con TTA
    print_subheader("🔬 CON TTA (Test Time Augmentation)")
    print(f"{'Modelo':<20} {'Predicción':<20} {'Confianza':<12} {'Tiempo':<12}")
    print("-"*70)
    
    for result in results:
        if result['use_tta'] and result['success']:
            info = DISEASE_INFO.get(result['predicted_class'], {'name': result['predicted_class'], 'icon': '❓'})
            pred_str = f"{info['icon']} {info['name']}"
            print(f"{result['model']:<20} {pred_str:<20} {result['confidence']:>6.2f}%     {result['prediction_time']:>6.2f}s")
    
    # Análisis de mejora con TTA
    print_subheader("📈 MEJORA CON TTA")
    print(f"{'Modelo':<20} {'Δ Confianza':<15} {'Δ Tiempo':<15} {'Cambió Predicción':<20}")
    print("-"*70)
    
    # Agrupar por modelo
    by_model = {}
    for result in results:
        if result['success']:
            model = result['model']
            if model not in by_model:
                by_model[model] = {}
            if result['use_tta']:
                by_model[model]['tta'] = result
            else:
                by_model[model]['direct'] = result
    
    for model, data in by_model.items():
        if 'direct' in data and 'tta' in data:
            direct = data['direct']
            tta = data['tta']
            
            conf_diff = tta['confidence'] - direct['confidence']
            time_diff = tta['prediction_time'] - direct['prediction_time']
            changed = "✅ SÍ" if direct['predicted_class'] != tta['predicted_class'] else "❌ NO"
            
            conf_str = f"{conf_diff:+.2f}%"
            time_str = f"{time_diff:+.2f}s"
            
            print(f"{model:<20} {conf_str:<15} {time_str:<15} {changed:<20}")

def save_results(results, output_file):
    """Guarda resultados en archivo JSON"""
    import json
    
    # Convertir a formato serializable
    serializable_results = []
    for result in results:
        serializable_result = result.copy()
        # Convertir probabilities a formato serializable
        if 'all_probabilities' in serializable_result:
            serializable_result['all_probabilities'] = {
                k: float(v) for k, v in serializable_result['all_probabilities'].items()
            }
        serializable_results.append(serializable_result)
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'test_image': TEST_IMAGE,
        'results': serializable_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados en: {output_file}")

# ============================
# MAIN
# ============================

def main():
    print_header("🍃 SACCHARUMVISION - TEST DE COMPARACIÓN DE MODELOS")
    print(f"\n📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📷 Imagen de prueba: {TEST_IMAGE}")
    
    # Verificar que existe la imagen
    if not os.path.exists(TEST_IMAGE):
        print(f"\n❌ Error: No se encuentra la imagen {TEST_IMAGE}")
        print("Por favor, asegúrate de que la imagen existe en la carpeta tests/")
        return
    
    results = []
    
    # Probar cada modelo sin TTA
    print_header("FASE 1: PREDICCIÓN DIRECTA (SIN TTA)")
    for model_name, config in MODELS_CONFIG.items():
        print_subheader(f"{model_name} - {config['description']}")
        result = test_model(model_name, config, TEST_IMAGE, use_tta=False)
        results.append(result)
        time.sleep(1)  # Pequeña pausa entre modelos
    
    # Probar cada modelo con TTA
    print_header("FASE 2: PREDICCIÓN CON TTA (TEST TIME AUGMENTATION)")
    for model_name, config in MODELS_CONFIG.items():
        print_subheader(f"{model_name} - {config['description']}")
        result = test_model(model_name, config, TEST_IMAGE, use_tta=True)
        results.append(result)
        time.sleep(1)
    
    # Mostrar tabla comparativa
    print_comparison_table(results)
    
    # Guardar resultados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"tests/test_comparison_{timestamp}.json"
    save_results(results, output_file)
    
    print_header("✅ TEST COMPLETADO")

if __name__ == "__main__":
    main()

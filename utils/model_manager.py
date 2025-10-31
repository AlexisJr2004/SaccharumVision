"""
🧠 SaccharumVision - Gestor del Modelo
======================================

Clase para manejar la carga y predicción con el modelo
de clasificación de enfermedades de caña de azúcar.

Autor: Sistema de Visión Agrónoma
"""

import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input  # type: ignore
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Gestor del modelo de Deep Learning para clasificación de imágenes
    """
    
    def __init__(self, model_path, classes_path, img_size=(256, 256)):
        """
        Inicializa el gestor del modelo
        
        Args:
            model_path (str): Ruta al archivo del modelo .keras
            classes_path (str): Ruta al archivo JSON con las clases
            img_size (tuple): Tamaño de imagen esperado (ancho, alto)
        """
        self.model_path = model_path
        self.classes_path = classes_path
        self.img_size = img_size
        self.model = None
        self.classes = []
        
        # Cargar modelo y clases
        self._load_model()
        self._load_classes()
        
    def _load_model(self):
        """
        Carga el modelo de TensorFlow/Keras
        """
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo no encontrado en: {self.model_path}")
            
            logger.info(f"📦 Cargando modelo desde: {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            logger.info(f"✅ Modelo cargado: {self.model.name}")
            
            # Información del modelo
            logger.info(f"📊 Input shape: {self.model.input_shape}")
            logger.info(f"📊 Output shape: {self.model.output_shape}")
            
        except Exception as e:
            logger.error(f"❌ Error al cargar el modelo: {str(e)}")
            raise
    
    def _load_classes(self):
        """
        Carga las clases desde el archivo JSON
        """
        try:
            if os.path.exists(self.classes_path):
                logger.info(f"📋 Cargando clases desde: {self.classes_path}")
                with open(self.classes_path, 'r') as f:
                    self.classes = json.load(f)
                logger.info(f"✅ Clases cargadas: {', '.join(self.classes)}")
            else:
                logger.warning(f"⚠️ Archivo de clases no encontrado: {self.classes_path}")
                logger.warning("⚠️ Usando clases por defecto")
                self.classes = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']
                
        except Exception as e:
            logger.error(f"❌ Error al cargar las clases: {str(e)}")
            logger.warning("⚠️ Usando clases por defecto")
            self.classes = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']
    
    def preprocess_image(self, image_path):
        """
        Preprocesa una imagen para la predicción
        
        Args:
            image_path (str): Ruta a la imagen
            
        Returns:
            np.array: Imagen preprocesada lista para el modelo
        """
        try:
            # Cargar imagen
            img = Image.open(image_path)
            
            # Convertir a RGB si es necesario
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Redimensionar
            img = img.resize(self.img_size)
            
            # Convertir a array numpy
            img_array = np.array(img)
            
            # Normalizar (0-1)
            img_array = img_array.astype('float32') / 255.0
            
            # Agregar dimensión del batch
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"❌ Error al preprocesar imagen: {str(e)}")
            raise
    
    def predict(self, image_path):
        """
        Realiza una predicción sobre una imagen
        
        Args:
            image_path (str): Ruta a la imagen
            
        Returns:
            dict: Diccionario con la predicción y probabilidades
                {
                    'class': str,
                    'confidence': float,
                    'probabilities': dict
                }
        """
        try:
            # Preprocesar imagen
            img_array = self.preprocess_image(image_path)
            
            # Realizar predicción
            predictions = self.model.predict(img_array, verbose=0)
            
            # Obtener la clase con mayor probabilidad
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.classes[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])
            
            # Crear diccionario de probabilidades
            probabilities = {
                self.classes[i]: float(predictions[0][i])
                for i in range(len(self.classes))
            }
            
            # Ordenar probabilidades de mayor a menor
            probabilities = dict(
                sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            )
            
            result = {
                'class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities
            }
            
            logger.info(f"🎯 Predicción: {predicted_class} ({confidence:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en predicción: {str(e)}")
            raise
    
    def predict_batch(self, image_paths):
        """
        Realiza predicciones sobre múltiples imágenes
        
        Args:
            image_paths (list): Lista de rutas a las imágenes
            
        Returns:
            list: Lista de diccionarios con las predicciones
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Error al procesar {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def get_classes(self):
        """
        Retorna la lista de clases disponibles
        
        Returns:
            list: Lista de nombres de clases
        """
        return self.classes
    
    def get_model_info(self):
        """
        Retorna información del modelo
        
        Returns:
            dict: Información del modelo
        """
        return {
            'name': self.model.name if self.model else None,
            'input_shape': str(self.model.input_shape) if self.model else None,
            'output_shape': str(self.model.output_shape) if self.model else None,
            'classes': self.classes,
            'num_classes': len(self.classes)
        }
    
    # ========================================================
    # 🚀 FUNCIONES DE PREDICCIÓN MEJORADA
    # ========================================================
    
    def load_and_preprocess_image_tf(self, image_path):
        """
        Cargar y preprocesar una imagen para ResNet50 usando TensorFlow
        
        Args:
            image_path (str): Ruta a la imagen
            
        Returns:
            tf.Tensor: Imagen preprocesada con preprocess_input de ResNet50
        """
        try:
            img = tf.io.read_file(image_path)
            img = tf.io.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, self.img_size)
            img = preprocess_input(img)
            return tf.expand_dims(img, 0)
        except Exception as e:
            logger.error(f"❌ Error cargando imagen: {e}")
            return None
    
    def get_top_3_predictions(self, image_path):
        """
        Obtener las 3 predicciones más probables
        
        Args:
            image_path (str): Ruta a la imagen
            
        Returns:
            list: Lista con las 3 mejores predicciones
        """
        try:
            img = self.load_and_preprocess_image_tf(image_path)
            if img is None:
                return None
            
            predictions = self.model.predict(img, verbose=0)[0]
            top_3_indices = np.argsort(predictions)[-3:][::-1]
            
            results = []
            for idx in top_3_indices:
                results.append({
                    'class': self.classes[idx],
                    'confidence': float(predictions[idx] * 100),
                    'probability': float(predictions[idx])
                })
            
            return results
        except Exception as e:
            logger.error(f"❌ Error obteniendo top 3: {e}")
            return None
    
    def predict_with_tta(self, image_path, num_augmentations=5):
        """
        Predicción con Test Time Augmentation (TTA)
        Promedia múltiples predicciones con augmentation para mayor precisión
        
        Args:
            image_path (str): Ruta a la imagen
            num_augmentations (int): Número de aumentaciones (5-10 recomendado)
            
        Returns:
            dict: Diccionario con la predicción mejorada
        """
        try:
            img_original = self.load_and_preprocess_image_tf(image_path)
            if img_original is None:
                return None
            
            predictions_list = []
            
            # Predicción sin augmentation
            pred = self.model.predict(img_original, verbose=0)[0]
            predictions_list.append(pred)
            
            # Predicciones con augmentation
            for _ in range(num_augmentations):
                img = tf.squeeze(img_original, 0)  # Remover batch dimension
                
                # 1. Rotaciones aleatorias
                img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
                
                # 2. Flips aleatorios
                if tf.random.uniform([]) > 0.5:
                    img = tf.image.flip_left_right(img)
                if tf.random.uniform([]) > 0.5:
                    img = tf.image.flip_up_down(img)
                
                # 3. Brightness aleatorio
                img = tf.image.random_brightness(img, 0.15)
                
                # 4. Contrast aleatorio
                img = tf.image.random_contrast(img, 0.85, 1.15)
                
                # Predicción
                img_batch = tf.expand_dims(img, 0)
                pred = self.model.predict(img_batch, verbose=0)[0]
                predictions_list.append(pred)
            
            # Promediar predicciones
            avg_prediction = np.mean(predictions_list, axis=0)
            confidence = np.max(avg_prediction)
            class_idx = np.argmax(avg_prediction)
            
            # Crear diccionario de probabilidades
            probabilities = {
                self.classes[i]: float(avg_prediction[i])
                for i in range(len(self.classes))
            }
            
            # Ordenar probabilidades de mayor a menor
            probabilities = dict(
                sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            )
            
            return {
                'class': self.classes[class_idx],
                'confidence': float(confidence * 100),
                'probability': float(confidence),
                'probabilities': probabilities,
                'num_augmentations': num_augmentations
            }
            
        except Exception as e:
            logger.error(f"❌ Error en TTA: {e}")
            return None
    
    def improved_predict(self, image_path, use_tta=True, threshold=0.70, num_augmentations=5):
        """
        Función de predicción MEJORADA con TTA y Threshold
        
        Args:
            image_path (str): Ruta a la imagen
            use_tta (bool): Usar Test Time Augmentation (True/False)
            threshold (float): Confianza mínima requerida (0.0-1.0)
            num_augmentations (int): Número de aumentaciones para TTA
            
        Returns:
            dict: Diccionario con:
                - status: 'success', 'warning' o 'error'
                - class: Clase predicha
                - confidence: Confianza en porcentaje
                - probability: Confianza en decimal
                - top_3: Top 3 predicciones
                - message: Mensaje descriptivo
                - method: Método de predicción usado
        """
        try:
            logger.info(f"🔍 Realizando predicción mejorada...")
            logger.info(f"   • TTA: {'Activado' if use_tta else 'Desactivado'}")
            logger.info(f"   • Threshold: {threshold*100:.0f}%")
            logger.info(f"   • Aumentaciones: {num_augmentations if use_tta else 'N/A'}")
            
            # Realizar predicción
            if use_tta:
                result = self.predict_with_tta(image_path, num_augmentations)
                method = f"TTA ({num_augmentations} aumentaciones)"
            else:
                # Predicción directa usando el método existente
                result = self.predict(image_path)
                result['probability'] = result['confidence']
                result['confidence'] = result['confidence'] * 100
                method = "Predicción directa"
            
            if result is None:
                return {
                    'status': 'error',
                    'message': 'Error en la predicción',
                    'confidence': 0,
                    'probability': 0
                }
            
            # Aplicar threshold
            probability = result['probability']
            
            # Obtener top 3
            top_3 = self.get_top_3_predictions(image_path)
            
            if probability < threshold:
                logger.warning(f"⚠️ Confianza baja: {result['confidence']:.1f}%")
                return {
                    'status': 'warning',
                    'class': result['class'],
                    'confidence': result['confidence'],
                    'probability': probability,
                    'probabilities': result.get('probabilities', {}),
                    'method': method,
                    'message': f"⚠️ Confianza baja ({result['confidence']:.1f}%). Resultado no confiable. Se requiere mínimo {threshold*100:.0f}%.",
                    'top_3': top_3
                }
            
            logger.info(f"✅ Predicción exitosa: {result['class']} ({result['confidence']:.1f}%)")
            
            return {
                'status': 'success',
                'class': result['class'],
                'confidence': result['confidence'],
                'probability': probability,
                'probabilities': result.get('probabilities', {}),
                'method': method,
                'message': f"✅ Detectado: {result['class']} ({result['confidence']:.1f}%)",
                'top_3': top_3
            }
            
        except Exception as e:
            logger.error(f"❌ Error en predicción mejorada: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error al procesar la imagen: {str(e)}',
                'confidence': 0,
                'probability': 0
            }

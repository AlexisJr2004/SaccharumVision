"""
🧠 SaccharumVision - Gestor del Modelo
======================================

Clase para manejar la carga y predicción con el modelo
de clasificación de enfermedades de caña de azúcar.

Autor: Sistema de Visión Agrónoma
"""

import os
import json
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)

# ============================================================
#                      MODEL MANAGER
# ============================================================

class ModelManager:
    """
    Gestor del modelo de Deep Learning para clasificación de imágenes.
    Contiene:
        - Carga del modelo
        - Carga de clases
        - Preprocesamiento (PIL y TF)
        - Predicción estándar
        - Predicción avanzada (Directa, TTA, Mejorada)
    """

    # ------------------------------------------------------------
    #                   INICIALIZACIÓN
    # ------------------------------------------------------------
    
    def __init__(self, model_path, classes_path, img_size=(256, 256), model_type='ResNet50'):
        """
        Inicializa el gestor del modelo.
        
        Args:
            model_path: Ruta al archivo del modelo
            classes_path: Ruta al archivo de clases
            img_size: Tamaño de imagen requerido
            model_type: Tipo de modelo (ResNet50, EfficientNetB0, MobileNetV2)
        """
        self.model_path = model_path
        self.classes_path = classes_path
        self.img_size = img_size
        self.model_type = model_type

        self.model = None
        self.classes = []
        
        # Determinar función de preprocesamiento según el tipo de modelo
        self._set_preprocess_function()

        # Cargar todos los componentes
        self._load_model()
        self._load_classes()

    def _set_preprocess_function(self):
        """
        Define la función de preprocesamiento correcta según el tipo de modelo
        """
        if 'resnet' in self.model_type.lower():
            from tensorflow.keras.applications.resnet50 import preprocess_input
            self.preprocess_fn = preprocess_input
            logger.info(f"✅ Usando preprocesamiento de ResNet50")
        elif 'efficientnet' in self.model_type.lower():
            from tensorflow.keras.applications.efficientnet import preprocess_input
            self.preprocess_fn = preprocess_input
            logger.info(f"✅ Usando preprocesamiento de EfficientNet")
        elif 'mobilenet' in self.model_type.lower():
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            self.preprocess_fn = preprocess_input
            logger.info(f"✅ Usando preprocesamiento de MobileNetV2")
        else:
            # Fallback: normalización estándar [-1, 1]
            self.preprocess_fn = lambda x: (x / 127.5) - 1.0
            logger.warning(f"⚠️ Tipo de modelo desconocido, usando preprocesamiento genérico")

    # ------------------------------------------------------------
    #                   CARGA DE MODELO Y CLASES
    # ------------------------------------------------------------

    def _load_model(self):
        """Carga el modelo desde un archivo .keras"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo no encontrado en: {self.model_path}")

            logger.info(f"📦 Cargando modelo desde: {self.model_path}")
            self.model = keras.models.load_model(self.model_path)

            logger.info(f"✅ Modelo cargado: {self.model.name}")
            logger.info(f"📊 Input shape: {self.model.input_shape}")
            logger.info(f"📊 Output shape: {self.model.output_shape}")

        except Exception as e:
            logger.error(f"❌ Error al cargar el modelo: {e}")
            raise

    def _load_classes(self):
        """Carga la lista de clases desde un archivo JSON"""
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
            logger.error(f"❌ Error al cargar clases: {e}")
            self.classes = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

    # ============================================================
    #                PREPROCESAMIENTO DE IMÁGENES
    # ============================================================

    def preprocess_image(self, image_path):
        """
        Preprocesamiento con PIL (Normalización 0-1).
        Usado por: predict(), predict_batch()
        """
        try:
            img = Image.open(image_path)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            img = img.resize(self.img_size)

            img_array = np.array(img).astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            return img_array

        except Exception as e:
            logger.error(f"❌ Error al preprocesar imagen: {e}")
            raise

    def load_and_preprocess_image_tf(self, image_path):
        """
        Preprocesamiento con TensorFlow usando la función correcta según el modelo.
        Usado por: predict_single_direct(), TTA, top_3
        """
        try:
            img = tf.io.read_file(image_path)
            img = tf.io.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, self.img_size)

            # Preprocesamiento específico del modelo
            img = self.preprocess_fn(img)

            return tf.expand_dims(img, 0)

        except Exception as e:
            logger.error(f"❌ Error cargando imagen con TF: {e}")
            return None

    # ============================================================
    #                        PREDICCIÓN BÁSICA
    # ============================================================

    def predict(self, image_path):
        """
        Predicción clásica usando preprocess_image() (PIL)
        """
        try:
            img_array = self.preprocess_image(image_path)
            predictions = self.model.predict(img_array, verbose=0)

            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.classes[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])

            probabilities = {
                self.classes[i]: float(predictions[0][i])
                for i in range(len(self.classes))
            }

            probabilities = dict(sorted(probabilities.items(),
                                        key=lambda x: x[1],
                                        reverse=True))

            logger.info(f"🎯 Predicción: {predicted_class} ({confidence:.2%})")

            return {
                'class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities
            }

        except Exception as e:
            logger.error(f"❌ Error en predicción: {e}")
            raise

    def predict_batch(self, image_paths):
        """Predice múltiples imágenes en secuencia"""
        results = []

        for image_path in image_paths:
            try:
                r = self.predict(image_path)
                r['image_path'] = image_path
                results.append(r)
            except Exception as e:
                logger.error(f"❌ Error en {image_path}: {e}")
                results.append({'image_path': image_path, 'error': str(e)})

        return results

    # ============================================================
    #                   UTILIDADES DE PREDICCIÓN
    # ============================================================

    def get_classes(self):
        """Devuelve la lista de clases"""
        return self.classes

    def get_model_info(self):
        """Devuelve información del modelo"""
        return {
            'name': self.model.name if self.model else None,
            'input_shape': str(self.model.input_shape) if self.model else None,
            'output_shape': str(self.model.output_shape) if self.model else None,
            'classes': self.classes,
            'num_classes': len(self.classes)
        }

    # ============================================================
    #                   PREDICCIÓN AVANZADA (TF)
    # ============================================================

    def get_top_3_predictions(self, image_path):
        """
        Obtiene las 3 predicciones más probables usando TF preprocessing
        """
        try:
            img = self.load_and_preprocess_image_tf(image_path)
            if img is None:
                return None

            predictions = self.model.predict(img, verbose=0)[0]
            top_3_idx = np.argsort(predictions)[-3:][::-1]

            return [
                {
                    'class': self.classes[idx],
                    'confidence': float(predictions[idx] * 100),
                    'probability': float(predictions[idx])
                }
                for idx in top_3_idx
            ]

        except Exception as e:
            logger.error(f"❌ Error top-3: {e}")
            return None

    # ------------------------------------------------------------
    #                PREDICCIÓN DIRECTA (SIN TTA)
    # ------------------------------------------------------------

    def predict_single_direct(self, image_path):
        """
        Predicción pura sin augmentación (baseline).
        """
        try:
            logger.info("📸 Predicción directa sin augmentación...")

            img = self.load_and_preprocess_image_tf(image_path)
            if img is None:
                return None

            predictions = self.model.predict(img, verbose=0)[0]

            class_idx = np.argmax(predictions)
            confidence = predictions[class_idx]

            probabilities = {
                self.classes[i]: float(predictions[i])
                for i in range(len(self.classes))
            }

            probabilities = dict(sorted(probabilities.items(),
                                        key=lambda x: x[1],
                                        reverse=True))

            return {
                'class': self.classes[class_idx],
                'confidence': float(confidence * 100),
                'probability': float(confidence),
                'probabilities': probabilities
            }

        except Exception as e:
            logger.error(f"❌ Error en predicción directa: {e}")
            return None

    # ------------------------------------------------------------
    #                TTA — TEST TIME AUGMENTATION
    # ------------------------------------------------------------

    def predict_with_tta(self, image_path, num_augmentations=5):
        """
        Realiza TTA aplicando augmentaciones aleatorias y promediando resultados.
        """
        try:
            logger.info(f"🔄 Ejecutando TTA con {num_augmentations} augmentaciones...")

            img_original = self.load_and_preprocess_image_tf(image_path)
            if img_original is None:
                return None

            predictions_list = []

            # Predicción original
            base_pred = self.model.predict(img_original, verbose=0)[0]
            predictions_list.append(base_pred)

            # Predicciones aumentadas
            for i in range(num_augmentations):
                img = tf.squeeze(img_original, 0)

                # Augmentaciones aleatorias (rotación, flips, brillo, contraste, saturación)
                rotation = tf.random.uniform([], 0, 4, dtype=tf.int32)
                img = tf.image.rot90(img, k=rotation)

                if tf.random.uniform([]) > 0.4:
                    img = tf.image.flip_left_right(img)

                if tf.random.uniform([]) > 0.7:
                    img = tf.image.flip_up_down(img)

                img = tf.image.adjust_brightness(img, tf.random.uniform([], -0.2, 0.2))
                img = tf.image.adjust_contrast(img, tf.random.uniform([], 0.7, 1.4))
                img = tf.image.adjust_saturation(img, tf.random.uniform([], 0.5, 1.8))

                img = tf.clip_by_value(img, 0, 255)
                img_batch = tf.expand_dims(img, 0)

                pred = self.model.predict(img_batch, verbose=0)[0]
                predictions_list.append(pred)

            avg_pred = np.mean(predictions_list, axis=0)

            class_idx = np.argmax(avg_pred)
            confidence = avg_pred[class_idx]

            probabilities = {
                self.classes[i]: float(avg_pred[i])
                for i in range(len(self.classes))
            }

            probabilities = dict(sorted(probabilities.items(),
                                        key=lambda x: x[1],
                                        reverse=True))

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

    # ------------------------------------------------------------
    #                  PREDICCIÓN MEJORADA (TTA + THRESHOLD)
    # ------------------------------------------------------------

    def improved_predict(self, image_path, use_tta=True, threshold=0.70, num_augmentations=5):
        """
        Predicción mejorada: TTA opcional + Threshold + Top-3.
        """
        try:
            method = (
                f"TTA ({num_augmentations} augmentaciones)"
                if use_tta else
                "Predicción directa (sin TTA)"
            )

            result = (
                self.predict_with_tta(image_path, num_augmentations)
                if use_tta else
                self.predict_single_direct(image_path)
            )

            if result is None:
                return {
                    'status': 'error',
                    'message': 'Error en la predicción',
                    'confidence': 0,
                    'probability': 0
                }

            probability = result['probability']
            top_3 = self.get_top_3_predictions(image_path)

            # Threshold
            if probability < threshold:
                return {
                    'status': 'warning',
                    'class': result['class'],
                    'confidence': result['confidence'],
                    'probability': probability,
                    'probabilities': result.get('probabilities', {}),
                    'method': method,
                    'message': f"⚠️ Confianza baja ({result['confidence']:.1f}%). " +
                               f"Se requiere mínimo {threshold*100:.0f}%.",
                    'top_3': top_3
                }

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
            logger.error(f"❌ Error en predicción mejorada: {e}")
            return {
                'status': 'error',
                'message': f'Error al procesar la imagen: {e}',
                'confidence': 0,
                'probability': 0
            }

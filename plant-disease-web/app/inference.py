"""
Tahmin (inference) modülü
Tek görüntü üzerinde model tahmini yapar
"""

import numpy as np
from PIL import Image
from typing import Dict, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import mobilenet_v2


def prepare_image(file_stream, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Görüntüyü model için hazırlar
    
    Args:
        file_stream: Dosya stream (Flask file object)
        target_size (Tuple[int, int]): Hedef görüntü boyutu
        
    Returns:
        np.ndarray: Hazırlanmış görüntü array'i (1, 224, 224, 3)
    """
    # Görüntüyü PIL ile yükle
    image = Image.open(file_stream)
    
    # RGB'ye çevir (RGBA veya grayscale ise)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 224x224 boyutuna resize et
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # NumPy array'e çevir
    image_array = np.array(image, dtype=np.float32)
    
    # Batch dimension ekle (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)
    
    # MobileNetV2 preprocessing uygula
    # preprocess_input: [-1, 1] aralığına normalize eder
    image_array = mobilenet_v2.preprocess_input(image_array)
    
    return image_array


def predict(image_array: np.ndarray, 
            model: keras.Model, 
            class_names: list) -> Tuple[str, float, Dict[str, float]]:
    """
    Model ile tahmin yapar
    
    Args:
        image_array (np.ndarray): Hazırlanmış görüntü array'i
        model (keras.Model): Eğitilmiş model
        class_names (list): Sınıf isimleri listesi
        
    Returns:
        Tuple[str, float, Dict[str, float]]: 
            - predicted_label: Tahmin edilen sınıf adı
            - confidence: Güven skoru (0-1 arası)
            - probabilities: Tüm sınıfların olasılıkları (sınıf_adı -> olasılık)
    """
    # Model ile tahmin yap
    predictions = model.predict(image_array, verbose=0)
    
    # En yüksek olasılığa sahip indeksi bul
    predicted_idx = np.argmax(predictions[0])
    
    # Güven skoru (en yüksek olasılık)
    confidence = float(predictions[0][predicted_idx])
    
    # Tahmin edilen sınıf adı
    predicted_label = class_names[predicted_idx]
    
    # Tüm sınıfların olasılıklarını dictionary olarak oluştur
    probabilities = {
        class_names[i]: float(predictions[0][i])
        for i in range(len(class_names))
    }
    
    # Olasılıkları yüksekten düşüğe sırala
    probabilities = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))
    
    return predicted_label, confidence, probabilities


def get_top_predictions(probabilities: Dict[str, float], top_k: int = 3) -> Dict[str, float]:
    """
    En yüksek olasılığa sahip top-k tahmini döndürür
    
    Args:
        probabilities (Dict[str, float]): Tüm sınıfların olasılıkları
        top_k (int): Kaç tahmin döndürülecek
        
    Returns:
        Dict[str, float]: Top-k tahminler
    """
    # Zaten sıralı olduğu için ilk top_k'yi al
    top_predictions = dict(list(probabilities.items())[:top_k])
    
    return top_predictions


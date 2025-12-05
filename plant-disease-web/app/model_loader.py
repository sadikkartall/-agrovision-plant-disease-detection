"""
Model yükleme modülü
Eğitilmiş Keras modelini ve sınıf isimlerini yükler
"""

import os
import json
import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple


def load_model(model_path: str = "models/mobilenetv2_best.keras") -> keras.Model:
    """
    Eğitilmiş Keras modelini yükler
    
    Args:
        model_path (str): Model dosyasının yolu
        
    Returns:
        keras.Model: Yüklenen model
        
    Raises:
        FileNotFoundError: Model dosyası bulunamazsa
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {model_path}\n"
            f"Lütfen mobilenetv2_best.keras dosyasını models/ klasörüne koyun."
        )
    
    print(f"Model yükleniyor: {model_path}")
    
    # Modeli yükle
    # Inference için loss ve metrics gerekmez, sadece tahmin yapacağız
    model = keras.models.load_model(model_path, compile=False)
    
    # Inference için derle (loss ve metrics olmadan)
    model.compile()
    
    print(f"Model başarıyla yüklendi!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    return model


def load_class_names(json_path: str = "data/class_names.json") -> List[str]:
    """
    Sınıf isimlerini JSON dosyasından yükler
    
    Args:
        json_path (str): JSON dosyasının yolu
        
    Returns:
        List[str]: Sınıf isimleri listesi
        
    Raises:
        FileNotFoundError: JSON dosyası bulunamazsa
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Sınıf isimleri dosyası bulunamadı: {json_path}\n"
            f"Lütfen class_names.json dosyasını data/ klasörüne koyun."
        )
    
    print(f"Sınıf isimleri yükleniyor: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        class_names = json.load(f)
    
    if not isinstance(class_names, list):
        raise ValueError("class_names.json dosyası bir liste içermelidir.")
    
    print(f"Toplam {len(class_names)} sınıf yüklendi.")
    
    return class_names


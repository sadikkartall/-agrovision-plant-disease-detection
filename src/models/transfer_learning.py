"""
Transfer learning modülü
Önceden eğitilmiş modelleri kullanarak transfer learning uygular
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, applications
from typing import Tuple, Optional, Dict, Any, List
import numpy as np


class TransferLearningModel:
    """
    Transfer learning modelleri oluşturmak için sınıf
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 10,
                 base_model_name: str = 'mobilenet_v2'):
        """
        TransferLearningModel sınıfını başlatır
        
        Args:
            input_shape (Tuple[int, int, int]): Giriş görüntü boyutu
            num_classes (int): Sınıf sayısı
            base_model_name (str): Temel model adı
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_model_name = base_model_name
        self.base_model = None
        self.model = None
    
    def get_base_model(self, 
                      include_top: bool = False,
                      weights: str = 'imagenet',
                      pooling: str = 'avg') -> keras.Model:
        """
        Temel modeli yükler
        
        Args:
            include_top (bool): Üst katmanları dahil et
            weights (str): Ağırlık türü
            pooling (str): Pooling türü
            
        Returns:
            keras.Model: Temel model
        """
        if self.base_model_name == 'mobilenet_v2':
            base_model = applications.MobileNetV2(
                input_shape=self.input_shape,
                include_top=include_top,
                weights=weights,
                pooling=pooling
            )
        elif self.base_model_name == 'resnet50':
            base_model = applications.ResNet50(
                input_shape=self.input_shape,
                include_top=include_top,
                weights=weights,
                pooling=pooling
            )
        elif self.base_model_name == 'inception_v3':
            base_model = applications.InceptionV3(
                input_shape=self.input_shape,
                include_top=include_top,
                weights=weights,
                pooling=pooling
            )
        elif self.base_model_name == 'efficientnet_b0':
            # EfficientNet için özel düzeltme - RGB görüntüler için
            base_model = applications.EfficientNetB0(
                input_shape=self.input_shape,
                include_top=include_top,
                weights='imagenet',  # ImageNet ağırlıklarını kullan
                pooling=pooling
            )
        elif self.base_model_name == 'densenet121':
            base_model = applications.DenseNet121(
                input_shape=self.input_shape,
                include_top=include_top,
                weights=weights,
                pooling=pooling
            )
        else:
            raise ValueError(f"Desteklenmeyen model: {self.base_model_name}")
        
        self.base_model = base_model
        return base_model
    
    def build_feature_extraction_model(self, 
                                     dropout_rate: float = 0.5,
                                     learning_rate: float = 0.001,
                                     freeze_base: bool = True) -> keras.Model:
        """
        Feature extraction modeli oluşturur (temel model dondurulur)
        
        Args:
            dropout_rate (float): Dropout oranı
            learning_rate (float): Öğrenme oranı
            freeze_base (bool): Temel modeli dondur
            
        Returns:
            keras.Model: Oluşturulan model
        """
        # Temel modeli yükle
        base_model = self.get_base_model()
        
        # Temel modeli dondur
        if freeze_base:
            base_model.trainable = False
        
        # Model oluştur
        inputs = keras.Input(shape=self.input_shape)

        # Model-özel ön işleme: ImageNet ile uyumlu ölçekleme
        if self.base_model_name == 'mobilenet_v2':
            preprocess_fn = applications.mobilenet_v2.preprocess_input
        elif self.base_model_name == 'resnet50':
            preprocess_fn = applications.resnet50.preprocess_input
        elif self.base_model_name == 'inception_v3':
            preprocess_fn = applications.inception_v3.preprocess_input
        elif self.base_model_name == 'efficientnet_b0':
            # EfficientNet, keras.layers.Rescaling(1./255) ile de uyumludur ancak
            # keras uygulamasındaki preprocess_input fonksiyonu en güvenlisidir
            preprocess_fn = applications.efficientnet.preprocess_input
        elif self.base_model_name == 'densenet121':
            preprocess_fn = applications.densenet.preprocess_input
        else:
            preprocess_fn = None

        if preprocess_fn is not None:
            x = layers.Lambda(preprocess_fn, name='preprocess')(inputs)
        else:
            x = inputs

        x = base_model(x, training=False)
        
        # Dense katmanları ekle
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Çıkış katmanı
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Modeli derle
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def build_fine_tuning_model(self, 
                               dropout_rate: float = 0.5,
                               learning_rate: float = 0.0001,
                               fine_tune_layers: int = 10) -> keras.Model:
        """
        Fine-tuning modeli oluşturur (temel modelin üst katmanları eğitilir)
        
        Args:
            dropout_rate (float): Dropout oranı
            learning_rate (float): Öğrenme oranı
            fine_tune_layers (int): Fine-tuning yapılacak katman sayısı
            
        Returns:
            keras.Model: Oluşturulan model
        """
        # Temel modeli yükle
        base_model = self.get_base_model()
        
        # Temel modeli dondur
        base_model.trainable = False
        
        # Model oluştur
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        
        # Dense katmanları ekle
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Çıkış katmanı
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Modeli derle
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Fine-tuning için temel modelin üst katmanlarını aç
        base_model.trainable = True
        
        # Fine-tuning yapılacak katmanları belirle
        fine_tune_at = len(base_model.layers) - fine_tune_layers
        
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Modeli yeniden derle (daha düşük öğrenme oranı ile)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate * 0.1),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def build_ensemble_model(self, 
                           model_names: List[str] = ['mobilenet_v2', 'resnet50'],
                           dropout_rate: float = 0.5,
                           learning_rate: float = 0.001) -> keras.Model:
        """
        Ensemble modeli oluşturur (birden fazla temel modeli birleştirir)
        
        Args:
            model_names (List[str]): Kullanılacak model isimleri
            dropout_rate (float): Dropout oranı
            learning_rate (float): Öğrenme oranı
            
        Returns:
            keras.Model: Oluşturulan model
        """
        # Giriş katmanı
        inputs = keras.Input(shape=self.input_shape)
        
        # Her model için feature extraction
        model_outputs = []
        
        for model_name in model_names:
            # Geçici olarak model adını değiştir
            original_name = self.base_model_name
            self.base_model_name = model_name
            
            # Temel modeli yükle
            base_model = self.get_base_model()
            base_model.trainable = False
            
            # Feature extraction
            features = base_model(inputs, training=False)
            model_outputs.append(features)
            
            # Model adını geri yükle
            self.base_model_name = original_name
        
        # Özellikleri birleştir
        if len(model_outputs) > 1:
            combined = layers.Concatenate()(model_outputs)
        else:
            combined = model_outputs[0]
        
        # Dense katmanları
        x = layers.Dense(512, activation='relu')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Çıkış katmanı
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Modeli derle
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def get_available_models(self) -> List[str]:
        """
        Kullanılabilir model isimlerini döndürür
        
        Returns:
            List[str]: Model isimleri
        """
        return [
            'mobilenet_v2',
            'resnet50',
            'inception_v3',
            'efficientnet_b0',
            'densenet121'
        ]
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Model bilgilerini döndürür
        
        Args:
            model_name (str): Model adı
            
        Returns:
            Dict[str, Any]: Model bilgileri
        """
        info = {
            'mobilenet_v2': {
                'description': 'MobileNetV2 - Mobil cihazlar için optimize edilmiş',
                'parameters': '3.4M',
                'size': '14MB',
                'top1_accuracy': '71.3%'
            },
            'resnet50': {
                'description': 'ResNet50 - Residual connections ile derin ağ',
                'parameters': '25.6M',
                'size': '98MB',
                'top1_accuracy': '74.9%'
            },
            'inception_v3': {
                'description': 'InceptionV3 - Inception modülleri ile',
                'parameters': '23.9M',
                'size': '92MB',
                'top1_accuracy': '78.0%'
            },
            'efficientnet_b0': {
                'description': 'EfficientNetB0 - Verimli ve doğru model',
                'parameters': '5.3M',
                'size': '20MB',
                'top1_accuracy': '77.1%'
            },
            'densenet121': {
                'description': 'DenseNet121 - Dense connections ile',
                'parameters': '8.1M',
                'size': '33MB',
                'top1_accuracy': '74.4%'
            }
        }
        
        return info.get(model_name, {})
    
    def print_model_comparison(self) -> None:
        """
        Model karşılaştırmasını yazdırır
        """
        print("TRANSFER LEARNING MODEL KARŞILAŞTIRMASI")
        print("=" * 50)
        
        for model_name in self.get_available_models():
            info = self.get_model_info(model_name)
            print(f"\n{model_name.upper()}:")
            print(f"  Açıklama: {info.get('description', 'N/A')}")
            print(f"  Parametre sayısı: {info.get('parameters', 'N/A')}")
            print(f"  Model boyutu: {info.get('size', 'N/A')}")
            print(f"  ImageNet doğruluğu: {info.get('top1_accuracy', 'N/A')}")
    
    def get_trainable_parameters(self, model: keras.Model) -> int:
        """
        Eğitilebilir parametre sayısını döndürür
        
        Args:
            model (keras.Model): Model
            
        Returns:
            int: Eğitilebilir parametre sayısı
        """
        return sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    
    def get_frozen_parameters(self, model: keras.Model) -> int:
        """
        Dondurulmuş parametre sayısını döndürür
        
        Args:
            model (keras.Model): Model
            
        Returns:
            int: Dondurulmuş parametre sayısı
        """
        return sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])

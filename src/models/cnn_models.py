"""
CNN model tanımlama modülü
Sıfırdan CNN modelleri oluşturur
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from typing import Tuple, Optional, Dict, Any
import numpy as np


class CNNModelBuilder:
    """
    CNN modelleri oluşturmak için sınıf
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 10):
        """
        CNNModelBuilder sınıfını başlatır
        
        Args:
            input_shape (Tuple[int, int, int]): Giriş görüntü boyutu
            num_classes (int): Sınıf sayısı
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build_basic_cnn(self, 
                       dropout_rate: float = 0.5,
                       learning_rate: float = 0.001) -> keras.Model:
        """
        Temel CNN modeli oluşturur
        
        Args:
            dropout_rate (float): Dropout oranı
            learning_rate (float): Öğrenme oranı
            
        Returns:
            keras.Model: Oluşturulan model
        """
        model = models.Sequential([
            # İlk konvolüsyon bloğu
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # İkinci konvolüsyon bloğu
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Üçüncü konvolüsyon bloğu
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dördüncü konvolüsyon bloğu
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense katmanları
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            # Çıkış katmanı
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Modeli derle
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_deep_cnn(self, 
                      dropout_rate: float = 0.5,
                      learning_rate: float = 0.001) -> keras.Model:
        """
        Derin CNN modeli oluşturur
        
        Args:
            dropout_rate (float): Dropout oranı
            learning_rate (float): Öğrenme oranı
            
        Returns:
            keras.Model: Oluşturulan model
        """
        model = models.Sequential([
            # İlk konvolüsyon bloğu
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # İkinci konvolüsyon bloğu
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Üçüncü konvolüsyon bloğu
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dördüncü konvolüsyon bloğu
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Beşinci konvolüsyon bloğu
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense katmanları
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            # Çıkış katmanı
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Modeli derle
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_residual_cnn(self, 
                          dropout_rate: float = 0.5,
                          learning_rate: float = 0.001) -> keras.Model:
        """
        Residual CNN modeli oluşturur
        
        Args:
            dropout_rate (float): Dropout oranı
            learning_rate (float): Öğrenme oranı
            
        Returns:
            keras.Model: Oluşturulan model
        """
        def residual_block(x, filters, kernel_size=3, stride=1):
            """Residual block oluşturur"""
            shortcut = x
            
            # İlk konvolüsyon
            x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            # İkinci konvolüsyon
            x = layers.Conv2D(filters, kernel_size, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Shortcut connection
            if stride != 1 or shortcut.shape[-1] != filters:
                shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            
            # Toplama
            x = layers.Add()([x, shortcut])
            x = layers.Activation('relu')(x)
            
            return x
        
        # Giriş katmanı
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Residual bloklar
        x = residual_block(x, 64)
        x = residual_block(x, 64)
        
        x = residual_block(x, 128, stride=2)
        x = residual_block(x, 128)
        
        x = residual_block(x, 256, stride=2)
        x = residual_block(x, 256)
        
        x = residual_block(x, 512, stride=2)
        x = residual_block(x, 512)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense katmanları
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Çıkış katmanı
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        
        # Modeli derle
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_attention_cnn(self, 
                           dropout_rate: float = 0.5,
                           learning_rate: float = 0.001) -> keras.Model:
        """
        Attention mekanizmalı CNN modeli oluşturur
        
        Args:
            dropout_rate (float): Dropout oranı
            learning_rate (float): Öğrenme oranı
            
        Returns:
            keras.Model: Oluşturulan model
        """
        def attention_block(x, filters):
            """Attention block oluşturur"""
            # Channel attention
            avg_pool = layers.GlobalAveragePooling2D()(x)
            max_pool = layers.GlobalMaxPooling2D()(x)
            
            avg_pool = layers.Reshape((1, 1, filters))(avg_pool)
            max_pool = layers.Reshape((1, 1, filters))(max_pool)
            
            avg_pool = layers.Dense(filters // 8, activation='relu')(avg_pool)
            avg_pool = layers.Dense(filters, activation='sigmoid')(avg_pool)
            
            max_pool = layers.Dense(filters // 8, activation='relu')(max_pool)
            max_pool = layers.Dense(filters, activation='sigmoid')(max_pool)
            
            attention = layers.Add()([avg_pool, max_pool])
            attention = layers.Activation('sigmoid')(attention)
            
            x = layers.Multiply()([x, attention])
            
            return x
        
        # Giriş katmanı
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # İlk attention bloğu
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = attention_block(x, 64)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # İkinci attention bloğu
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = attention_block(x, 128)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Üçüncü attention bloğu
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = attention_block(x, 256)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense katmanları
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Çıkış katmanı
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        
        # Modeli derle
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_model_summary(self, model: keras.Model) -> str:
        """
        Model özetini döndürür
        
        Args:
            model (keras.Model): Model
            
        Returns:
            str: Model özeti
        """
        import io
        import sys
        
        # StringIO kullanarak model özetini yakala
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            model.summary()
            summary = buffer.getvalue()
        finally:
            sys.stdout = old_stdout
        
        return summary
    
    def count_parameters(self, model: keras.Model) -> int:
        """
        Model parametre sayısını hesaplar
        
        Args:
            model (keras.Model): Model
            
        Returns:
            int: Toplam parametre sayısı
        """
        return model.count_params()
    
    def get_model_size_mb(self, model: keras.Model) -> float:
        """
        Model boyutunu MB cinsinden hesaplar
        
        Args:
            model (keras.Model): Model
            
        Returns:
            float: Model boyutu (MB)
        """
        # Modeli geçici olarak kaydet
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            model.save(tmp_file.name)
            size_mb = os.path.getsize(tmp_file.name) / (1024 * 1024)
            os.unlink(tmp_file.name)
        
        return size_mb

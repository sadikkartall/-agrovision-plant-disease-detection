"""
Veri üretici modülü
TensorFlow veri pipeline'ı oluşturur
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import cv2
from sklearn.preprocessing import LabelEncoder


class DataGenerator:
    """
    TensorFlow veri üretici sınıfı
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (224, 224),
                 batch_size: int = 32,
                 num_classes: int = 10,
                 augmentation: bool = True):
        """
        DataGenerator sınıfını başlatır
        
        Args:
            image_size (Tuple[int, int]): Görüntü boyutu
            batch_size (int): Batch boyutu
            num_classes (int): Sınıf sayısı
            augmentation (bool): Veri artırma uygulanacak mı
        """
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.label_encoder = LabelEncoder()
    
    def create_augmentation_layer(self) -> tf.keras.Sequential:
        """
        Veri artırma katmanı oluşturur
        
        Returns:
            tf.keras.Sequential: Artırma katmanı
        """
        if not self.augmentation:
            return tf.keras.Sequential([])
        
        augmentation_layers = tf.keras.Sequential([
            # Rastgele yatay çevirme
            tf.keras.layers.RandomFlip("horizontal"),
            
            # Rastgele döndürme
            tf.keras.layers.RandomRotation(0.2),
            
            # Rastgele zoom
            tf.keras.layers.RandomZoom(0.2),
            
            # Rastgele parlaklık ve kontrast
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomContrast(0.2),
            
            # Rastgele çevirme (hue, saturation)
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ])
        
        return augmentation_layers
    
    def preprocess_image(self, image_path: str) -> tf.Tensor:
        """
        Görüntüyü önişler
        
        Args:
            image_path (str): Görüntü dosya yolu
            
        Returns:
            tf.Tensor: Önişlenmiş görüntü
        """
        # Görüntüyü oku
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        
        # Boyutu değiştir
        image = tf.image.resize(image, self.image_size)
        
        # Float'a çevir (0-255 aralığında bırak).
        # Model-özel preprocess_input katmanı transfer learning tarafında uygulanır.
        image = tf.cast(image, tf.float32)
        
        return image
    
    def create_dataset_from_dataframe(self, 
                                    df: pd.DataFrame,
                                    image_column: str = 'image_path',
                                    label_column: str = 'numeric_label',
                                    shuffle: bool = True,
                                    cache: bool = True) -> tf.data.Dataset:
        """
        DataFrame'den TensorFlow Dataset oluşturur
        
        Args:
            df (pd.DataFrame): Veri DataFrame'i
            image_column (str): Görüntü yolu sütunu
            label_column (str): Etiket sütunu
            shuffle (bool): Veriyi karıştır
            cache (bool): Veriyi önbelleğe al
            
        Returns:
            tf.data.Dataset: TensorFlow Dataset
        """
        # Görüntü yolları ve etiketleri al
        image_paths = df[image_column].values
        labels = df[label_column].values
        
        # Dataset oluştur
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        # Görüntüleri işle
        dataset = dataset.map(
            lambda x, y: (self.preprocess_image(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Etiketleri one-hot encode et
        dataset = dataset.map(
            lambda x, y: (x, tf.one_hot(y, self.num_classes)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Veri artırma uygula
        if self.augmentation:
            augmentation_layer = self.create_augmentation_layer()
            dataset = dataset.map(
                lambda x, y: (augmentation_layer(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Performans optimizasyonları
        if cache:
            dataset = dataset.cache()
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        # Batch oluştur
        dataset = dataset.batch(self.batch_size)
        
        # Prefetch
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def create_dataset_from_directory(self, 
                                    directory: str,
                                    shuffle: bool = True,
                                    cache: bool = True) -> tf.data.Dataset:
        """
        Dizinden TensorFlow Dataset oluşturur
        
        Args:
            directory (str): Veri dizini
            shuffle (bool): Veriyi karıştır
            cache (bool): Veriyi önbelleğe al
            
        Returns:
            tf.data.Dataset: TensorFlow Dataset
        """
        # ImageDataGenerator oluştur
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            validation_split=0.0  # Validation split kullanmıyoruz
        )
        
        # Dataset oluştur
        dataset = datagen.flow_from_directory(
            directory,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=shuffle
        )
        
        return dataset
    
    def create_train_val_test_datasets(self, 
                                     train_df: pd.DataFrame,
                                     val_df: pd.DataFrame,
                                     test_df: pd.DataFrame,
                                     image_column: str = 'image_path',
                                     label_column: str = 'numeric_label') -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Eğitim, doğrulama ve test dataset'lerini oluşturur
        
        Args:
            train_df (pd.DataFrame): Eğitim verisi
            val_df (pd.DataFrame): Doğrulama verisi
            test_df (pd.DataFrame): Test verisi
            image_column (str): Görüntü yolu sütunu
            label_column (str): Etiket sütunu
            
        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: (train, val, test) dataset'leri
        """
        # Eğitim dataset'i (artırma ile)
        train_dataset = self.create_dataset_from_dataframe(
            train_df, image_column, label_column, shuffle=True, cache=True
        )
        
        # Doğrulama dataset'i (artırma olmadan)
        val_augmentation = self.augmentation
        self.augmentation = False
        val_dataset = self.create_dataset_from_dataframe(
            val_df, image_column, label_column, shuffle=False, cache=True
        )
        self.augmentation = val_augmentation
        
        # Test dataset'i (artırma olmadan)
        test_augmentation = self.augmentation
        self.augmentation = False
        test_dataset = self.create_dataset_from_dataframe(
            test_df, image_column, label_column, shuffle=False, cache=True
        )
        self.augmentation = test_augmentation
        
        return train_dataset, val_dataset, test_dataset
    
    def get_dataset_info(self, dataset: tf.data.Dataset) -> Dict[str, Any]:
        """
        Dataset bilgilerini döndürür
        
        Args:
            dataset (tf.data.Dataset): Dataset
            
        Returns:
            Dict[str, Any]: Dataset bilgileri
        """
        # Batch sayısını hesapla
        num_batches = tf.data.experimental.cardinality(dataset).numpy()
        
        # Toplam örnek sayısı
        total_samples = num_batches * self.batch_size
        
        # İlk batch'i al ve şekil bilgilerini al
        for batch in dataset.take(1):
            images, labels = batch
            image_shape = images.shape
            label_shape = labels.shape
            break
        
        info = {
            'num_batches': num_batches,
            'batch_size': self.batch_size,
            'total_samples': total_samples,
            'image_shape': image_shape,
            'label_shape': label_shape,
            'num_classes': self.num_classes
        }
        
        return info
    
    def print_dataset_summary(self, 
                            train_dataset: tf.data.Dataset,
                            val_dataset: tf.data.Dataset,
                            test_dataset: tf.data.Dataset) -> None:
        """
        Dataset özetini yazdırır
        
        Args:
            train_dataset (tf.data.Dataset): Eğitim dataset'i
            val_dataset (tf.data.Dataset): Doğrulama dataset'i
            test_dataset (tf.data.Dataset): Test dataset'i
        """
        print("VERİ SETİ ÖZETİ")
        print("=" * 40)
        
        datasets = {
            'Eğitim': train_dataset,
            'Doğrulama': val_dataset,
            'Test': test_dataset
        }
        
        for name, dataset in datasets.items():
            info = self.get_dataset_info(dataset)
            print(f"\n{name} Seti:")
            print(f"  Batch sayısı: {info['num_batches']}")
            print(f"  Batch boyutu: {info['batch_size']}")
            print(f"  Toplam örnek: {info['total_samples']}")
            print(f"  Görüntü boyutu: {info['image_shape']}")
            print(f"  Etiket boyutu: {info['label_shape']}")
    
    def create_sample_batch(self, dataset: tf.data.Dataset, num_samples: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Örnek batch oluşturur
        
        Args:
            dataset (tf.data.Dataset): Dataset
            num_samples (int): Örnek sayısı
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (görüntüler, etiketler)
        """
        # İlk batch'i al
        for batch in dataset.take(1):
            images, labels = batch
            break
        
        # İstenen sayıda örnek al
        sample_images = images[:num_samples].numpy()
        sample_labels = labels[:num_samples].numpy()
        
        return sample_images, sample_labels

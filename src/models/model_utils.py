"""
Model yardımcı fonksiyonları
Model eğitimi, kaydetme ve yükleme için yardımcı fonksiyonlar
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any, Optional, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class ModelUtils:
    """
    Model yardımcı fonksiyonları için sınıf
    """
    
    def __init__(self):
        """
        ModelUtils sınıfını başlatır
        """
        self.history = None
        self.model = None
    
    def create_callbacks(self, 
                        model_path: str,
                        patience: int = 10,
                        monitor: str = 'val_accuracy',
                        mode: str = 'max',
                        reduce_lr_patience: int = 5,
                        reduce_lr_factor: float = 0.5) -> List[keras.callbacks.Callback]:
        """
        Eğitim için callback'leri oluşturur
        
        Args:
            model_path (str): Model kayıt yolu
            patience (int): Early stopping sabrı
            monitor (str): İzlenecek metrik
            mode (str): Mod ('max' veya 'min')
            reduce_lr_patience (int): Learning rate azaltma sabrı
            reduce_lr_factor (float): Learning rate azaltma faktörü
            
        Returns:
            List[keras.callbacks.Callback]: Callback listesi
        """
        callbacks = [
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor=monitor,
                mode=mode,
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                mode=mode,
                factor=reduce_lr_factor,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV logger
            keras.callbacks.CSVLogger(
                filename=model_path.replace('.h5', '_training.log'),
                separator=',',
                append=False
            )
        ]
        
        return callbacks
    
    def train_model(self, 
                   model: keras.Model,
                   train_data: tf.data.Dataset,
                   val_data: tf.data.Dataset,
                   epochs: int = 100,
                   callbacks: Optional[List[keras.callbacks.Callback]] = None,
                   verbose: int = 1) -> keras.callbacks.History:
        """
        Modeli eğitir
        
        Args:
            model (keras.Model): Eğitilecek model
            train_data (tf.data.Dataset): Eğitim verisi
            val_data (tf.data.Dataset): Doğrulama verisi
            epochs (int): Eğitim epoch sayısı
            callbacks (Optional[List[keras.callbacks.Callback]]): Callback'ler
            verbose (int): Verbose seviyesi
            
        Returns:
            keras.callbacks.History: Eğitim geçmişi
        """
        print(f"Model eğitimi başlıyor...")
        print(f"Epoch sayısı: {epochs}")
        print(f"Eğitim verisi: {len(train_data)} batch")
        print(f"Doğrulama verisi: {len(val_data)} batch")
        
        # Modeli eğit
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.history = history
        self.model = model
        
        print("Model eğitimi tamamlandı!")
        
        return history
    
    def save_model(self, 
                  model: keras.Model, 
                  filepath: str,
                  save_format: str = 'h5') -> None:
        """
        Modeli kaydeder
        
        Args:
            model (keras.Model): Kaydedilecek model
            filepath (str): Kayıt dosya yolu
            save_format (str): Kayıt formatı ('h5' veya 'tf')
        """
        # Dizin oluştur
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Modeli kaydet
        if save_format == 'h5':
            model.save(filepath)
        elif save_format == 'tf':
            model.save(filepath, save_format='tf')
        else:
            raise ValueError(f"Desteklenmeyen format: {save_format}")
        
        print(f"Model kaydedildi: {filepath}")
    
    def load_model(self, filepath: str) -> keras.Model:
        """
        Modeli yükler
        
        Args:
            filepath (str): Model dosya yolu
            
        Returns:
            keras.Model: Yüklenen model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {filepath}")
        
        model = keras.models.load_model(filepath)
        print(f"Model yüklendi: {filepath}")
        
        return model
    
    def save_model_info(self, 
                       model: keras.Model,
                       filepath: str,
                       additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Model bilgilerini kaydeder
        
        Args:
            model (keras.Model): Model
            filepath (str): Kayıt dosya yolu
            additional_info (Optional[Dict[str, Any]]): Ek bilgiler
        """
        # Model bilgilerini topla
        model_info = {
            'model_name': model.name,
            'total_params': model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
            'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]),
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'created_at': datetime.now().isoformat(),
            'additional_info': additional_info or {}
        }
        
        # JSON olarak kaydet
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print(f"Model bilgileri kaydedildi: {filepath}")
    
    def plot_training_history(self, 
                             history: keras.callbacks.History,
                             save_path: Optional[str] = None,
                             show_plot: bool = True) -> None:
        """
        Eğitim geçmişini görselleştirir
        
        Args:
            history (keras.callbacks.History): Eğitim geçmişi
            save_path (Optional[str]): Kayıt yolu
            show_plot (bool): Grafiği göster
        """
        # Grafik boyutunu ayarla
        plt.figure(figsize=(15, 5))
        
        # 1. Accuracy grafiği
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
        plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
        plt.title('Model Doğruluğu')
        plt.xlabel('Epoch')
        plt.ylabel('Doğruluk')
        plt.legend()
        plt.grid(True)
        
        # 2. Loss grafiği
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Eğitim Kaybı')
        plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
        plt.title('Model Kaybı')
        plt.xlabel('Epoch')
        plt.ylabel('Kayıp')
        plt.legend()
        plt.grid(True)
        
        # 3. Learning rate grafiği (varsa)
        plt.subplot(1, 3, 3)
        if 'lr' in history.history:
            plt.plot(history.history['lr'])
            plt.title('Öğrenme Oranı')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
        else:
            plt.text(0.5, 0.5, 'Learning Rate\nBilgisi Yok', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Öğrenme Oranı')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Kaydet
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Eğitim grafiği kaydedildi: {save_path}")
        
        # Göster
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def evaluate_model(self, 
                      model: keras.Model,
                      test_data: tf.data.Dataset,
                      class_names: List[str]) -> Dict[str, float]:
        """
        Modeli değerlendirir
        
        Args:
            model (keras.Model): Değerlendirilecek model
            test_data (tf.data.Dataset): Test verisi
            class_names (List[str]): Sınıf isimleri
            
        Returns:
            Dict[str, float]: Değerlendirme metrikleri
        """
        print("Model değerlendiriliyor...")
        
        # Modeli değerlendir
        results = model.evaluate(test_data, verbose=1)
        
        # Metrik isimlerini al
        metric_names = model.metrics_names
        
        # Sonuçları sözlük olarak döndür
        evaluation_results = dict(zip(metric_names, results))
        
        print("Değerlendirme sonuçları:")
        for metric, value in evaluation_results.items():
            print(f"  {metric}: {value:.4f}")
        
        return evaluation_results
    
    def predict_and_analyze(self, 
                           model: keras.Model,
                           test_data: tf.data.Dataset,
                           class_names: List[str],
                           num_samples: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tahmin yapar ve analiz eder
        
        Args:
            model (keras.Model): Model
            test_data (tf.data.Dataset): Test verisi
            class_names (List[str]): Sınıf isimleri
            num_samples (int): Analiz edilecek örnek sayısı
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (tahminler, etiketler, olasılıklar)
        """
        print("Tahmin yapılıyor...")
        
        # Tahmin yap
        predictions = model.predict(test_data, verbose=1)
        
        # Sınıf tahminleri
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Olasılıklar
        probabilities = np.max(predictions, axis=1)
        
        # Gerçek etiketleri al
        true_labels = []
        for images, labels in test_data.take(num_samples):
            true_labels.extend(np.argmax(labels.numpy(), axis=1))
        
        true_labels = np.array(true_labels)
        
        print(f"Tahmin edilen sınıflar: {predicted_classes[:num_samples]}")
        print(f"Gerçek sınıflar: {true_labels[:num_samples]}")
        print(f"Olasılıklar: {probabilities[:num_samples]}")
        
        return predicted_classes, true_labels, probabilities
    
    def create_confusion_matrix(self, 
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               class_names: List[str],
                               save_path: Optional[str] = None,
                               show_plot: bool = True) -> np.ndarray:
        """
        Karışıklık matrisi oluşturur
        
        Args:
            y_true (np.ndarray): Gerçek etiketler
            y_pred (np.ndarray): Tahmin edilen etiketler
            class_names (List[str]): Sınıf isimleri
            save_path (Optional[str]): Kayıt yolu
            show_plot (bool): Grafiği göster
            
        Returns:
            np.ndarray: Karışıklık matrisi
        """
        from sklearn.metrics import confusion_matrix
        
        # Karışıklık matrisini hesapla
        cm = confusion_matrix(y_true, y_pred)
        
        # Grafik oluştur
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Karışıklık Matrisi')
        plt.xlabel('Tahmin Edilen Sınıf')
        plt.ylabel('Gerçek Sınıf')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Kaydet
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Karışıklık matrisi kaydedildi: {save_path}")
        
        # Göster
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return cm
    
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

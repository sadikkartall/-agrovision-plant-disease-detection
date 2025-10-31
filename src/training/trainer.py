"""
Model eğitici modülü
Modelleri eğitir ve değerlendirir
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import matplotlib.pyplot as plt

from ..models.cnn_models import CNNModelBuilder
from ..models.transfer_learning import TransferLearningModel
from ..models.model_utils import ModelUtils
from .data_generator import DataGenerator


class ModelTrainer:
    """
    Model eğitici sınıfı
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 output_dir: str = "models/saved"):
        """
        ModelTrainer sınıfını başlatır
        
        Args:
            config (Dict[str, Any]): Eğitim konfigürasyonu
            output_dir (str): Çıktı dizini
        """
        self.config = config
        self.output_dir = output_dir
        self.model_utils = ModelUtils()
        self.training_history = None
        self.best_model = None
        
        # Çıktı dizinini oluştur
        os.makedirs(output_dir, exist_ok=True)
    
    def create_model(self, model_type: str, **kwargs) -> keras.Model:
        """
        Model oluşturur
        
        Args:
            model_type (str): Model türü
            **kwargs: Model parametreleri
            
        Returns:
            keras.Model: Oluşturulan model
        """
        input_shape = self.config.get('input_shape', (224, 224, 3))
        num_classes = self.config.get('num_classes', 10)
        
        if model_type == 'basic_cnn':
            builder = CNNModelBuilder(input_shape, num_classes)
            model = builder.build_basic_cnn(**kwargs)
            
        elif model_type == 'deep_cnn':
            builder = CNNModelBuilder(input_shape, num_classes)
            model = builder.build_deep_cnn(**kwargs)
            
        elif model_type == 'residual_cnn':
            builder = CNNModelBuilder(input_shape, num_classes)
            model = builder.build_residual_cnn(**kwargs)
            
        elif model_type == 'attention_cnn':
            builder = CNNModelBuilder(input_shape, num_classes)
            model = builder.build_attention_cnn(**kwargs)
            
        elif model_type == 'mobilenet_v2':
            transfer_model = TransferLearningModel(input_shape, num_classes, 'mobilenet_v2')
            model = transfer_model.build_feature_extraction_model(**kwargs)
            
        elif model_type == 'resnet50':
            transfer_model = TransferLearningModel(input_shape, num_classes, 'resnet50')
            model = transfer_model.build_feature_extraction_model(**kwargs)
            
        elif model_type == 'efficientnet_b0':
            transfer_model = TransferLearningModel(input_shape, num_classes, 'efficientnet_b0')
            model = transfer_model.build_feature_extraction_model(**kwargs)
            
        else:
            raise ValueError(f"Desteklenmeyen model türü: {model_type}")
        
        return model
    
    def train_model(self, 
                   model: keras.Model,
                   train_dataset: tf.data.Dataset,
                   val_dataset: tf.data.Dataset,
                   model_name: str,
                   epochs: int = 100) -> keras.Model:
        """
        Modeli eğitir
        
        Args:
            model (keras.Model): Eğitilecek model
            train_dataset (tf.data.Dataset): Eğitim verisi
            val_dataset (tf.data.Dataset): Doğrulama verisi
            model_name (str): Model adı
            epochs (int): Eğitim epoch sayısı
            
        Returns:
            keras.Model: Eğitilmiş model
        """
        print(f"\n{model_name} modeli eğitiliyor...")
        print("=" * 50)
        
        # Model dosya yolu
        model_path = os.path.join(self.output_dir, f"{model_name}_best.h5")
        
        # Callback'leri oluştur
        callbacks = self.model_utils.create_callbacks(
            model_path=model_path,
            patience=self.config.get('patience', 10),
            monitor=self.config.get('monitor', 'val_accuracy'),
            mode=self.config.get('mode', 'max')
        )
        
        # Modeli eğit
        history = self.model_utils.train_model(
            model=model,
            train_data=train_dataset,
            val_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        
        # En iyi modeli yükle
        best_model = self.model_utils.load_model(model_path)
        
        # Eğitim geçmişini kaydet
        self.training_history = history
        self.best_model = best_model
        
        # Eğitim grafiğini kaydet
        plot_path = os.path.join(self.output_dir, f"{model_name}_training_history.png")
        self.model_utils.plot_training_history(
            history, 
            save_path=plot_path, 
            show_plot=False
        )
        
        # Model bilgilerini kaydet
        info_path = os.path.join(self.output_dir, f"{model_name}_info.json")
        self.model_utils.save_model_info(
            best_model,
            info_path,
            additional_info={
                'model_type': model_name,
                'epochs_trained': len(history.history['loss']),
                'best_val_accuracy': max(history.history['val_accuracy']),
                'best_val_loss': min(history.history['val_loss']),
                'final_train_accuracy': history.history['accuracy'][-1],
                'final_val_accuracy': history.history['val_accuracy'][-1]
            }
        )
        
        print(f"\n{model_name} modeli eğitimi tamamlandı!")
        print(f"En iyi doğrulama doğruluğu: {max(history.history['val_accuracy']):.4f}")
        print(f"Model kaydedildi: {model_path}")
        
        return best_model
    
    def evaluate_model(self, 
                      model: keras.Model,
                      test_dataset: tf.data.Dataset,
                      class_names: List[str],
                      model_name: str) -> Dict[str, float]:
        """
        Modeli değerlendirir
        
        Args:
            model (keras.Model): Değerlendirilecek model
            test_dataset (tf.data.Dataset): Test verisi
            class_names (List[str]): Sınıf isimleri
            model_name (str): Model adı
            
        Returns:
            Dict[str, float]: Değerlendirme sonuçları
        """
        print(f"\n{model_name} modeli değerlendiriliyor...")
        print("=" * 50)
        
        # Modeli değerlendir
        evaluation_results = self.model_utils.evaluate_model(
            model, test_dataset, class_names
        )
        
        # Tahmin yap ve analiz et
        predictions, true_labels, probabilities = self.model_utils.predict_and_analyze(
            model, test_dataset, class_names, num_samples=10
        )
        
        # Karışıklık matrisi oluştur
        cm_path = os.path.join(self.output_dir, f"{model_name}_confusion_matrix.png")
        cm = self.model_utils.create_confusion_matrix(
            true_labels, predictions, class_names,
            save_path=cm_path, show_plot=False
        )
        
        # Değerlendirme sonuçlarını kaydet
        results_path = os.path.join(self.output_dir, f"{model_name}_evaluation.json")
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"Değerlendirme sonuçları kaydedildi: {results_path}")
        
        return evaluation_results
    
    def compare_models(self, 
                      models: Dict[str, keras.Model],
                      test_dataset: tf.data.Dataset,
                      class_names: List[str]) -> pd.DataFrame:
        """
        Modelleri karşılaştırır
        
        Args:
            models (Dict[str, keras.Model]): Model sözlüğü
            test_dataset (tf.data.Dataset): Test verisi
            class_names (List[str]): Sınıf isimleri
            
        Returns:
            pd.DataFrame: Karşılaştırma sonuçları
        """
        print("\nModeller karşılaştırılıyor...")
        print("=" * 50)
        
        comparison_results = []
        
        for model_name, model in models.items():
            print(f"\n{model_name} değerlendiriliyor...")
            
            # Modeli değerlendir
            results = self.model_utils.evaluate_model(model, test_dataset, class_names)
            
            # Parametre sayısını al
            total_params = model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            
            # Sonuçları kaydet
            result = {
                'Model': model_name,
                'Test Accuracy': results.get('accuracy', 0),
                'Test Loss': results.get('loss', 0),
                'Top-3 Accuracy': 0,  # Geçici olarak 0
                'Total Parameters': total_params,
                'Trainable Parameters': trainable_params
            }
            
            comparison_results.append(result)
        
        # DataFrame oluştur
        comparison_df = pd.DataFrame(comparison_results)
        
        # Sonuçları kaydet
        comparison_path = os.path.join(self.output_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        
        # En iyi modeli belirle
        best_model_name = comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Model']
        print(f"\nEn iyi model: {best_model_name}")
        print(f"Test doğruluğu: {comparison_df['Test Accuracy'].max():.4f}")
        
        return comparison_df
    
    def create_training_report(self, 
                             comparison_df: pd.DataFrame,
                             class_names: List[str]) -> None:
        """
        Eğitim raporu oluşturur
        
        Args:
            comparison_df (pd.DataFrame): Model karşılaştırma sonuçları
            class_names (List[str]): Sınıf isimleri
        """
        print("\nEğitim raporu oluşturuluyor...")
        
        # Rapor dosyası
        report_path = os.path.join(self.output_dir, "training_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("BİTKİ HASTALIK TESPİT SİSTEMİ - EĞİTİM RAPORU\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Toplam Sınıf Sayısı: {len(class_names)}\n")
            f.write(f"Eğitilen Model Sayısı: {len(comparison_df)}\n\n")
            
            f.write("SINIFLAR:\n")
            f.write("-" * 20 + "\n")
            for i, class_name in enumerate(class_names):
                f.write(f"{i+1:2d}. {class_name}\n")
            
            f.write("\nMODEL KARŞILAŞTIRMASI:\n")
            f.write("-" * 30 + "\n")
            f.write(comparison_df.to_string(index=False))
            
            f.write("\n\nEN İYİ MODEL:\n")
            f.write("-" * 20 + "\n")
            best_model = comparison_df.loc[comparison_df['Test Accuracy'].idxmax()]
            f.write(f"Model: {best_model['Model']}\n")
            f.write(f"Test Doğruluğu: {best_model['Test Accuracy']:.4f}\n")
            f.write(f"Test Kaybı: {best_model['Test Loss']:.4f}\n")
            f.write(f"Top-3 Doğruluğu: {best_model['Top-3 Accuracy']:.4f}\n")
            f.write(f"Toplam Parametre: {best_model['Total Parameters']:,}\n")
            f.write(f"Eğitilebilir Parametre: {best_model['Trainable Parameters']:,}\n")
            
            f.write("\n\nÖNERİLER:\n")
            f.write("-" * 20 + "\n")
            f.write("1. En iyi performans gösteren modeli üretim ortamında kullanın\n")
            f.write("2. Daha fazla veri toplayarak model performansını artırabilirsiniz\n")
            f.write("3. Veri artırma tekniklerini daha etkili kullanabilirsiniz\n")
            f.write("4. Model mimarisini optimize edebilirsiniz\n")
            f.write("5. Hyperparameter tuning yapabilirsiniz\n")
        
        print(f"Eğitim raporu kaydedildi: {report_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Eğitim özetini döndürür
        
        Returns:
            Dict[str, Any]: Eğitim özeti
        """
        if self.training_history is None:
            return {}
        
        summary = {
            'epochs_trained': len(self.training_history.history['loss']),
            'final_train_accuracy': self.training_history.history['accuracy'][-1],
            'final_val_accuracy': self.training_history.history['val_accuracy'][-1],
            'best_val_accuracy': max(self.training_history.history['val_accuracy']),
            'final_train_loss': self.training_history.history['loss'][-1],
            'final_val_loss': self.training_history.history['val_loss'][-1],
            'best_val_loss': min(self.training_history.history['val_loss'])
        }
        
        return summary

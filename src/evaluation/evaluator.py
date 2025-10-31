"""
Model değerlendirici modülü
Modelleri kapsamlı olarak değerlendirir
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .metrics import MetricsCalculator
from .visualizer import ResultsVisualizer
from .report_generator import ReportGenerator


class ModelEvaluator:
    """
    Model değerlendirici sınıfı
    """
    
    def __init__(self, 
                 class_names: List[str],
                 output_dir: str = "reports"):
        """
        ModelEvaluator sınıfını başlatır
        
        Args:
            class_names (List[str]): Sınıf isimleri
            output_dir (str): Çıktı dizini
        """
        self.class_names = class_names
        self.output_dir = output_dir
        self.metrics_calculator = MetricsCalculator(class_names)
        self.visualizer = ResultsVisualizer(class_names)
        self.report_generator = ReportGenerator(class_names, output_dir)
        
        # Çıktı dizinini oluştur
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_model(self, 
                      model: keras.Model,
                      test_data: tf.data.Dataset,
                      model_name: str,
                      save_results: bool = True) -> Dict[str, Any]:
        """
        Modeli kapsamlı olarak değerlendirir
        
        Args:
            model (keras.Model): Değerlendirilecek model
            test_data (tf.data.Dataset): Test verisi
            model_name (str): Model adı
            save_results (bool): Sonuçları kaydet
            
        Returns:
            Dict[str, Any]: Değerlendirme sonuçları
        """
        print(f"\n{model_name} modeli değerlendiriliyor...")
        print("=" * 50)
        
        # Tahmin yap
        print("Tahmin yapılıyor...")
        predictions = model.predict(test_data, verbose=1)
        
        # Sınıf tahminleri ve olasılıkları
        y_pred = np.argmax(predictions, axis=1)
        y_pred_proba = predictions
        
        # Gerçek etiketleri al
        y_true = []
        for images, labels in test_data:
            y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_true = np.array(y_true)
        
        print(f"Test örnek sayısı: {len(y_true)}")
        print(f"Tahmin edilen sınıf sayısı: {len(np.unique(y_pred))}")
        
        # Metrikleri hesapla
        print("Metrikler hesaplanıyor...")
        metrics = self.metrics_calculator.calculate_all_metrics(
            y_true, y_pred, y_pred_proba, model
        )
        
        # Görselleştirmeler oluştur
        if save_results:
            print("Görselleştirmeler oluşturuluyor...")
            self._create_visualizations(metrics, model_name)
        
        # Rapor oluştur
        if save_results:
            print("Rapor oluşturuluyor...")
            self.report_generator.generate_model_report(
                model_name, metrics, model
            )
        
        # Sonuçları kaydet
        if save_results:
            self._save_evaluation_results(model_name, metrics)
        
        print(f"{model_name} değerlendirmesi tamamlandı!")
        
        return metrics
    
    def compare_models(self, 
                      models: Dict[str, keras.Model],
                      test_data: tf.data.Dataset,
                      save_results: bool = True) -> pd.DataFrame:
        """
        Modelleri karşılaştırır
        
        Args:
            models (Dict[str, keras.Model]): Model sözlüğü
            test_data (tf.data.Dataset): Test verisi
            save_results (bool): Sonuçları kaydet
            
        Returns:
            pd.DataFrame: Karşılaştırma sonuçları
        """
        print("\nModeller karşılaştırılıyor...")
        print("=" * 50)
        
        comparison_results = []
        
        for model_name, model in models.items():
            print(f"\n{model_name} değerlendiriliyor...")
            
            # Modeli değerlendir
            metrics = self.evaluate_model(
                model, test_data, model_name, save_results=False
            )
            
            # Temel metrikleri al
            basic_metrics = metrics.get('basic_metrics', {})
            model_complexity = metrics.get('model_complexity', {})
            
            # Sonuçları kaydet
            result = {
                'Model': model_name,
                'Test Accuracy': basic_metrics.get('accuracy', 0),
                'Test Precision (Macro)': basic_metrics.get('precision_macro', 0),
                'Test Recall (Macro)': basic_metrics.get('recall_macro', 0),
                'Test F1-Score (Macro)': basic_metrics.get('f1_macro', 0),
                'Total Parameters': model_complexity.get('total_parameters', 0),
                'Trainable Parameters': model_complexity.get('trainable_parameters', 0),
                'Model Size (MB)': model_complexity.get('model_size_mb', 0)
            }
            
            comparison_results.append(result)
        
        # DataFrame oluştur
        comparison_df = pd.DataFrame(comparison_results)
        
        # En iyi modeli belirle
        best_model = comparison_df.loc[comparison_df['Test Accuracy'].idxmax()]
        print(f"\nEn iyi model: {best_model['Model']}")
        print(f"Test doğruluğu: {best_model['Test Accuracy']:.4f}")
        
        # Karşılaştırma görselleştirmesi
        if save_results:
            self._create_comparison_visualizations(comparison_df)
        
        # Karşılaştırma raporu
        if save_results:
            self.report_generator.generate_comparison_report(comparison_df)
        
        return comparison_df
    
    def analyze_errors(self, 
                      model: keras.Model,
                      test_data: tf.data.Dataset,
                      model_name: str,
                      num_samples: int = 20) -> Dict[str, Any]:
        """
        Model hatalarını analiz eder
        
        Args:
            model (keras.Model): Model
            test_data (tf.data.Dataset): Test verisi
            model_name (str): Model adı
            num_samples (int): Analiz edilecek örnek sayısı
            
        Returns:
            Dict[str, Any]: Hata analizi sonuçları
        """
        print(f"\n{model_name} hata analizi yapılıyor...")
        
        # Tahmin yap
        predictions = model.predict(test_data, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_pred_proba = predictions
        
        # Gerçek etiketleri al
        y_true = []
        for images, labels in test_data:
            y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_true = np.array(y_true)
        
        # Hatalı tahminleri bul
        wrong_predictions = y_true != y_pred
        wrong_indices = np.where(wrong_predictions)[0]
        
        print(f"Toplam hatalı tahmin: {len(wrong_indices)}")
        print(f"Hata oranı: {len(wrong_indices) / len(y_true) * 100:.2f}%")
        
        # Hata analizi
        error_analysis = {
            'total_errors': len(wrong_indices),
            'error_rate': len(wrong_indices) / len(y_true),
            'correct_predictions': len(y_true) - len(wrong_indices),
            'accuracy': (len(y_true) - len(wrong_indices)) / len(y_true)
        }
        
        # En çok karıştırılan sınıf çiftleri
        confusion_pairs = {}
        for i in wrong_indices:
            true_class = self.class_names[y_true[i]]
            pred_class = self.class_names[y_pred[i]]
            pair = f"{true_class} -> {pred_class}"
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        
        # En çok karıştırılan çiftleri sırala
        most_confused = sorted(confusion_pairs.items(), 
                              key=lambda x: x[1], reverse=True)[:10]
        
        error_analysis['most_confused_pairs'] = most_confused
        
        # Güven skorları analizi
        wrong_confidences = y_pred_proba[wrong_indices]
        correct_confidences = y_pred_proba[~wrong_predictions]
        
        error_analysis['wrong_prediction_confidence'] = {
            'mean': np.mean(np.max(wrong_confidences, axis=1)),
            'std': np.std(np.max(wrong_confidences, axis=1)),
            'min': np.min(np.max(wrong_confidences, axis=1)),
            'max': np.max(np.max(wrong_confidences, axis=1))
        }
        
        error_analysis['correct_prediction_confidence'] = {
            'mean': np.mean(np.max(correct_confidences, axis=1)),
            'std': np.std(np.max(correct_confidences, axis=1)),
            'min': np.min(np.max(correct_confidences, axis=1)),
            'max': np.max(np.max(correct_confidences, axis=1))
        }
        
        # Hata analizi görselleştirmesi
        self.visualizer.plot_error_analysis(
            y_true, y_pred, y_pred_proba, self.class_names,
            num_samples=num_samples,
            save_path=os.path.join(self.output_dir, f"{model_name}_error_analysis.png"),
            show_plot=False
        )
        
        # Sonuçları kaydet
        error_analysis_path = os.path.join(self.output_dir, f"{model_name}_error_analysis.json")
        with open(error_analysis_path, 'w', encoding='utf-8') as f:
            json.dump(error_analysis, f, indent=2, ensure_ascii=False)
        
        print(f"Hata analizi tamamlandı: {error_analysis_path}")
        
        return error_analysis
    
    def _create_visualizations(self, 
                              metrics: Dict[str, Any], 
                              model_name: str) -> None:
        """
        Görselleştirmeleri oluşturur
        
        Args:
            metrics (Dict[str, Any]): Hesaplanan metrikler
            model_name (str): Model adı
        """
        # Karışıklık matrisi
        if 'confusion_matrix_metrics' in metrics:
            cm = metrics['confusion_matrix_metrics']['confusion_matrix']
            self.visualizer.plot_confusion_matrix(
                cm,
                save_path=os.path.join(self.output_dir, f"{model_name}_confusion_matrix.png"),
                show_plot=False
            )
            
            self.visualizer.plot_normalized_confusion_matrix(
                cm,
                save_path=os.path.join(self.output_dir, f"{model_name}_confusion_matrix_normalized.png"),
                show_plot=False
            )
        
        # ROC eğrileri
        if 'roc_metrics' in metrics and 'roc_curves' in metrics['roc_metrics']:
            self.visualizer.plot_roc_curves(
                metrics['roc_metrics']['roc_curves'],
                save_path=os.path.join(self.output_dir, f"{model_name}_roc_curves.png"),
                show_plot=False
            )
        
        # Precision-Recall eğrileri
        if 'roc_metrics' in metrics and 'pr_curves' in metrics['roc_metrics']:
            self.visualizer.plot_precision_recall_curves(
                metrics['roc_metrics']['pr_curves'],
                save_path=os.path.join(self.output_dir, f"{model_name}_precision_recall_curves.png"),
                show_plot=False
            )
        
        # Sınıf bazında metrikler
        if 'per_class_metrics' in metrics:
            self.visualizer.plot_per_class_metrics(
                metrics['per_class_metrics'],
                save_path=os.path.join(self.output_dir, f"{model_name}_per_class_metrics.png"),
                show_plot=False
            )
    
    def _create_comparison_visualizations(self, 
                                        comparison_df: pd.DataFrame) -> None:
        """
        Karşılaştırma görselleştirmelerini oluşturur
        
        Args:
            comparison_df (pd.DataFrame): Karşılaştırma verisi
        """
        # Doğruluk karşılaştırması
        self.visualizer.plot_model_comparison(
            comparison_df,
            metric='Test Accuracy',
            title='Model Doğruluk Karşılaştırması',
            save_path=os.path.join(self.output_dir, 'model_accuracy_comparison.png'),
            show_plot=False
        )
        
        # F1-Score karşılaştırması
        self.visualizer.plot_model_comparison(
            comparison_df,
            metric='Test F1-Score (Macro)',
            title='Model F1-Score Karşılaştırması',
            save_path=os.path.join(self.output_dir, 'model_f1_comparison.png'),
            show_plot=False
        )
        
        # Parametre sayısı karşılaştırması
        self.visualizer.plot_model_comparison(
            comparison_df,
            metric='Total Parameters',
            title='Model Parametre Sayısı Karşılaştırması',
            save_path=os.path.join(self.output_dir, 'model_parameters_comparison.png'),
            show_plot=False
        )
    
    def _save_evaluation_results(self, 
                                model_name: str, 
                                metrics: Dict[str, Any]) -> None:
        """
        Değerlendirme sonuçlarını kaydeder
        
        Args:
            model_name (str): Model adı
            metrics (Dict[str, Any]): Hesaplanan metrikler
        """
        # Sonuçları kaydet
        results_path = os.path.join(self.output_dir, f"{model_name}_evaluation_results.json")
        
        # JSON serializable hale getir
        serializable_metrics = self._make_serializable(metrics)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
        
        print(f"Değerlendirme sonuçları kaydedildi: {results_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Objeyi JSON serializable hale getirir
        
        Args:
            obj (Any): Dönüştürülecek obje
            
        Returns:
            Any: JSON serializable obje
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def get_evaluation_summary(self, 
                              metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Değerlendirme özetini döndürür
        
        Args:
            metrics (Dict[str, Any]): Hesaplanan metrikler
            
        Returns:
            Dict[str, Any]: Değerlendirme özeti
        """
        summary = {}
        
        # Temel metrikler
        if 'basic_metrics' in metrics:
            summary.update(metrics['basic_metrics'])
        
        # Top-K doğruluk
        if 'top_k_accuracy' in metrics:
            summary.update(metrics['top_k_accuracy'])
        
        # ROC AUC
        if 'roc_metrics' in metrics:
            roc = metrics['roc_metrics']
            summary['roc_auc_macro'] = roc.get('roc_auc_macro', 0)
            summary['roc_auc_weighted'] = roc.get('roc_auc_weighted', 0)
        
        # Model karmaşıklığı
        if 'model_complexity' in metrics:
            complexity = metrics['model_complexity']
            summary['total_parameters'] = complexity.get('total_parameters', 0)
            summary['model_size_mb'] = complexity.get('model_size_mb', 0)
        
        return summary

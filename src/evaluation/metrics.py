"""
Metrik hesaplama modülü
Model performans metriklerini hesaplar
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import tensorflow as tf
from tensorflow import keras


class MetricsCalculator:
    """
    Metrik hesaplama sınıfı
    """
    
    def __init__(self, class_names: List[str]):
        """
        MetricsCalculator sınıfını başlatır
        
        Args:
            class_names (List[str]): Sınıf isimleri
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def calculate_basic_metrics(self, 
                               y_true: np.ndarray, 
                               y_pred: np.ndarray) -> Dict[str, float]:
        """
        Temel metrikleri hesaplar
        
        Args:
            y_true (np.ndarray): Gerçek etiketler
            y_pred (np.ndarray): Tahmin edilen etiketler
            
        Returns:
            Dict[str, float]: Temel metrikler
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def calculate_per_class_metrics(self, 
                                   y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> pd.DataFrame:
        """
        Sınıf bazında metrikleri hesaplar
        
        Args:
            y_true (np.ndarray): Gerçek etiketler
            y_pred (np.ndarray): Tahmin edilen etiketler
            
        Returns:
            pd.DataFrame: Sınıf bazında metrikler
        """
        # Classification report'u al
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names, 
            output_dict=True, 
            zero_division=0
        )
        
        # DataFrame oluştur
        per_class_metrics = []
        
        for class_name in self.class_names:
            if class_name in report:
                metrics = report[class_name]
                metrics['class_name'] = class_name
                per_class_metrics.append(metrics)
        
        df = pd.DataFrame(per_class_metrics)
        
        # Sütun sıralamasını düzenle
        columns = ['class_name', 'precision', 'recall', 'f1-score', 'support']
        df = df[columns]
        
        return df
    
    def calculate_confusion_matrix_metrics(self, 
                                         y_true: np.ndarray, 
                                         y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Karışıklık matrisi metriklerini hesaplar
        
        Args:
            y_true (np.ndarray): Gerçek etiketler
            y_pred (np.ndarray): Tahmin edilen etiketler
            
        Returns:
            Dict[str, Any]: Karışıklık matrisi metrikleri
        """
        # Karışıklık matrisini hesapla
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize edilmiş karışıklık matrisi
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Sınıf bazında doğruluk
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Genel doğruluk
        overall_accuracy = cm.diagonal().sum() / cm.sum()
        
        # Sınıf bazında hata oranları
        class_error_rate = 1 - class_accuracy
        
        # En çok karıştırılan sınıflar
        most_confused = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and cm[i, j] > 0:
                    most_confused.append({
                        'true_class': self.class_names[i],
                        'predicted_class': self.class_names[j],
                        'count': cm[i, j],
                        'percentage': cm[i, j] / cm[i, :].sum() * 100
                    })
        
        # En çok karıştırılan sınıfları sırala
        most_confused.sort(key=lambda x: x['count'], reverse=True)
        
        metrics = {
            'confusion_matrix': cm,
            'confusion_matrix_normalized': cm_normalized,
            'class_accuracy': class_accuracy,
            'overall_accuracy': overall_accuracy,
            'class_error_rate': class_error_rate,
            'most_confused_pairs': most_confused[:10]  # En çok karıştırılan 10 çift
        }
        
        return metrics
    
    def calculate_roc_metrics(self, 
                             y_true: np.ndarray, 
                             y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        ROC metriklerini hesaplar
        
        Args:
            y_true (np.ndarray): Gerçek etiketler
            y_pred_proba (np.ndarray): Tahmin olasılıkları
            
        Returns:
            Dict[str, Any]: ROC metrikleri
        """
        # One-hot encoding'i binary'ye çevir
        y_true_binary = tf.keras.utils.to_categorical(y_true, num_classes=self.num_classes)
        
        # Macro average ROC AUC
        roc_auc_macro = roc_auc_score(y_true_binary, y_pred_proba, average='macro', multi_class='ovr')
        
        # Weighted average ROC AUC
        roc_auc_weighted = roc_auc_score(y_true_binary, y_pred_proba, average='weighted', multi_class='ovr')
        
        # Sınıf bazında ROC AUC
        roc_auc_per_class = []
        for i in range(self.num_classes):
            try:
                auc = roc_auc_score(y_true_binary[:, i], y_pred_proba[:, i])
                roc_auc_per_class.append(auc)
            except ValueError:
                roc_auc_per_class.append(0.0)
        
        # ROC eğrileri
        roc_curves = {}
        for i in range(self.num_classes):
            try:
                fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_pred_proba[:, i])
                roc_curves[self.class_names[i]] = {'fpr': fpr, 'tpr': tpr}
            except ValueError:
                roc_curves[self.class_names[i]] = {'fpr': [0, 1], 'tpr': [0, 1]}
        
        # Precision-Recall eğrileri
        pr_curves = {}
        pr_auc_per_class = []
        for i in range(self.num_classes):
            try:
                precision, recall, _ = precision_recall_curve(y_true_binary[:, i], y_pred_proba[:, i])
                pr_auc = average_precision_score(y_true_binary[:, i], y_pred_proba[:, i])
                pr_curves[self.class_names[i]] = {'precision': precision, 'recall': recall}
                pr_auc_per_class.append(pr_auc)
            except ValueError:
                pr_curves[self.class_names[i]] = {'precision': [1, 0], 'recall': [0, 1]}
                pr_auc_per_class.append(0.0)
        
        # Macro average PR AUC
        pr_auc_macro = np.mean(pr_auc_per_class)
        
        metrics = {
            'roc_auc_macro': roc_auc_macro,
            'roc_auc_weighted': roc_auc_weighted,
            'roc_auc_per_class': roc_auc_per_class,
            'roc_curves': roc_curves,
            'pr_auc_macro': pr_auc_macro,
            'pr_auc_per_class': pr_auc_per_class,
            'pr_curves': pr_curves
        }
        
        return metrics
    
    def calculate_top_k_accuracy(self, 
                                y_true: np.ndarray, 
                                y_pred_proba: np.ndarray, 
                                k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """
        Top-K doğruluğunu hesaplar
        
        Args:
            y_true (np.ndarray): Gerçek etiketler
            y_pred_proba (np.ndarray): Tahmin olasılıkları
            k_values (List[int]): K değerleri
            
        Returns:
            Dict[str, float]: Top-K doğrulukları
        """
        top_k_metrics = {}
        
        for k in k_values:
            if k <= self.num_classes:
                # Top-K tahminleri
                top_k_pred = np.argsort(y_pred_proba, axis=1)[:, -k:]
                
                # Top-K doğruluğu
                correct = 0
                for i in range(len(y_true)):
                    if y_true[i] in top_k_pred[i]:
                        correct += 1
                
                top_k_metrics[f'top_{k}_accuracy'] = correct / len(y_true)
        
        return top_k_metrics
    
    def calculate_model_complexity_metrics(self, 
                                          model: keras.Model) -> Dict[str, Any]:
        """
        Model karmaşıklık metriklerini hesaplar
        
        Args:
            model (keras.Model): Model
            
        Returns:
            Dict[str, Any]: Model karmaşıklık metrikleri
        """
        # Toplam parametre sayısı
        total_params = model.count_params()
        
        # Eğitilebilir parametre sayısı
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        
        # Dondurulmuş parametre sayısı
        non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        
        # Katman sayısı
        num_layers = len(model.layers)
        
        # Eğitilebilir katman sayısı
        trainable_layers = len([layer for layer in model.layers if layer.trainable])
        
        # Model boyutu (MB)
        model_size_mb = self._estimate_model_size(model)
        
        metrics = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params,
            'total_layers': num_layers,
            'trainable_layers': trainable_layers,
            'model_size_mb': model_size_mb,
            'parameters_per_layer': total_params / num_layers if num_layers > 0 else 0
        }
        
        return metrics
    
    def _estimate_model_size(self, model: keras.Model) -> float:
        """
        Model boyutunu tahmin eder (MB)
        
        Args:
            model (keras.Model): Model
            
        Returns:
            float: Model boyutu (MB)
        """
        # Basit tahmin: parametre sayısı * 4 byte (float32)
        total_params = model.count_params()
        size_bytes = total_params * 4
        size_mb = size_bytes / (1024 * 1024)
        
        return size_mb
    
    def calculate_all_metrics(self, 
                             y_true: np.ndarray, 
                             y_pred: np.ndarray, 
                             y_pred_proba: np.ndarray,
                             model: Optional[keras.Model] = None) -> Dict[str, Any]:
        """
        Tüm metrikleri hesaplar
        
        Args:
            y_true (np.ndarray): Gerçek etiketler
            y_pred: np.ndarray: Tahmin edilen etiketler
            y_pred_proba (np.ndarray): Tahmin olasılıkları
            model (Optional[keras.Model]): Model (karmaşıklık metrikleri için)
            
        Returns:
            Dict[str, Any]: Tüm metrikler
        """
        print("Metrikler hesaplanıyor...")
        
        all_metrics = {}
        
        # Temel metrikler
        all_metrics['basic_metrics'] = self.calculate_basic_metrics(y_true, y_pred)
        
        # Sınıf bazında metrikler
        all_metrics['per_class_metrics'] = self.calculate_per_class_metrics(y_true, y_pred)
        
        # Karışıklık matrisi metrikleri
        all_metrics['confusion_matrix_metrics'] = self.calculate_confusion_matrix_metrics(y_true, y_pred)
        
        # ROC metrikleri
        all_metrics['roc_metrics'] = self.calculate_roc_metrics(y_true, y_pred_proba)
        
        # Top-K doğruluk
        all_metrics['top_k_accuracy'] = self.calculate_top_k_accuracy(y_true, y_pred_proba)
        
        # Model karmaşıklık metrikleri
        if model is not None:
            all_metrics['model_complexity'] = self.calculate_model_complexity_metrics(model)
        
        print("Metrik hesaplama tamamlandı!")
        
        return all_metrics
    
    def print_metrics_summary(self, metrics: Dict[str, Any]) -> None:
        """
        Metrik özetini yazdırır
        
        Args:
            metrics (Dict[str, Any]): Hesaplanan metrikler
        """
        print("\nMODEL PERFORMANS ÖZETİ")
        print("=" * 50)
        
        # Temel metrikler
        basic = metrics.get('basic_metrics', {})
        print(f"\nTemel Metrikler:")
        print(f"  Doğruluk: {basic.get('accuracy', 0):.4f}")
        print(f"  Precision (Macro): {basic.get('precision_macro', 0):.4f}")
        print(f"  Recall (Macro): {basic.get('recall_macro', 0):.4f}")
        print(f"  F1-Score (Macro): {basic.get('f1_macro', 0):.4f}")
        
        # Top-K doğruluk
        top_k = metrics.get('top_k_accuracy', {})
        if top_k:
            print(f"\nTop-K Doğruluk:")
            for k, acc in top_k.items():
                print(f"  {k}: {acc:.4f}")
        
        # ROC AUC
        roc = metrics.get('roc_metrics', {})
        if roc:
            print(f"\nROC AUC:")
            print(f"  Macro: {roc.get('roc_auc_macro', 0):.4f}")
            print(f"  Weighted: {roc.get('roc_auc_weighted', 0):.4f}")
        
        # Model karmaşıklığı
        complexity = metrics.get('model_complexity', {})
        if complexity:
            print(f"\nModel Karmaşıklığı:")
            print(f"  Toplam Parametre: {complexity.get('total_parameters', 0):,}")
            print(f"  Eğitilebilir Parametre: {complexity.get('trainable_parameters', 0):,}")
            print(f"  Model Boyutu: {complexity.get('model_size_mb', 0):.2f} MB")

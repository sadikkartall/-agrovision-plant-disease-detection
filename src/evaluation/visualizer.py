"""
Görselleştirme modülü
Model performansını görselleştirir
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import os
from sklearn.metrics import roc_curve, precision_recall_curve


class ResultsVisualizer:
    """
    Sonuç görselleştirme sınıfı
    """
    
    def __init__(self, class_names: List[str], figsize: Tuple[int, int] = (12, 8)):
        """
        ResultsVisualizer sınıfını başlatır
        
        Args:
            class_names (List[str]): Sınıf isimleri
            figsize (Tuple[int, int]): Grafik boyutu
        """
        self.class_names = class_names
        self.figsize = figsize
        
        # Matplotlib stilini ayarla
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self, 
                             confusion_matrix: np.ndarray,
                             title: str = "Karışıklık Matrisi",
                             save_path: Optional[str] = None,
                             show_plot: bool = True) -> None:
        """
        Karışıklık matrisini çizer
        
        Args:
            confusion_matrix (np.ndarray): Karışıklık matrisi
            title (str): Grafik başlığı
            save_path (Optional[str]): Kayıt yolu
            show_plot (bool): Grafiği göster
        """
        plt.figure(figsize=self.figsize)
        
        # Karışıklık matrisini çiz
        sns.heatmap(confusion_matrix, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Tahmin Edilen Sınıf', fontsize=12)
        plt.ylabel('Gerçek Sınıf', fontsize=12)
        plt.xticks(rotation=45, ha='right')
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
    
    def plot_normalized_confusion_matrix(self, 
                                       confusion_matrix: np.ndarray,
                                       title: str = "Normalize Edilmiş Karışıklık Matrisi",
                                       save_path: Optional[str] = None,
                                       show_plot: bool = True) -> None:
        """
        Normalize edilmiş karışıklık matrisini çizer
        
        Args:
            confusion_matrix (np.ndarray): Karışıklık matrisi
            title (str): Grafik başlığı
            save_path (Optional[str]): Kayıt yolu
            show_plot (bool): Grafiği göster
        """
        # Normalize et
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=self.figsize)
        
        # Normalize edilmiş karışıklık matrisini çiz
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Tahmin Edilen Sınıf', fontsize=12)
        plt.ylabel('Gerçek Sınıf', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Kaydet
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Normalize edilmiş karışıklık matrisi kaydedildi: {save_path}")
        
        # Göster
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_roc_curves(self, 
                       roc_curves: Dict[str, Dict[str, np.ndarray]],
                       title: str = "ROC Eğrileri",
                       save_path: Optional[str] = None,
                       show_plot: bool = True) -> None:
        """
        ROC eğrilerini çizer
        
        Args:
            roc_curves (Dict[str, Dict[str, np.ndarray]]): ROC eğrileri
            title (str): Grafik başlığı
            save_path (Optional[str]): Kayıt yolu
            show_plot (bool): Grafiği göster
        """
        plt.figure(figsize=self.figsize)
        
        # Her sınıf için ROC eğrisi çiz
        for class_name, curve_data in roc_curves.items():
            fpr = curve_data['fpr']
            tpr = curve_data['tpr']
            plt.plot(fpr, tpr, label=f'{class_name}', linewidth=2)
        
        # Diagonal çizgi (rastgele sınıflandırıcı)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Rastgele Sınıflandırıcı')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Yanlış Pozitif Oranı (FPR)', fontsize=12)
        plt.ylabel('Doğru Pozitif Oranı (TPR)', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Kaydet
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC eğrileri kaydedildi: {save_path}")
        
        # Göster
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_precision_recall_curves(self, 
                                    pr_curves: Dict[str, Dict[str, np.ndarray]],
                                    title: str = "Precision-Recall Eğrileri",
                                    save_path: Optional[str] = None,
                                    show_plot: bool = True) -> None:
        """
        Precision-Recall eğrilerini çizer
        
        Args:
            pr_curves (Dict[str, Dict[str, np.ndarray]]): PR eğrileri
            title (str): Grafik başlığı
            save_path (Optional[str]): Kayıt yolu
            show_plot (bool): Grafiği göster
        """
        plt.figure(figsize=self.figsize)
        
        # Her sınıf için PR eğrisi çiz
        for class_name, curve_data in pr_curves.items():
            precision = curve_data['precision']
            recall = curve_data['recall']
            plt.plot(recall, precision, label=f'{class_name}', linewidth=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Kaydet
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall eğrileri kaydedildi: {save_path}")
        
        # Göster
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_per_class_metrics(self, 
                              per_class_metrics: pd.DataFrame,
                              title: str = "Sınıf Bazında Metrikler",
                              save_path: Optional[str] = None,
                              show_plot: bool = True) -> None:
        """
        Sınıf bazında metrikleri çizer
        
        Args:
            per_class_metrics (pd.DataFrame): Sınıf bazında metrikler
            title (str): Grafik başlığı
            save_path (Optional[str]): Kayıt yolu
            show_plot (bool): Grafiği göster
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Precision
        axes[0, 0].bar(per_class_metrics['class_name'], per_class_metrics['precision'])
        axes[0, 0].set_title('Precision', fontweight='bold')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Recall
        axes[0, 1].bar(per_class_metrics['class_name'], per_class_metrics['recall'])
        axes[0, 1].set_title('Recall', fontweight='bold')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1-Score
        axes[1, 0].bar(per_class_metrics['class_name'], per_class_metrics['f1-score'])
        axes[1, 0].set_title('F1-Score', fontweight='bold')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Support
        axes[1, 1].bar(per_class_metrics['class_name'], per_class_metrics['support'])
        axes[1, 1].set_title('Support', fontweight='bold')
        axes[1, 1].set_ylabel('Örnek Sayısı')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Kaydet
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sınıf bazında metrikler kaydedildi: {save_path}")
        
        # Göster
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_model_comparison(self, 
                             comparison_data: pd.DataFrame,
                             metric: str = 'Test Accuracy',
                             title: str = "Model Karşılaştırması",
                             save_path: Optional[str] = None,
                             show_plot: bool = True) -> None:
        """
        Model karşılaştırmasını çizer
        
        Args:
            comparison_data (pd.DataFrame): Karşılaştırma verisi
            metric (str): Karşılaştırılacak metrik
            title (str): Grafik başlığı
            save_path (Optional[str]): Kayıt yolu
            show_plot (bool): Grafiği göster
        """
        plt.figure(figsize=(12, 6))
        
        # Bar plot
        bars = plt.bar(comparison_data['Model'], comparison_data[metric])
        
        # En iyi modeli vurgula
        best_idx = comparison_data[metric].idxmax()
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Değerleri bar üzerine yaz
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Kaydet
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model karşılaştırması kaydedildi: {save_path}")
        
        # Göster
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_training_history(self, 
                             history: Dict[str, List[float]],
                             title: str = "Eğitim Geçmişi",
                             save_path: Optional[str] = None,
                             show_plot: bool = True) -> None:
        """
        Eğitim geçmişini çizer
        
        Args:
            history (Dict[str, List[float]]): Eğitim geçmişi
            title (str): Grafik başlığı
            save_path (Optional[str]): Kayıt yolu
            show_plot (bool): Grafiği göster
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Accuracy
        if 'accuracy' in history and 'val_accuracy' in history:
            axes[0].plot(history['accuracy'], label='Eğitim', linewidth=2)
            axes[0].plot(history['val_accuracy'], label='Doğrulama', linewidth=2)
            axes[0].set_title('Doğruluk', fontweight='bold')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Doğruluk')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Loss
        if 'loss' in history and 'val_loss' in history:
            axes[1].plot(history['loss'], label='Eğitim', linewidth=2)
            axes[1].plot(history['val_loss'], label='Doğrulama', linewidth=2)
            axes[1].set_title('Kayıp', fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Kayıp')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Kaydet
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Eğitim geçmişi kaydedildi: {save_path}")
        
        # Göster
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_error_analysis(self, 
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           y_pred_proba: np.ndarray,
                           class_names: List[str],
                           num_samples: int = 10,
                           title: str = "Hata Analizi",
                           save_path: Optional[str] = None,
                           show_plot: bool = True) -> None:
        """
        Hata analizini çizer
        
        Args:
            y_true (np.ndarray): Gerçek etiketler
            y_pred (np.ndarray): Tahmin edilen etiketler
            y_pred_proba (np.ndarray): Tahmin olasılıkları
            class_names (List[str]): Sınıf isimleri
            num_samples (int): Gösterilecek örnek sayısı
            title (str): Grafik başlığı
            save_path (Optional[str]): Kayıt yolu
            show_plot (bool): Grafiği göster
        """
        # Hatalı tahminleri bul
        wrong_predictions = y_true != y_pred
        wrong_indices = np.where(wrong_predictions)[0]
        
        if len(wrong_indices) == 0:
            print("Hatalı tahmin bulunamadı!")
            return
        
        # En yüksek olasılıklı hataları seç
        wrong_probs = y_pred_proba[wrong_indices]
        max_probs = np.max(wrong_probs, axis=1)
        top_wrong_indices = wrong_indices[np.argsort(max_probs)[-num_samples:]]
        
        # Grafik oluştur
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i, idx in enumerate(top_wrong_indices[:10]):
            row = i // 5
            col = i % 5
            
            true_class = class_names[y_true[idx]]
            pred_class = class_names[y_pred[idx]]
            confidence = y_pred_proba[idx][y_pred[idx]]
            
            axes[row, col].text(0.5, 0.5, 
                               f'Gerçek: {true_class}\nTahmin: {pred_class}\nGüven: {confidence:.3f}',
                               ha='center', va='center', fontsize=10,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[row, col].set_title(f'Örnek {i+1}', fontweight='bold')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Kaydet
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Hata analizi kaydedildi: {save_path}")
        
        # Göster
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def create_comprehensive_report(self, 
                                  metrics: Dict[str, Any],
                                  save_dir: str) -> None:
        """
        Kapsamlı rapor oluşturur
        
        Args:
            metrics (Dict[str, Any]): Hesaplanan metrikler
            save_dir (str): Kayıt dizini
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Karışıklık matrisi
        if 'confusion_matrix_metrics' in metrics:
            cm = metrics['confusion_matrix_metrics']['confusion_matrix']
            self.plot_confusion_matrix(
                cm, 
                save_path=os.path.join(save_dir, 'confusion_matrix.png'),
                show_plot=False
            )
            
            self.plot_normalized_confusion_matrix(
                cm,
                save_path=os.path.join(save_dir, 'confusion_matrix_normalized.png'),
                show_plot=False
            )
        
        # ROC eğrileri
        if 'roc_metrics' in metrics and 'roc_curves' in metrics['roc_metrics']:
            self.plot_roc_curves(
                metrics['roc_metrics']['roc_curves'],
                save_path=os.path.join(save_dir, 'roc_curves.png'),
                show_plot=False
            )
        
        # Precision-Recall eğrileri
        if 'roc_metrics' in metrics and 'pr_curves' in metrics['roc_metrics']:
            self.plot_precision_recall_curves(
                metrics['roc_metrics']['pr_curves'],
                save_path=os.path.join(save_dir, 'precision_recall_curves.png'),
                show_plot=False
            )
        
        # Sınıf bazında metrikler
        if 'per_class_metrics' in metrics:
            self.plot_per_class_metrics(
                metrics['per_class_metrics'],
                save_path=os.path.join(save_dir, 'per_class_metrics.png'),
                show_plot=False
            )
        
        print(f"Kapsamlı rapor oluşturuldu: {save_dir}")

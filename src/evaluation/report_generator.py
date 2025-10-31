"""
Rapor oluşturma modülü
Model performans raporları oluşturur
"""

import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class ReportGenerator:
    """
    Rapor oluşturma sınıfı
    """
    
    def __init__(self, 
                 class_names: List[str],
                 output_dir: str = "reports"):
        """
        ReportGenerator sınıfını başlatır
        
        Args:
            class_names (List[str]): Sınıf isimleri
            output_dir (str): Çıktı dizini
        """
        self.class_names = class_names
        self.output_dir = output_dir
        self.num_classes = len(class_names)
        
        # Çıktı dizinini oluştur
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_model_report(self, 
                             model_name: str,
                             metrics: Dict[str, Any],
                             model: Optional[Any] = None) -> None:
        """
        Model raporu oluşturur
        
        Args:
            model_name (str): Model adı
            metrics (Dict[str, Any]): Hesaplanan metrikler
            model (Optional[Any]): Model (opsiyonel)
        """
        print(f"{model_name} raporu oluşturuluyor...")
        
        # Rapor dosyası
        report_path = os.path.join(self.output_dir, f"{model_name}_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("BİTKİ HASTALIK TESPİT SİSTEMİ - MODEL RAPORU\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Model Adı: {model_name}\n")
            f.write(f"Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Toplam Sınıf Sayısı: {self.num_classes}\n\n")
            
            # Temel metrikler
            if 'basic_metrics' in metrics:
                basic = metrics['basic_metrics']
                f.write("TEMEL METRİKLER\n")
                f.write("-" * 20 + "\n")
                f.write(f"Doğruluk (Accuracy): {basic.get('accuracy', 0):.4f}\n")
                f.write(f"Precision (Macro): {basic.get('precision_macro', 0):.4f}\n")
                f.write(f"Recall (Macro): {basic.get('recall_macro', 0):.4f}\n")
                f.write(f"F1-Score (Macro): {basic.get('f1_macro', 0):.4f}\n")
                f.write(f"Precision (Weighted): {basic.get('precision_weighted', 0):.4f}\n")
                f.write(f"Recall (Weighted): {basic.get('recall_weighted', 0):.4f}\n")
                f.write(f"F1-Score (Weighted): {basic.get('f1_weighted', 0):.4f}\n\n")
            
            # Top-K doğruluk
            if 'top_k_accuracy' in metrics:
                top_k = metrics['top_k_accuracy']
                f.write("TOP-K DOĞRULUK\n")
                f.write("-" * 20 + "\n")
                for k, acc in top_k.items():
                    f.write(f"{k}: {acc:.4f}\n")
                f.write("\n")
            
            # ROC AUC
            if 'roc_metrics' in metrics:
                roc = metrics['roc_metrics']
                f.write("ROC AUC METRİKLERİ\n")
                f.write("-" * 20 + "\n")
                f.write(f"ROC AUC (Macro): {roc.get('roc_auc_macro', 0):.4f}\n")
                f.write(f"ROC AUC (Weighted): {roc.get('roc_auc_weighted', 0):.4f}\n")
                f.write(f"PR AUC (Macro): {roc.get('pr_auc_macro', 0):.4f}\n\n")
            
            # Sınıf bazında metrikler
            if 'per_class_metrics' in metrics:
                per_class = metrics['per_class_metrics']
                f.write("SINIF BAZINDA METRİKLER\n")
                f.write("-" * 30 + "\n")
                f.write(per_class.to_string(index=False))
                f.write("\n\n")
            
            # Model karmaşıklığı
            if 'model_complexity' in metrics:
                complexity = metrics['model_complexity']
                f.write("MODEL KARMAŞIKLIĞI\n")
                f.write("-" * 20 + "\n")
                f.write(f"Toplam Parametre: {complexity.get('total_parameters', 0):,}\n")
                f.write(f"Eğitilebilir Parametre: {complexity.get('trainable_parameters', 0):,}\n")
                f.write(f"Dondurulmuş Parametre: {complexity.get('non_trainable_parameters', 0):,}\n")
                f.write(f"Toplam Katman: {complexity.get('total_layers', 0)}\n")
                f.write(f"Eğitilebilir Katman: {complexity.get('trainable_layers', 0)}\n")
                f.write(f"Model Boyutu: {complexity.get('model_size_mb', 0):.2f} MB\n")
                f.write(f"Katman Başına Parametre: {complexity.get('parameters_per_layer', 0):.0f}\n\n")
            
            # En çok karıştırılan sınıflar
            if 'confusion_matrix_metrics' in metrics:
                cm_metrics = metrics['confusion_matrix_metrics']
                if 'most_confused_pairs' in cm_metrics:
                    f.write("EN ÇOK KARIŞTIRILAN SINIF ÇİFTLERİ\n")
                    f.write("-" * 40 + "\n")
                    for i, (pair, count) in enumerate(cm_metrics['most_confused_pairs'][:10]):
                        f.write(f"{i+1:2d}. {pair}: {count} kez\n")
                    f.write("\n")
            
            # Öneriler
            f.write("ÖNERİLER\n")
            f.write("-" * 20 + "\n")
            self._write_recommendations(f, metrics)
        
        print(f"Model raporu kaydedildi: {report_path}")
    
    def generate_comparison_report(self, 
                                 comparison_df: pd.DataFrame) -> None:
        """
        Model karşılaştırma raporu oluşturur
        
        Args:
            comparison_df (pd.DataFrame): Karşılaştırma verisi
        """
        print("Model karşılaştırma raporu oluşturuluyor...")
        
        # Rapor dosyası
        report_path = os.path.join(self.output_dir, "model_comparison_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("BİTKİ HASTALIK TESPİT SİSTEMİ - MODEL KARŞILAŞTIRMA RAPORU\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Karşılaştırılan Model Sayısı: {len(comparison_df)}\n\n")
            
            # Genel karşılaştırma
            f.write("GENEL KARŞILAŞTIRMA\n")
            f.write("-" * 30 + "\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
            
            # En iyi modeller
            f.write("EN İYİ MODELLER\n")
            f.write("-" * 20 + "\n")
            
            # En yüksek doğruluk
            best_accuracy = comparison_df.loc[comparison_df['Test Accuracy'].idxmax()]
            f.write(f"En Yüksek Doğruluk: {best_accuracy['Model']} ({best_accuracy['Test Accuracy']:.4f})\n")
            
            # En yüksek F1-Score
            best_f1 = comparison_df.loc[comparison_df['Test F1-Score (Macro)'].idxmax()]
            f.write(f"En Yüksek F1-Score: {best_f1['Model']} ({best_f1['Test F1-Score (Macro)']:.4f})\n")
            
            # En az parametre (yüksek doğruluklu modeller arasında)
            high_acc_models = comparison_df[comparison_df['Test Accuracy'] >= best_accuracy['Test Accuracy'] - 0.02]
            if len(high_acc_models) > 1:
                most_efficient = high_acc_models.loc[high_acc_models['Total Parameters'].idxmin()]
                f.write(f"En Verimli Model: {most_efficient['Model']} ({most_efficient['Total Parameters']:,} parametre)\n")
            
            f.write("\n")
            
            # Detaylı analiz
            f.write("DETAYLI ANALİZ\n")
            f.write("-" * 20 + "\n")
            
            # Doğruluk dağılımı
            accuracy_stats = comparison_df['Test Accuracy'].describe()
            f.write(f"Doğruluk İstatistikleri:\n")
            f.write(f"  Ortalama: {accuracy_stats['mean']:.4f}\n")
            f.write(f"  Standart Sapma: {accuracy_stats['std']:.4f}\n")
            f.write(f"  Minimum: {accuracy_stats['min']:.4f}\n")
            f.write(f"  Maksimum: {accuracy_stats['max']:.4f}\n\n")
            
            # Parametre dağılımı
            param_stats = comparison_df['Total Parameters'].describe()
            f.write(f"Parametre İstatistikleri:\n")
            f.write(f"  Ortalama: {param_stats['mean']:,.0f}\n")
            f.write(f"  Standart Sapma: {param_stats['std']:,.0f}\n")
            f.write(f"  Minimum: {param_stats['min']:,.0f}\n")
            f.write(f"  Maksimum: {param_stats['max']:,.0f}\n\n")
            
            # Öneriler
            f.write("ÖNERİLER\n")
            f.write("-" * 20 + "\n")
            self._write_comparison_recommendations(f, comparison_df)
        
        print(f"Karşılaştırma raporu kaydedildi: {report_path}")
    
    def generate_summary_report(self, 
                               all_results: Dict[str, Dict[str, Any]]) -> None:
        """
        Özet rapor oluşturur
        
        Args:
            all_results (Dict[str, Dict[str, Any]]): Tüm sonuçlar
        """
        print("Özet rapor oluşturuluyor...")
        
        # Rapor dosyası
        report_path = os.path.join(self.output_dir, "summary_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("BİTKİ HASTALIK TESPİT SİSTEMİ - ÖZET RAPOR\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Değerlendirilen Model Sayısı: {len(all_results)}\n\n")
            
            # Genel özet
            f.write("GENEL ÖZET\n")
            f.write("-" * 20 + "\n")
            
            accuracies = []
            for model_name, results in all_results.items():
                if 'basic_metrics' in results:
                    acc = results['basic_metrics'].get('accuracy', 0)
                    accuracies.append(acc)
                    f.write(f"{model_name}: {acc:.4f}\n")
            
            if accuracies:
                f.write(f"\nOrtalama Doğruluk: {np.mean(accuracies):.4f}\n")
                f.write(f"En Yüksek Doğruluk: {np.max(accuracies):.4f}\n")
                f.write(f"En Düşük Doğruluk: {np.min(accuracies):.4f}\n")
            
            f.write("\n")
            
            # Sınıf analizi
            f.write("SINIF ANALİZİ\n")
            f.write("-" * 20 + "\n")
            f.write(f"Toplam Sınıf Sayısı: {self.num_classes}\n")
            f.write("Sınıflar:\n")
            for i, class_name in enumerate(self.class_names):
                f.write(f"  {i+1:2d}. {class_name}\n")
            
            f.write("\n")
            
            # Öneriler
            f.write("GENEL ÖNERİLER\n")
            f.write("-" * 20 + "\n")
            f.write("1. En yüksek performans gösteren modeli üretim ortamında kullanın\n")
            f.write("2. Daha fazla veri toplayarak model performansını artırabilirsiniz\n")
            f.write("3. Veri artırma tekniklerini optimize edebilirsiniz\n")
            f.write("4. Model mimarisini daha da geliştirebilirsiniz\n")
            f.write("5. Hyperparameter tuning yapabilirsiniz\n")
            f.write("6. Ensemble yöntemleri deneyebilirsiniz\n")
        
        print(f"Özet rapor kaydedildi: {report_path}")
    
    def _write_recommendations(self, f, metrics: Dict[str, Any]) -> None:
        """
        Önerileri yazar
        
        Args:
            f: Dosya objesi
            metrics (Dict[str, Any]): Metrikler
        """
        # Temel metrikler
        if 'basic_metrics' in metrics:
            basic = metrics['basic_metrics']
            accuracy = basic.get('accuracy', 0)
            precision = basic.get('precision_macro', 0)
            recall = basic.get('recall_macro', 0)
            f1 = basic.get('f1_macro', 0)
            
            if accuracy < 0.8:
                f.write("• Model doğruluğu düşük. Daha fazla veri veya daha iyi özellikler gerekebilir.\n")
            
            if precision < recall:
                f.write("• Precision recall'dan düşük. False positive'leri azaltmak için threshold ayarlayın.\n")
            elif recall < precision:
                f.write("• Recall precision'dan düşük. False negative'leri azaltmak için threshold ayarlayın.\n")
            
            if f1 < 0.7:
                f.write("• F1-Score düşük. Model performansını artırmak için veri kalitesini kontrol edin.\n")
        
        # ROC AUC
        if 'roc_metrics' in metrics:
            roc_auc = metrics['roc_metrics'].get('roc_auc_macro', 0)
            if roc_auc < 0.8:
                f.write("• ROC AUC düşük. Model sınıflandırma yeteneğini artırmak gerekebilir.\n")
        
        # Model karmaşıklığı
        if 'model_complexity' in metrics:
            complexity = metrics['model_complexity']
            total_params = complexity.get('total_parameters', 0)
            model_size = complexity.get('model_size_mb', 0)
            
            if total_params > 10_000_000:
                f.write("• Model çok büyük. Daha küçük bir mimari veya pruning düşünün.\n")
            
            if model_size > 100:
                f.write("• Model boyutu büyük. Mobil deployment için optimize edin.\n")
        
        f.write("• Daha fazla veri toplayarak model performansını artırabilirsiniz.\n")
        f.write("• Veri artırma tekniklerini daha etkili kullanabilirsiniz.\n")
        f.write("• Transfer learning ile daha iyi sonuçlar alabilirsiniz.\n")
        f.write("• Ensemble yöntemleri deneyebilirsiniz.\n")
    
    def _write_comparison_recommendations(self, f, comparison_df: pd.DataFrame) -> None:
        """
        Karşılaştırma önerilerini yazar
        
        Args:
            f: Dosya objesi
            comparison_df (pd.DataFrame): Karşılaştırma verisi
        """
        best_model = comparison_df.loc[comparison_df['Test Accuracy'].idxmax()]
        
        f.write(f"• En iyi model: {best_model['Model']} (Doğruluk: {best_model['Test Accuracy']:.4f})\n")
        
        # Verimlilik analizi
        high_acc_models = comparison_df[comparison_df['Test Accuracy'] >= best_model['Test Accuracy'] - 0.02]
        if len(high_acc_models) > 1:
            most_efficient = high_acc_models.loc[high_acc_models['Total Parameters'].idxmin()]
            f.write(f"• En verimli model: {most_efficient['Model']} (En az parametre ile yüksek doğruluk)\n")
        
        # Performans dağılımı
        accuracy_std = comparison_df['Test Accuracy'].std()
        if accuracy_std < 0.05:
            f.write("• Modeller arasında performans farkı az. Tüm modeller benzer kalitede.\n")
        else:
            f.write("• Modeller arasında performans farkı büyük. En iyi modeli seçin.\n")
        
        f.write("• Üretim ortamında en iyi performans gösteren modeli kullanın.\n")
        f.write("• Gereksinimlerinize göre doğruluk ve verimlilik arasında denge kurun.\n")
        f.write("• Mobil deployment için daha küçük modelleri tercih edin.\n")
    
    def create_html_report(self, 
                          model_name: str,
                          metrics: Dict[str, Any]) -> None:
        """
        HTML raporu oluşturur
        
        Args:
            model_name (str): Model adı
            metrics (Dict[str, Any]): Metrikler
        """
        html_path = os.path.join(self.output_dir, f"{model_name}_report.html")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{model_name} - Model Raporu</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #ecf0f1; padding: 10px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>BİTKİ HASTALIK TESPİT SİSTEMİ - MODEL RAPORU</h1>
            <h2>{model_name}</h2>
            <p>Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        # Temel metrikler
        if 'basic_metrics' in metrics:
            basic = metrics['basic_metrics']
            html_content += f"""
            <h2>Temel Metrikler</h2>
            <div class="metric">
                <strong>Doğruluk:</strong> {basic.get('accuracy', 0):.4f}<br>
                <strong>Precision (Macro):</strong> {basic.get('precision_macro', 0):.4f}<br>
                <strong>Recall (Macro):</strong> {basic.get('recall_macro', 0):.4f}<br>
                <strong>F1-Score (Macro):</strong> {basic.get('f1_macro', 0):.4f}
            </div>
            """
        
        # Sınıf bazında metrikler
        if 'per_class_metrics' in metrics:
            per_class = metrics['per_class_metrics']
            html_content += f"""
            <h2>Sınıf Bazında Metrikler</h2>
            {per_class.to_html(index=False)}
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML raporu kaydedildi: {html_path}")

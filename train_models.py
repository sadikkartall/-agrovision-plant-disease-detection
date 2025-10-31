"""
Model Eğitimi Betiği
Bitki hastalık tespiti için transfer learning model eğitimi
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

# Proje modüllerini import et
from src.preprocess import DataLoader, DataSplitter
from src.training import DataGenerator, ModelTrainer, TrainingConfig
from src.evaluation import ModelEvaluator


def main():
    """Ana eğitim fonksiyonu"""
    print("🌿 BİTKİ HASTALIK TESPİT SİSTEMİ - MODEL EĞİTİMİ")
    print("=" * 70)
    
    # Konfigürasyonu yükle
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Veri yolu
    data_path = "data/raw"  # Otomatik olarak ayarla
    
    if not os.path.exists(data_path):
        print(f"❌ Veri seti bulunamadı: {data_path}")
        return
    
    print(f"📁 Veri seti yolu: {data_path}")
    
    # Veri hazırlama
    print("\n1️⃣ VERİ HAZIRLAMA")
    print("-" * 30)
    
    try:
        data_loader = DataLoader(data_path)
        image_paths, labels, class_names = data_loader.load_dataset()
        
        # DataFrame oluştur
        df = data_loader.create_dataframe()
        
        # Sınıf dağılımını göster
        class_dist = data_loader.get_class_distribution()
        print(f"\n📊 Sınıf Dağılımı:")
        for class_name, count in class_dist.items():
            print(f"  {class_name}: {count} örnek")
        
        # Label encoding kontrolü ve düzeltme
        print(f"\n🔍 Label Encoding Kontrolü:")
        print(f"  Numeric label aralığı: {df['numeric_label'].min()} - {df['numeric_label'].max()}")
        print(f"  Toplam sınıf sayısı: {len(class_names)}")
        
        # Label'ların 0'dan başlayıp num_classes-1'e kadar gitmesi gerekiyor
        unique_labels = sorted(df['numeric_label'].unique())
        expected_labels = list(range(len(class_names)))
        
        if unique_labels != expected_labels:
            print(f"  ⚠️ Label encoding düzeltiliyor...")
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df['numeric_label'] = le.fit_transform(df['class_name'])
            print(f"  ✅ Label encoding düzeltildi!")
            print(f"  Yeni aralık: {df['numeric_label'].min()} - {df['numeric_label'].max()}")
        
        # Veri bölme
        data_splitter = DataSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        train_df, val_df, test_df = data_splitter.split_and_organize_dataset(
            df, data_path, "data/splits"
        )
        
        # Split sonrası label kontrolü
        print(f"\n🔍 Split Sonrası Label Kontrolü:")
        print(f"  Train labels: {train_df['numeric_label'].min()} - {train_df['numeric_label'].max()}")
        print(f"  Val labels: {val_df['numeric_label'].min()} - {val_df['numeric_label'].max()}")
        print(f"  Test labels: {test_df['numeric_label'].min()} - {test_df['numeric_label'].max()}")
        
        # Veri bilgilerini kaydet
        with open('data/splits/data_info.json', 'w', encoding='utf-8') as f:
            json.dump({
                'class_names': class_names,
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df)
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Veri hazırlama tamamlandı!")
        print(f"  Eğitim: {len(train_df)} örnek")
        print(f"  Doğrulama: {len(val_df)} örnek")
        print(f"  Test: {len(test_df)} örnek")
        
    except Exception as e:
        print(f"❌ Veri hazırlama hatası: {e}")
        return
    
    # Eğitim konfigürasyonu
    print("\n2️⃣ EĞİTİM KONFİGÜRASYONU")
    print("-" * 30)
    
    training_config = TrainingConfig(
        input_shape=tuple(config['input_shape']),
        num_classes=len(class_names),
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        patience=config['patience']
    )
    
    training_config.print_config()
    
    # Veri üretici
    data_generator = DataGenerator(
        image_size=training_config.input_shape[:2],
        batch_size=training_config.batch_size,
        num_classes=training_config.num_classes,
        augmentation=training_config.augmentation
    )
    
    # Dataset'leri oluştur
    train_dataset, val_dataset, test_dataset = data_generator.create_train_val_test_datasets(
        train_df, val_df, test_df
    )
    
    # Model eğitici
    trainer = ModelTrainer(training_config.to_dict())
    
    # Eğitilecek modeller (sadece stabil olanlar)
    models_to_train = ['mobilenet_v2', 'resnet50']  # Transfer learning modelleri
    print(f"\n🤖 Eğitilecek Modeller: {', '.join(models_to_train)}")
    
    # Modelleri eğit
    print("\n3️⃣ MODEL EĞİTİMİ")
    print("-" * 30)
    
    trained_models = {}
    
    for i, model_name in enumerate(models_to_train, 1):
        print(f"\n[{i}/{len(models_to_train)}] {model_name} modeli eğitiliyor...")
        
        try:
            # Transfer learning için optimize edilmiş learning rate
            # Feature extraction modunda daha düşük LR kullanılır
            if model_name in ['mobilenet_v2', 'resnet50', 'efficientnet_b0']:
                model_lr = 0.0001  # Feature extraction için düşük LR
                print(f"  ⚙️ Transfer learning modu - Learning rate: {model_lr}")
            else:
                model_lr = training_config.learning_rate
            
            # Model oluştur
            model = trainer.create_model(
                model_name,
                dropout_rate=training_config.dropout_rate,
                learning_rate=model_lr,
                freeze_base=True  # Base model dondur (feature extraction)
            )
            
            # Model bilgilerini yazdır
            print(f"  📊 Toplam parametre: {model.count_params():,}")
            
            # Eğitilebilir parametreleri göster
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            frozen_params = model.count_params() - trainable_params
            print(f"  📊 Eğitilebilir: {trainable_params:,}, Dondurulmuş: {frozen_params:,}")
            
            # Modeli eğit
            trained_model = trainer.train_model(
                model, train_dataset, val_dataset, model_name,
                epochs=training_config.epochs
            )
            
            trained_models[model_name] = trained_model
            print(f"  ✅ {model_name} eğitimi tamamlandı!")
            
        except Exception as e:
            print(f"  ❌ {model_name} eğitimi başarısız: {str(e)}")
            continue
    
    if not trained_models:
        print("❌ Hiçbir model başarıyla eğitilemedi!")
        return
    
    # Model değerlendirmesi
    print("\n4️⃣ MODEL DEĞERLENDİRME")
    print("-" * 30)
    
    evaluator = ModelEvaluator(class_names=class_names, output_dir="reports")
    
    # Modelleri değerlendir
    evaluation_results = {}
    
    for model_name, model in trained_models.items():
        print(f"\n🔍 {model_name} değerlendiriliyor...")
        
        try:
            metrics = evaluator.evaluate_model(
                model, test_dataset, model_name
            )
            evaluation_results[model_name] = metrics
            
            # Temel metrikleri yazdır
            basic_metrics = metrics.get('basic_metrics', {})
            print(f"  📈 Test Doğruluğu: {basic_metrics.get('accuracy', 0):.4f}")
            print(f"  📈 F1-Score: {basic_metrics.get('f1_macro', 0):.4f}")
            
        except Exception as e:
            print(f"  ❌ {model_name} değerlendirmesi başarısız: {str(e)}")
            continue
    
    # Model karşılaştırması
    if len(evaluation_results) > 1:
        print("\n5️⃣ MODEL KARŞILAŞTIRMASI")
        print("-" * 30)
        
        comparison_df = evaluator.compare_models(trained_models, test_dataset)
        
        # En iyi modeli belirle
        best_model_name = comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Model']
        best_model = trained_models[best_model_name]
        
        print(f"\n🏆 En iyi model: {best_model_name}")
        print(f"📊 Test doğruluğu: {comparison_df['Test Accuracy'].max():.4f}")
        
        # En iyi modeli kaydet
        best_model_path = f"models/saved/best_model_{best_model_name}.h5"
        best_model.save(best_model_path)
        print(f"💾 En iyi model kaydedildi: {best_model_path}")
        
        # Karşılaştırma sonuçlarını kaydet
        comparison_df.to_csv('reports/model_comparison.csv', index=False)
        print(f"📊 Karşılaştırma sonuçları kaydedildi: reports/model_comparison.csv")
    
    print("\n✅ Model eğitimi tamamlandı!")
    print(f"📁 Sonuçlar: reports/ klasöründe")
    print(f"💾 Modeller: models/saved/ klasöründe")


if __name__ == "__main__":
    main()

# 🌿 Bitki Hastalık Tespit Sistemi

Bu proje, **Sayısal Görüntü İşleme** dersi kapsamında geliştirilmiş bir yapay zeka uygulamasıdır. Bitki yapraklarının fotoğraflarını analiz ederek hastalık tespiti yapar ve tarımda erken teşhis imkanı sağlar.

## 📋 İçindekiler

- [Proje Amacı](#proje-amacı)
- [Özellikler](#özellikler)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Eğitim Süreci](#eğitim-süreci)
- [Sonuçlar ve Raporlar](#sonuçlar-ve-raporlar)
- [Teknolojiler](#teknolojiler)

## 🎯 Proje Amacı

Tarımsal üretimde hastalık tespiti, ürün kalitesi ve verimliliği doğrudan etkileyen kritik bir faktördür. Bu proje:

- **Erken Teşhis**: Yapraklardaki hastalıkları henüz görsel olarak belirgin hale gelmeden tespit eder
- **Doğru Sınıflandırma**: 38 farklı bitki hastalığı türünü %90+ doğrulukla ayırt eder
- **Pratik Uygulama**: Çiftçiler ve tarım uzmanları için kullanılabilir bir araç sunar
- **Veri Tabanlı Karar**: Görsel temelli otomatik analiz ile subjektif değerlendirmeyi azaltır

## 🌟 Özellikler

- ✅ **Transfer Learning**: MobileNetV2 ve ResNet50 ile yüksek doğruluk
- ✅ **38 Sınıf**: PlantVillage dataset'inin tam kapsamı
- ✅ **Veri Artırma**: Rotation, zoom, flip gibi augmentasyon teknikleri
- ✅ **Detaylı Raporlama**: Karışıklık matrisi, ROC eğrileri, metrikler
- ✅ **Otomatik Optimizasyon**: Early stopping ve learning rate reduction
- ✅ **Modüler Kod**: Kolay anlaşılır ve genişletilebilir yapı

## 🏗️ Proje Yapısı

```
sayisal_goruntu/
├── src/
│   ├── preprocess/               # Veri hazırlama ve önişleme
│   │   ├── data_loader.py        # PlantVillage veri setini yükleme
│   │   └── data_splitter.py      # Train/Validation/Test bölme
│   ├── models/                   # Model tanımları
│   │   ├── cnn_models.py         # CNN modelleri
│   │   ├── transfer_learning.py  # Transfer learning modelleri
│   │   └── model_utils.py        # Yardımcı fonksiyonlar
│   ├── training/                 # Eğitim işlemleri
│   │   ├── data_generator.py     # TensorFlow dataset oluşturma
│   │   ├── trainer.py            # Model eğitici
│   │   └── training_config.py    # Eğitim konfigürasyonu
│   └── evaluation/               # Değerlendirme ve görselleştirme
│       ├── metrics.py            # Metrik hesaplama
│       ├── evaluator.py          # Model değerlendirici
│       ├── visualizer.py         # Grafik oluşturma
│       └── report_generator.py   # Raporlar
├── data/
│   ├── raw/                      # Ham veri (PlantVillage dataset)
│   └── splits/                   # Bölünmüş veri
├── models/
│   └── saved/                    # Eğitilmiş modeller
├── reports/                      # Raporlar ve grafikler
├── train_models.py               # Ana eğitim scripti
├── config.json                   # Konfigürasyon dosyası
├── requirements.txt              # Bağımlılıklar
└── README.md                     # Dokümantasyon
```

## 🚀 Kurulum

### 1. Gereksinimler

- **Python**: 3.8 veya üzeri
- **RAM**: Minimum 8GB (önerilen 16GB)
- **Depolama**: Veri seti için ~2GB boş alan
- **GPU**: Opsiyonel ancak önerilen (eğitim süresini ciddi oranda kısaltır)

### 2. Sanal Ortam Oluşturma

```bash
# Sanal ortam oluştur
python -m venv venv

# Sanal ortamı etkinleştir
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### 3. Paket Kurulumu

```bash
pip install -r requirements.txt
```

Ana paketler:
- TensorFlow 2.8+
- OpenCV
- scikit-image
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn

### 4. Veri Seti Hazırlama

PlantVillage dataset'ini indirin ve `data/raw/` klasörüne yerleştirin:

```bash
mkdir -p data/raw

# Veri seti yapısı:
data/raw/
├── Apple_Black_rot/
├── Apple_healthy/
├── Apple_scab/
├── Grape_Black_rot/
├── Grape_healthy/
├── Potato_healthy/
├── Tomato_healthy/
└── ... (38 sınıf toplam)
```

**Veri Seti Kaynağı**: Kaggle - PlantVillage Dataset

## 🎮 Kullanım

### Model Eğitimi

```bash
# 1. Sanal ortamı etkinleştir
venv\Scripts\activate

# 2. Eğitimi başlat
python train_models.py
```

Eğitim süresi:
- **CPU**: ~4-6 saat
- **GPU**: ~1-2 saat (önerilen)

## 🎓 Eğitim Süreci

Eğitim otomatik olarak şu adımları gerçekleştirir:

### 1. Veri Hazırlama
- PlantVillage dataset'i yüklenir
- 38 sınıf otomatik tespit edilir
- Veri seti otomatik bölünür: **Train (70%)** - **Validation (15%)** - **Test (15%)**
- Sınıf dağılımı gösterilir

### 2. Model Oluşturma
- **MobileNetV2** veya **ResNet50** transfer learning modelleri
- Base model dondurulur (ImageNet ağırlıkları kullanılır)
- Özel sınıflandırıcı katmanları eklenir (GlobalAveragePooling, Dense, Dropout)

### 3. Eğitim
- **Data Augmentation**: Rotation (20°), zoom (±20%), horizontal flip, brightness (±20%)
- **Early Stopping**: Patience=10 ile overfitting önlenir
- **Learning Rate Reduction**: Plateau tespit edildiğinde LR azaltılır
- **Model Checkpoint**: En iyi model otomatik kaydedilir

### 4. Değerlendirme
- Test seti üzerinde performans ölçülür
- Confusion matrix oluşturulur
- ROC curves çizilir
- Detaylı metrikler hesaplanır

## 📊 Sonuçlar ve Raporlar

Eğitim tamamlandığında `reports/` klasöründe şu dosyalar oluşturulur:

### 📈 Grafikler (Sunum İçin)
- **`*_training_history.png`**: Eğitim ve validation loss/accuracy grafikleri
- **`*_confusion_matrix.png`**: Detaylı karışıklık matrisi
- **`*_roc_curves.png`**: ROC eğrileri (her sınıf için)
- **Model karşılaştırması**: Birden fazla model eğitilirse

### 📋 Veri Dosyaları
- **`model_comparison.csv`**: Model karşılaştırma tablosu
- **`*_report.txt`**: Detaylı metrik raporları

### 💾 Eğitilmiş Modeller
- **`best_model_*.h5`**: En iyi performans gösteren model
- Model dosyası doğrudan tahmin için kullanılabilir

### 📊 Sunum Örneği

```
📁 reports/
├── mobilenet_v2_training_history.png    # Eğitim grafiği
├── mobilenet_v2_confusion_matrix.png    # Karışıklık matrisi
├── mobilenet_v2_roc_curves.png         # ROC eğrileri
├── model_comparison.csv                 # Model karşılaştırması
└── mobilenet_v2_report.txt             # Detaylı rapor
```

Bu grafikleri sunumunuzda kullanabilirsiniz!

## 📈 Performans Metrikleri

Sistem aşağıdaki metrikleri hesaplar:

- **Accuracy**: Genel doğruluk oranı
- **Precision**: Kesinlik (pozitif tahminlerin doğruluğu)
- **Recall**: Duyarlılık (gerçek pozitiflerin ne kadarının bulunduğu)
- **F1-Score**: Precision ve Recall'ın harmonik ortalaması
- **Confusion Matrix**: Sınıf bazlı hata analizi
- **ROC-AUC**: Sınıflandırma kalitesi göstergesi

**Beklenen Performans**: %90+ doğruluk (transfer learning ile)

## 🔧 Konfigürasyon

`config.json` dosyasından eğitim parametrelerini değiştirebilirsiniz:

```json
{
  "input_shape": [224, 224, 3],
  "num_classes": 38,
  "batch_size": 32,
  "epochs": 50,
  "learning_rate": 0.001,
  "patience": 10,
  "augmentation": true,
  "rotation_range": 20,
  "zoom_range": 0.2,
  "horizontal_flip": true,
  "dropout_rate": 0.5,
  "models_to_train": ["mobilenet_v2", "resnet50"]
}
```

## 📚 Teknolojiler

Bu proje aşağıdaki teknolojiler kullanılarak geliştirilmiştir:

- **Python 3.8+**: Ana programlama dili
- **TensorFlow/Keras**: Derin öğrenme framework'ü
- **OpenCV**: Görüntü işleme kütüphanesi
- **scikit-image**: Görüntü analizi
- **NumPy/Pandas**: Veri işleme
- **Matplotlib/Seaborn**: Veri görselleştirme
- **scikit-learn**: Makine öğrenmesi metrikleri

## 📄 Lisans

Bu proje **eğitim amaçlı** geliştirilmiştir. Ticari kullanım için gerekli lisans kontrolleri yapılmalıdır.

## 👥 Yazar

- **Ders**: Sayısal Görüntü İşleme
- **Proje Türü**: Bitki hastalık tespiti yapay zeka uygulaması
- **Teknikler**: Transfer Learning, CNN, Data Augmentation

## 🙏 Teşekkürler

- PlantVillage veri seti sağlayıcıları
- TensorFlow/Keras geliştirici ekibi
- Açık kaynak topluluğu
- İlgili tüm araştırma ve eğitim kaynakları

---

**Not**: Tüm kodlar açıklamalı ve modüler olarak yazılmıştır. Eğitim amaçlı öğrenme için ideal bir projedir.

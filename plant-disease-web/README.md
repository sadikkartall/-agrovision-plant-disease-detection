# 🌿 Plant Disease Detection Web Application

Bitki yaprak hastalığı tespiti için geliştirilmiş profesyonel web uygulaması. PlantVillage veri seti üzerinde eğitilmiş MobileNetV2 transfer learning modeli kullanarak 38 farklı bitki hastalığını tespit eder.

## 📋 Özellikler

- ✅ **38 Sınıf Tespit**: PlantVillage dataset'ine göre eğitilmiş model
- ✅ **Modern Web Arayüzü**: Bootstrap 5 ile responsive tasarım
- ✅ **Yüksek Doğruluk**: MobileNetV2 transfer learning ile %90+ doğruluk
- ✅ **Gerçek Zamanlı Tahmin**: Anında görüntü analizi
- ✅ **Detaylı Sonuçlar**: Top-3 tahmin ve tüm sınıf olasılıkları
- ✅ **Kullanıcı Dostu**: Drag & drop görüntü yükleme, önizleme

## 🛠️ Teknoloji Stack

- **Backend**: Flask (Python)
- **Deep Learning**: TensorFlow/Keras
- **Model**: MobileNetV2 (Transfer Learning)
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript
- **Image Processing**: Pillow (PIL)

## 📦 Kurulum

### 1. Gereksinimler

- Python 3.8 veya üzeri
- pip (Python paket yöneticisi)

### 2. Projeyi İndirin

```bash
git clone <repo-url>
cd plant-disease-web
```

### 3. Sanal Ortam Oluşturun

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Paketleri Yükleyin

```bash
pip install -r requirements.txt
```

### 5. Model Dosyasını Ekleyin

1. `mobilenetv2_best.keras` dosyasını `models/` klasörüne kopyalayın
2. `data/class_names.json` dosyasını kontrol edin ve gerekirse güncelleyin

**Klasör yapısı:**
```
plant-disease-web/
├── models/
│   └── mobilenetv2_best.keras   # ← Buraya model dosyasını koyun
├── data/
│   └── class_names.json         # Sınıf isimleri (zaten var)
└── app/
    └── ...
```

## 🚀 Çalıştırma

### Geliştirme Modu

```bash
# Windows
set FLASK_APP=app/main.py
flask run

# Linux/Mac
export FLASK_APP=app/main.py
flask run
```

**Alternatif (doğrudan Python ile):**
```bash
python app/main.py
```

Uygulama şu adreste çalışacak: **http://127.0.0.1:5000**

### Production Modu (Gunicorn)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app.main:app
```

## 📖 Kullanım

1. Web tarayıcınızda `http://127.0.0.1:5000` adresine gidin
2. Sol taraftan bir bitki yaprak görüntüsü seçin (JPG, JPEG, PNG)
3. "Tahmin Yap" butonuna tıklayın
4. Sağ tarafta tahmin sonuçlarını görüntüleyin:
   - **Tahmin Edilen Sınıf**: En yüksek olasılıklı hastalık/sağlık durumu
   - **Güven Skoru**: Tahminin güvenilirlik yüzdesi
   - **Top-3 Tahminler**: En yüksek 3 olasılık
   - **Tüm Sınıf Olasılıkları**: Detaylı istatistikler

## 📊 Model Bilgileri

- **Model**: MobileNetV2
- **Input Shape**: 224x224x3 (RGB)
- **Preprocessing**: `tf.keras.applications.mobilenet_v2.preprocess_input`
- **Sınıf Sayısı**: 38
- **Veri Seti**: PlantVillage Dataset
- **Eğitim Yöntemi**: Transfer Learning (ImageNet ağırlıkları)

## 🔧 Yapılandırma

### Model Yolu Değiştirme

`app/model_loader.py` dosyasında `load_model()` fonksiyonundaki varsayılan yolu değiştirebilirsiniz:

```python
model = load_model("models/mobilenetv2_best.keras")  # Yol buradan değiştirilebilir
```

### Sınıf İsimleri Güncelleme

`data/class_names.json` dosyasını düzenleyerek sınıf isimlerini güncelleyebilirsiniz. **Önemli**: Sınıf sırası, model eğitimindeki sırayla aynı olmalıdır!

## 🚀 Deployment (Railway/Render)

Bu uygulama Railway veya Render gibi PaaS platformlarında kolayca deploy edilebilir.

### Önkoşullar

1. Projenin GitHub'da bir repository'si olmalı
2. `models/mobilenetv2_best.keras` dosyası repository'de olmalı (Git LFS kullanabilirsiniz)
3. `data/class_names.json` dosyası repository'de olmalı

### Railway ile Deployment

#### 1. Projeyi GitHub'a Push Edin

```bash
# Git repository'sini başlat (eğer yoksa)
git init
git add .
git commit -m "Initial commit: Plant Disease Detection Web App"

# GitHub'da yeni bir repository oluşturun, sonra:
git remote add origin https://github.com/yourusername/plant-disease-web.git
git branch -M main
git push -u origin main
```

**Önemli:** `models/mobilenetv2_best.keras` dosyası büyük olabilir. Git LFS kullanmanız önerilir:

```bash
# Git LFS kurulumu (ilk kez)
git lfs install
git lfs track "*.keras"
git add .gitattributes
git add models/mobilenetv2_best.keras
git commit -m "Add model file with Git LFS"
git push
```

#### 2. Railway'de Yeni Proje Oluşturun

1. [Railway.app](https://railway.app) adresine gidin ve hesabınızla giriş yapın
2. "New Project" butonuna tıklayın
3. "Deploy from GitHub repo" seçeneğini seçin
4. GitHub repository'nizi seçin
5. Railway otomatik olarak Python projesini algılayacak

#### 3. Yapılandırma

Railway otomatik olarak şunları algılar:
- **Python Version**: `requirements.txt` dosyasından
- **Start Command**: `Procfile` dosyasından veya manuel olarak ayarlayın

**Start Command** (Railway Settings → Deploy → Start Command):
```
gunicorn app.main:app --bind 0.0.0.0:$PORT
```

Veya `Procfile` dosyası zaten mevcut olduğu için Railway otomatik olarak kullanacaktır.

#### 4. Environment Variables (Opsiyonel)

Railway Settings → Variables bölümünden gerekirse environment variable'lar ekleyebilirsiniz:
- `SECRET_KEY`: Flask secret key (production için önerilir)
- `FLASK_ENV`: `production` (default)

#### 5. Deploy

Railway otomatik olarak deploy edecektir. Deploy tamamlandığında:
- Railway size bir public URL verecektir (örn: `https://your-app-name.up.railway.app`)
- Bu URL'den uygulamanıza erişebilirsiniz

### Render ile Deployment

#### 1. Projeyi GitHub'a Push Edin

Yukarıdaki "Railway ile Deployment" bölümündeki adımları takip edin.

#### 2. Render'da Yeni Web Service Oluşturun

1. [Render.com](https://render.com) adresine gidin ve hesabınızla giriş yapın
2. Dashboard'dan "New +" → "Web Service" seçin
3. GitHub repository'nizi bağlayın
4. Aşağıdaki ayarları yapın:

**Settings:**
- **Name**: `plant-disease-web` (veya istediğiniz isim)
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app.main:app --bind 0.0.0.0:$PORT`

#### 3. Deploy

Render otomatik olarak deploy edecektir. Deploy tamamlandığında:
- Render size bir public URL verecektir (örn: `https://plant-disease-web.onrender.com`)
- Bu URL'den uygulamanıza erişebilirsiniz

### Deployment Sonrası Kontrol

1. Public URL'nizi tarayıcıda açın
2. `/health` endpoint'ini kontrol edin: `https://your-app-url.com/health`
3. Ana sayfadan bir görüntü yükleyip tahmin yapmayı test edin

### Notlar

- **Model Dosyası Boyutu**: `mobilenetv2_best.keras` dosyası büyük olabilir. Git LFS kullanmanız önerilir.
- **Build Time**: İlk deploy sırasında TensorFlow kurulumu biraz zaman alabilir (5-10 dakika).
- **Cold Start**: Render'da free tier kullanıyorsanız, uygulama 15 dakika kullanılmadıktan sonra "sleep" moduna geçer. İlk istek biraz yavaş olabilir.
- **Memory**: TensorFlow modeli yüklemek için yeterli RAM gerekir. Railway/Render free tier'ları genellikle yeterlidir.

## 🐳 Docker ile Çalıştırma (Opsiyonel)

```bash
# Dockerfile oluşturulduktan sonra
docker build -t plant-disease-web .
docker run -p 5000:5000 plant-disease-web
```

## 📝 API Endpoints

### GET `/`
Ana sayfa - Web arayüzü

### POST `/predict`
Görüntü tahmini yapar
- **Form Data**: `file` (image file)
- **Response**: HTML sayfası (tahmin sonuçları ile)

### GET `/health`
Sağlık kontrolü - Model ve sınıf isimlerinin yüklü olup olmadığını kontrol eder
- **Response**: JSON
```json
{
  "status": "ok",
  "message": "Model ve sınıf isimleri yüklü",
  "num_classes": 38,
  "model_input_shape": "(None, 224, 224, 3)",
  "model_output_shape": "(None, 38)"
}
```

## 🐛 Sorun Giderme

### Model Yüklenemedi
- `models/mobilenetv2_best.keras` dosyasının varlığını kontrol edin
- Dosya yolunun doğru olduğundan emin olun

### Sınıf İsimleri Yüklenemedi
- `data/class_names.json` dosyasının varlığını kontrol edin
- JSON formatının doğru olduğundan emin olun (string listesi)

### Tahmin Hatası
- Yüklenen görüntünün geçerli bir format olduğundan emin olun (JPG, JPEG, PNG)
- Görüntü boyutunun çok büyük olmadığından emin olun (max 10MB)

## 📄 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

## 👨‍💻 Geliştirici

Plant Disease Detection System - Powered by MobileNetV2 & TensorFlow

## 🙏 Teşekkürler

- **PlantVillage Dataset**: Bitki hastalığı görüntüleri için
- **TensorFlow/Keras**: Deep learning framework
- **Flask**: Web framework
- **Bootstrap**: UI framework


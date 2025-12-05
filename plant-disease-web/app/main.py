"""
Flask ana uygulama dosyası
Bitki hastalık tespiti web arayüzü
"""

import os
import sys
import base64
from pathlib import Path
from io import BytesIO
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image

# Proje root'unu Python path'ine ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Kendi modüllerimizi import et
from app.model_loader import load_model, load_class_names
from app.inference import prepare_image, predict, get_top_predictions
from app.utils import validate_file, allowed_file

# Flask uygulamasını oluştur
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Global değişkenler (uygulama başlangıcında yüklenecek)
model = None
class_names = []


def initialize_app():
    """
    Uygulama başlangıcında modeli ve sınıf isimlerini yükler
    """
    global model, class_names
    
    try:
        print("=" * 50)
        print("Uygulama başlatılıyor...")
        print("=" * 50)
        
        # Proje root'unu bul
        project_root = Path(__file__).parent.parent
        
        # Model ve data yollarını oluştur
        model_path = project_root / "models" / "mobilenetv2_best.keras"
        class_names_path = project_root / "data" / "class_names.json"
        
        # Modeli yükle
        model = load_model(str(model_path))
        
        # Sınıf isimlerini yükle
        class_names = load_class_names(str(class_names_path))
        
        print("=" * 50)
        print("Uygulama hazır!")
        print("=" * 50)
        
    except FileNotFoundError as e:
        print(f"❌ HATA: {e}")
        print("Lütfen model dosyasını ve class_names.json dosyasını kontrol edin.")
    except Exception as e:
        print(f"❌ Beklenmeyen hata: {e}")


# Uygulama başlangıcında modeli yükle
initialize_app()


@app.route('/')
def index():
    """
    Ana sayfa
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_route():
    """
    Tahmin endpoint'i
    POST isteği ile görüntü alır ve tahmin yapar
    """
    global model, class_names
    
    # Model yüklenmemişse hata döndür
    if model is None or len(class_names) == 0:
        flash("Model yüklenemedi. Lütfen model dosyalarını kontrol edin.", "error")
        return redirect(url_for('index'))
    
    # Dosya kontrolü
    if 'file' not in request.files:
        flash("Dosya seçilmedi. Lütfen bir görüntü dosyası seçin.", "error")
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # Dosya validasyonu
    is_valid, error_message = validate_file(file)
    if not is_valid:
        flash(error_message, "error")
        return redirect(url_for('index'))
    
    try:
        # Görüntüyü hazırla
        image_array = prepare_image(file)
        
        # Tahmin yap
        predicted_label, confidence, probabilities = predict(image_array, model, class_names)
        
        # Top-3 tahminleri al
        top_predictions = get_top_predictions(probabilities, top_k=3)
        
        # Görüntüyü base64'e çevir (önizleme için)
        file.seek(0)  # Dosyayı başa al
        image = Image.open(file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Görüntüyü küçült (önizleme için)
        image.thumbnail((400, 400), Image.Resampling.LANCZOS)
        
        # Base64'e çevir
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Sonuçları template'e gönder
        return render_template('index.html',
                             prediction=predicted_label,
                             confidence=confidence,
                             top_predictions=top_predictions,
                             all_predictions=probabilities,
                             image_base64=img_str,
                             has_result=True)
    
    except Exception as e:
        print(f"Tahmin hatası: {e}")
        flash(f"Tahmin yapılırken bir hata oluştu: {str(e)}", "error")
        return redirect(url_for('index'))


@app.route('/health', methods=['GET'])
def health_check():
    """
    Sağlık kontrolü endpoint'i
    Model ve sınıf isimlerinin yüklenip yüklenmediğini kontrol eder
    """
    global model, class_names
    
    if model is None:
        return jsonify({
            "status": "error",
            "message": "Model yüklenemedi"
        }), 500
    
    if len(class_names) == 0:
        return jsonify({
            "status": "error",
            "message": "Sınıf isimleri yüklenemedi"
        }), 500
    
    return jsonify({
        "status": "ok",
        "message": "Model ve sınıf isimleri yüklü",
        "num_classes": len(class_names),
        "model_input_shape": str(model.input_shape),
        "model_output_shape": str(model.output_shape)
    })


if __name__ == '__main__':
    # Development mode: Flask development server
    # Production mode: Use gunicorn (gunicorn app.main:app)
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)


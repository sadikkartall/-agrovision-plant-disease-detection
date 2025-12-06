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
        print(f"📂 Project root: {project_root}")
        print(f"📂 Current working directory: {os.getcwd()}")
        
        # Model ve data yollarını oluştur
        model_path = project_root / "models" / "mobilenetv2_best.keras"
        class_names_path = project_root / "data" / "class_names.json"
        
        # Dosya varlığını kontrol et
        print(f"🔍 Checking model path: {model_path}")
        print(f"   Exists: {model_path.exists()}")
        if model_path.exists():
            print(f"   Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
        
        print(f"🔍 Checking class names path: {class_names_path}")
        print(f"   Exists: {class_names_path.exists()}")
        
        # Dizin içeriğini listele (debug için)
        models_dir = project_root / "models"
        data_dir = project_root / "data"
        print(f"📂 Models directory contents:")
        if models_dir.exists():
            for item in models_dir.iterdir():
                print(f"   - {item.name} ({item.stat().st_size / (1024*1024):.2f} MB)" if item.is_file() else f"   - {item.name}/")
        else:
            print("   ❌ Models directory does not exist!")
        
        print(f"📂 Data directory contents:")
        if data_dir.exists():
            for item in data_dir.iterdir():
                print(f"   - {item.name}")
        else:
            print("   ❌ Data directory does not exist!")
        
        # Modeli yükle
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model dosyası bulunamadı: {model_path}\n"
                f"Lütfen mobilenetv2_best.keras dosyasını models/ klasörüne koyun.\n"
                f"Git LFS kullanıyorsanız, build sırasında 'git lfs pull' komutunu çalıştırdığınızdan emin olun."
            )
        
        model = load_model(str(model_path))
        
        # Sınıf isimlerini yükle
        if not class_names_path.exists():
            raise FileNotFoundError(
                f"Sınıf isimleri dosyası bulunamadı: {class_names_path}\n"
                f"Lütfen class_names.json dosyasını data/ klasörüne koyun."
            )
        
        class_names = load_class_names(str(class_names_path))
        
        print("=" * 50)
        print("✅ Uygulama hazır!")
        print("=" * 50)
        
    except FileNotFoundError as e:
        print(f"❌ HATA: {e}")
        print("Lütfen model dosyasını ve class_names.json dosyasını kontrol edin.")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()


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
    
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "mobilenetv2_best.keras"
    class_names_path = project_root / "data" / "class_names.json"
    
    response = {
        "status": "ok" if model is not None and len(class_names) > 0 else "error",
        "model_loaded": model is not None,
        "class_names_loaded": len(class_names) > 0,
        "model_file_exists": model_path.exists(),
        "class_names_file_exists": class_names_path.exists(),
        "model_path": str(model_path),
        "class_names_path": str(class_names_path),
        "current_directory": os.getcwd(),
        "project_root": str(project_root)
    }
    
    if model is not None:
        response["num_classes"] = len(class_names)
        response["model_input_shape"] = str(model.input_shape)
        response["model_output_shape"] = str(model.output_shape)
    else:
        response["message"] = "Model yüklenemedi"
        if not model_path.exists():
            response["error"] = f"Model dosyası bulunamadı: {model_path}"
    
    if len(class_names) == 0:
        response["message"] = "Sınıf isimleri yüklenemedi"
        if not class_names_path.exists():
            response["error"] = f"Class names dosyası bulunamadı: {class_names_path}"
    
    status_code = 200 if response["status"] == "ok" else 500
    return jsonify(response), status_code


if __name__ == '__main__':
    # Development mode: Flask development server
    # Production mode: Use gunicorn (gunicorn app.main:app)
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)


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
        
        # Proje root'unu bul - Railway'de root directory plant-disease-web olabilir
        current_dir = Path(os.getcwd())
        file_dir = Path(__file__).parent.parent
        
        # Railway'de root directory plant-disease-web ise, working directory zaten orada
        # Eğer models/ klasörü current_dir'de varsa, orayı kullan
        if (current_dir / "models" / "mobilenetv2_best.keras").exists():
            project_root = current_dir
        elif (file_dir / "models" / "mobilenetv2_best.keras").exists():
            project_root = file_dir
        else:
            # Son çare: current_dir'i kullan
            project_root = current_dir
        
        print(f"📂 Project root: {project_root}")
        print(f"📂 Current working directory: {os.getcwd()}")
        print(f"📂 File directory: {file_dir}")
        
        # Model ve data yollarını oluştur - birden fazla yol dene
        possible_model_paths = [
            project_root / "models" / "mobilenetv2_best.keras",
            current_dir / "models" / "mobilenetv2_best.keras",
            file_dir / "models" / "mobilenetv2_best.keras",
            Path("models") / "mobilenetv2_best.keras",
        ]
        
        possible_class_paths = [
            project_root / "data" / "class_names.json",
            current_dir / "data" / "class_names.json",
            file_dir / "data" / "class_names.json",
            Path("data") / "class_names.json",
        ]
        
        # İlk var olan yolu bul
        model_path = None
        for path in possible_model_paths:
            if path.exists():
                model_path = path
                break
        
        class_names_path = None
        for path in possible_class_paths:
            if path.exists():
                class_names_path = path
                break
        
        # Dosya varlığını kontrol et
        print(f"🔍 Checking model paths:")
        for path in possible_model_paths:
            exists = path.exists()
            print(f"   {path}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
            if exists:
                print(f"      Size: {path.stat().st_size / (1024*1024):.2f} MB")
        
        print(f"🔍 Checking class names paths:")
        for path in possible_class_paths:
            exists = path.exists()
            print(f"   {path}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
        
        # Dizin içeriğini listele (debug için)
        print(f"📂 Current directory tree (top 3 levels):")
        try:
            for root, dirs, files in os.walk(current_dir):
                level = root.replace(str(current_dir), '').count(os.sep)
                if level <= 2:
                    indent = ' ' * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files[:5]:  # İlk 5 dosya
                        print(f"{subindent}{file}")
                    if len(files) > 5:
                        print(f"{subindent}... ({len(files)} more files)")
        except Exception as e:
            print(f"   Error listing directory: {e}")
        
        # Modeli yükle
        if model_path is None or not model_path.exists():
            error_msg = "Model dosyası bulunamadı. Denenen yollar:\n"
            for path in possible_model_paths:
                error_msg += f"  - {path}\n"
            error_msg += "\nLütfen mobilenetv2_best.keras dosyasını models/ klasörüne koyun.\n"
            error_msg += "Git LFS kullanıyorsanız, build sırasında 'git lfs pull' komutunu çalıştırdığınızdan emin olun."
            raise FileNotFoundError(error_msg)
        
        print(f"✅ Using model path: {model_path}")
        model = load_model(str(model_path))
        
        # Sınıf isimlerini yükle
        if class_names_path is None or not class_names_path.exists():
            error_msg = "Sınıf isimleri dosyası bulunamadı. Denenen yollar:\n"
            for path in possible_class_paths:
                error_msg += f"  - {path}\n"
            error_msg += "\nLütfen class_names.json dosyasını data/ klasörüne koyun."
            raise FileNotFoundError(error_msg)
        
        print(f"✅ Using class names path: {class_names_path}")
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


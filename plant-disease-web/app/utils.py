"""
Genel yardımcı fonksiyonlar
Görüntü ön işleme ve validasyon fonksiyonları
"""

import os
from typing import Tuple
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np


# İzin verilen dosya uzantıları
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Maksimum dosya boyutu (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024


def allowed_file(filename: str) -> bool:
    """
    Dosya uzantısının izin verilen listede olup olmadığını kontrol eder
    
    Args:
        filename (str): Dosya adı
        
    Returns:
        bool: İzin verilen uzantı ise True
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_file(file) -> Tuple[bool, str]:
    """
    Yüklenen dosyayı doğrular
    
    Args:
        file: Flask file object
        
    Returns:
        Tuple[bool, str]: (geçerli mi, hata mesajı)
    """
    if file is None or file.filename == '':
        return False, "Dosya seçilmedi. Lütfen bir görüntü dosyası seçin."
    
    if not allowed_file(file.filename):
        return False, f"Desteklenmeyen dosya formatı. İzin verilen formatlar: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Dosya boyutunu kontrol et
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Dosyayı başa al
    
    if file_size > MAX_FILE_SIZE:
        return False, f"Dosya çok büyük. Maksimum boyut: {MAX_FILE_SIZE / (1024*1024):.1f} MB"
    
    if file_size == 0:
        return False, "Dosya boş."
    
    return True, ""


def get_secure_filename(filename: str) -> str:
    """
    Güvenli dosya adı oluşturur
    
    Args:
        filename (str): Orijinal dosya adı
        
    Returns:
        str: Güvenli dosya adı
    """
    return secure_filename(filename)


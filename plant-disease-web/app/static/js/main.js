/**
 * Plant Disease Detection - Main JavaScript
 * Form handling ve UI iyileştirmeleri
 */

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictForm');
    const fileInput = document.getElementById('fileInput');
    const predictBtn = document.getElementById('predictBtn');
    
    // Dosya seçildiğinde önizleme göster
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                // Dosya boyutunu kontrol et (10MB)
                const maxSize = 10 * 1024 * 1024; // 10MB
                if (file.size > maxSize) {
                    alert('Dosya çok büyük! Maksimum boyut: 10MB');
                    fileInput.value = '';
                    return;
                }
                
                // Dosya formatını kontrol et
                const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];
                if (!allowedTypes.includes(file.type)) {
                    alert('Desteklenmeyen dosya formatı! Lütfen JPG, JPEG veya PNG formatında bir dosya seçin.');
                    fileInput.value = '';
                    return;
                }
            }
        });
    }
    
    // Form submit olduğunda loading state
    if (form) {
        form.addEventListener('submit', function(e) {
            // Dosya seçilmiş mi kontrol et
            if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
                e.preventDefault();
                alert('Lütfen bir görüntü dosyası seçin!');
                return;
            }
            
            // Butona loading state ekle
            if (predictBtn) {
                predictBtn.disabled = true;
                predictBtn.classList.add('btn-loading');
                predictBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> Tahmin Yapılıyor...';
            }
        });
    }
    
    // Alert'leri otomatik kapat (5 saniye sonra)
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(function(alert) {
        if (alert.classList.contains('alert-dismissible')) {
            setTimeout(function() {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }, 5000);
        }
    });
    
    // Smooth scroll to results if result exists
    if (document.querySelector('.card-header.bg-success')) {
        setTimeout(function() {
            document.querySelector('.card-header.bg-success').scrollIntoView({
                behavior: 'smooth',
                block: 'nearest'
            });
        }, 300);
    }
});


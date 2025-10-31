"""
Eğitim konfigürasyon modülü
Eğitim parametrelerini yönetir
"""

from typing import Dict, Any, List
from dataclasses import dataclass
import json
import os


@dataclass
class TrainingConfig:
    """
    Eğitim konfigürasyon sınıfı
    """
    
    # Veri parametreleri
    input_shape: tuple = (224, 224, 3)
    num_classes: int = 10
    batch_size: int = 32
    
    # Eğitim parametreleri
    epochs: int = 100
    learning_rate: float = 0.001
    patience: int = 10
    monitor: str = 'val_accuracy'
    mode: str = 'max'
    
    # Veri artırma parametreleri
    augmentation: bool = True
    rotation_range: int = 20
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    horizontal_flip: bool = True
    zoom_range: float = 0.2
    brightness_range: tuple = (0.8, 1.2)
    
    # Model parametreleri
    dropout_rate: float = 0.5
    fine_tune_layers: int = 10
    
    # Optimizasyon parametreleri
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konfigürasyonu sözlük olarak döndürür
        
        Returns:
            Dict[str, Any]: Konfigürasyon sözlüğü
        """
        return {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'patience': self.patience,
            'monitor': self.monitor,
            'mode': self.mode,
            'augmentation': self.augmentation,
            'rotation_range': self.rotation_range,
            'width_shift_range': self.width_shift_range,
            'height_shift_range': self.height_shift_range,
            'horizontal_flip': self.horizontal_flip,
            'zoom_range': self.zoom_range,
            'brightness_range': self.brightness_range,
            'dropout_rate': self.dropout_rate,
            'fine_tune_layers': self.fine_tune_layers,
            'reduce_lr_patience': self.reduce_lr_patience,
            'reduce_lr_factor': self.reduce_lr_factor,
            'min_lr': self.min_lr
        }
    
    def save(self, filepath: str) -> None:
        """
        Konfigürasyonu dosyaya kaydeder
        
        Args:
            filepath (str): Kayıt dosya yolu
        """
        # Dizin oluştur
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # JSON olarak kaydet
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"Konfigürasyon kaydedildi: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """
        Konfigürasyonu dosyadan yükler
        
        Args:
            filepath (str): Yükleme dosya yolu
            
        Returns:
            TrainingConfig: Yüklenen konfigürasyon
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Konfigürasyon dosyası bulunamadı: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # TrainingConfig nesnesi oluştur
        config = cls()
        
        # Değerleri ata
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        print(f"Konfigürasyon yüklendi: {filepath}")
        return config
    
    def update(self, **kwargs) -> None:
        """
        Konfigürasyonu günceller
        
        Args:
            **kwargs: Güncellenecek parametreler
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Uyarı: Geçersiz parametre: {key}")
    
    def print_config(self) -> None:
        """
        Konfigürasyonu yazdırır
        """
        print("EĞİTİM KONFİGÜRASYONU")
        print("=" * 40)
        
        config_dict = self.to_dict()
        
        for key, value in config_dict.items():
            print(f"{key}: {value}")
    
    def get_model_configs(self) -> List[Dict[str, Any]]:
        """
        Farklı model konfigürasyonlarını döndürür
        
        Returns:
            List[Dict[str, Any]]: Model konfigürasyonları
        """
        model_configs = [
            {
                'name': 'basic_cnn',
                'type': 'cnn',
                'description': 'Temel CNN modeli',
                'params': {
                    'dropout_rate': self.dropout_rate,
                    'learning_rate': self.learning_rate
                }
            },
            {
                'name': 'deep_cnn',
                'type': 'cnn',
                'description': 'Derin CNN modeli',
                'params': {
                    'dropout_rate': self.dropout_rate,
                    'learning_rate': self.learning_rate
                }
            },
            {
                'name': 'residual_cnn',
                'type': 'cnn',
                'description': 'Residual CNN modeli',
                'params': {
                    'dropout_rate': self.dropout_rate,
                    'learning_rate': self.learning_rate
                }
            },
            {
                'name': 'attention_cnn',
                'type': 'cnn',
                'description': 'Attention CNN modeli',
                'params': {
                    'dropout_rate': self.dropout_rate,
                    'learning_rate': self.learning_rate
                }
            },
            {
                'name': 'mobilenet_v2',
                'type': 'transfer_learning',
                'description': 'MobileNetV2 transfer learning',
                'params': {
                    'dropout_rate': self.dropout_rate,
                    'learning_rate': self.learning_rate,
                    'freeze_base': True
                }
            },
            {
                'name': 'resnet50',
                'type': 'transfer_learning',
                'description': 'ResNet50 transfer learning',
                'params': {
                    'dropout_rate': self.dropout_rate,
                    'learning_rate': self.learning_rate,
                    'freeze_base': True
                }
            },
            {
                'name': 'efficientnet_b0',
                'type': 'transfer_learning',
                'description': 'EfficientNetB0 transfer learning',
                'params': {
                    'dropout_rate': self.dropout_rate,
                    'learning_rate': self.learning_rate,
                    'freeze_base': True
                }
            }
        ]
        
        return model_configs
    
    def get_optimized_configs(self) -> List[Dict[str, Any]]:
        """
        Optimize edilmiş konfigürasyonları döndürür
        
        Returns:
            List[Dict[str, Any]]: Optimize edilmiş konfigürasyonlar
        """
        optimized_configs = []
        
        # Farklı öğrenme oranları
        learning_rates = [0.001, 0.0001, 0.00001]
        
        # Farklı dropout oranları
        dropout_rates = [0.3, 0.5, 0.7]
        
        # Farklı batch boyutları
        batch_sizes = [16, 32, 64]
        
        for lr in learning_rates:
            for dropout in dropout_rates:
                for batch_size in batch_sizes:
                    config = {
                        'learning_rate': lr,
                        'dropout_rate': dropout,
                        'batch_size': batch_size,
                        'epochs': self.epochs,
                        'patience': self.patience
                    }
                    optimized_configs.append(config)
        
        return optimized_configs
    
    def validate_config(self) -> List[str]:
        """
        Konfigürasyonu doğrular
        
        Returns:
            List[str]: Hata mesajları
        """
        errors = []
        
        # Pozitif değerler kontrolü
        if self.batch_size <= 0:
            errors.append("Batch size pozitif olmalıdır")
        
        if self.epochs <= 0:
            errors.append("Epochs pozitif olmalıdır")
        
        if self.learning_rate <= 0:
            errors.append("Learning rate pozitif olmalıdır")
        
        if self.patience <= 0:
            errors.append("Patience pozitif olmalıdır")
        
        # Aralık kontrolleri
        if not (0 <= self.dropout_rate <= 1):
            errors.append("Dropout rate 0-1 arasında olmalıdır")
        
        if not (0 <= self.width_shift_range <= 1):
            errors.append("Width shift range 0-1 arasında olmalıdır")
        
        if not (0 <= self.height_shift_range <= 1):
            errors.append("Height shift range 0-1 arasında olmalıdır")
        
        if not (0 <= self.zoom_range <= 1):
            errors.append("Zoom range 0-1 arasında olmalıdır")
        
        # Boyut kontrolleri
        if len(self.input_shape) != 3:
            errors.append("Input shape 3 boyutlu olmalıdır (height, width, channels)")
        
        if self.input_shape[2] != 3:
            errors.append("Input shape 3 kanal (RGB) olmalıdır")
        
        return errors

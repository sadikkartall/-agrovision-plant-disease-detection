"""
Veri yükleme modülü
PlantVillage veri setini yüklemek ve organize etmek için kullanılır
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
from sklearn.preprocessing import LabelEncoder


class DataLoader:
    """
    PlantVillage veri setini yüklemek ve organize etmek için sınıf
    """
    
    def __init__(self, data_path: str):
        """
        DataLoader sınıfını başlatır
        
        Args:
            data_path (str): Veri setinin bulunduğu klasör yolu
        """
        self.data_path = Path(data_path)
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.image_paths = []
        self.labels = []
        
    def load_dataset(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Veri setini yükler ve organize eder
        
        Returns:
            Tuple[List[str], List[str], List[str]]: (image_paths, labels, class_names)
        """
        print("Veri seti yükleniyor...")
        
        # Tüm alt klasörleri tara (her klasör bir sınıfı temsil eder)
        for class_folder in self.data_path.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name
                self.class_names.append(class_name)
                
                # Bu sınıfa ait tüm görüntüleri al
                for image_file in class_folder.glob("*.jpg"):
                    self.image_paths.append(str(image_file))
                    self.labels.append(class_name)
        
        # Sınıf isimlerini alfabetik olarak sırala
        self.class_names = sorted(self.class_names)
        
        # Label encoder'ı eğit
        self.label_encoder.fit(self.labels)
        
        print(f"Toplam {len(self.image_paths)} görüntü yüklendi")
        print(f"Toplam {len(self.class_names)} sınıf bulundu")
        print(f"Sınıflar: {self.class_names}")
        
        return self.image_paths, self.labels, self.class_names
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Sınıf dağılımını hesaplar
        
        Returns:
            Dict[str, int]: Sınıf isimleri ve örnek sayıları
        """
        class_counts = {}
        for label in self.labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        return class_counts
    
    def create_dataframe(self) -> pd.DataFrame:
        """
        Veri setini pandas DataFrame olarak döndürür
        
        Returns:
            pd.DataFrame: Görüntü yolları ve etiketleri içeren DataFrame
        """
        df = pd.DataFrame({
            'image_path': self.image_paths,
            'label': self.labels,
            'class_name': self.labels
        })
        
        # Numeric label ekle
        df['numeric_label'] = self.label_encoder.transform(self.labels)
        
        return df
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Tek bir görüntüyü yükler
        
        Args:
            image_path (str): Görüntü dosya yolu
            
        Returns:
            np.ndarray: Yüklenen görüntü
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Görüntü yüklenemedi: {image_path}")
        
        # BGR'den RGB'ye çevir
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def get_sample_images(self, n_samples: int = 5) -> List[np.ndarray]:
        """
        Her sınıftan örnek görüntüler alır
        
        Args:
            n_samples (int): Her sınıftan alınacak örnek sayısı
            
        Returns:
            List[np.ndarray]: Örnek görüntüler
        """
        sample_images = []
        
        for class_name in self.class_names:
            class_indices = [i for i, label in enumerate(self.labels) if label == class_name]
            
            # Her sınıftan n_samples kadar örnek al
            selected_indices = np.random.choice(class_indices, 
                                             min(n_samples, len(class_indices)), 
                                             replace=False)
            
            for idx in selected_indices:
                image = self.load_image(self.image_paths[idx])
                sample_images.append(image)
        
        return sample_images

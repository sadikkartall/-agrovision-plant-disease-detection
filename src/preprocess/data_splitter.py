"""
Veri bölme modülü
Veri setini eğitim, doğrulama ve test setlerine böler
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, List
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


class DataSplitter:
    """
    Veri setini bölmek için sınıf
    """
    
    def __init__(self, train_ratio: float = 0.7, 
                 val_ratio: float = 0.15, 
                 test_ratio: float = 0.15,
                 random_state: int = 42):
        """
        DataSplitter sınıfını başlatır
        
        Args:
            train_ratio (float): Eğitim seti oranı
            val_ratio (float): Doğrulama seti oranı
            test_ratio (float): Test seti oranı
            random_state (int): Rastgele sayı üreteci seed değeri
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        # Oranların toplamının 1.0 olduğunu kontrol et
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Oranların toplamı 1.0 olmalıdır")
    
    def split_dataframe(self, df: pd.DataFrame, 
                       stratify_column: str = 'numeric_label') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        DataFrame'i eğitim, doğrulama ve test setlerine böler
        
        Args:
            df (pd.DataFrame): Bölünecek DataFrame
            stratify_column (str): Stratifikasyon için kullanılacak sütun
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
        """
        print("Veri seti bölünüyor...")
        
        # Stratify column'un var olduğunu kontrol et
        if stratify_column not in df.columns:
            # Eğer numeric_label yoksa class_name kullan
            if 'class_name' in df.columns:
                stratify_column = 'class_name'
            else:
                stratify_column = None
        
        # İlk olarak eğitim ve geçici set (val + test) olarak böl
        if stratify_column:
            train_df, temp_df = train_test_split(
                df, 
                test_size=(self.val_ratio + self.test_ratio),
                stratify=df[stratify_column],
                random_state=self.random_state
            )
        else:
            train_df, temp_df = train_test_split(
                df, 
                test_size=(self.val_ratio + self.test_ratio),
                random_state=self.random_state
            )
        
        # Geçici seti doğrulama ve test olarak böl
        val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
        if stratify_column:
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1 - val_size),
                stratify=temp_df[stratify_column],
                random_state=self.random_state
            )
        else:
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1 - val_size),
                random_state=self.random_state
            )
        
        print(f"Eğitim seti: {len(train_df)} örnek")
        print(f"Doğrulama seti: {len(val_df)} örnek")
        print(f"Test seti: {len(test_df)} örnek")
        
        return train_df, val_df, test_df
    
    def create_directory_structure(self, output_dir: str) -> None:
        """
        Çıktı dizin yapısını oluşturur
        
        Args:
            output_dir (str): Çıktı dizini yolu
        """
        output_path = Path(output_dir)
        
        # Ana dizinleri oluştur
        (output_path / 'train').mkdir(parents=True, exist_ok=True)
        (output_path / 'val').mkdir(parents=True, exist_ok=True)
        (output_path / 'test').mkdir(parents=True, exist_ok=True)
        
        print(f"Dizin yapısı oluşturuldu: {output_dir}")
    
    def copy_images_to_splits(self, df: pd.DataFrame, 
                             source_dir: str, 
                             output_dir: str,
                             split_name: str) -> None:
        """
        Görüntüleri belirtilen bölüme kopyalar
        
        Args:
            df (pd.DataFrame): Kopyalanacak görüntülerin DataFrame'i
            source_dir (str): Kaynak dizin
            output_dir (str): Çıktı dizini
            split_name (str): Bölüm adı ('train', 'val', 'test')
        """
        source_path = Path(source_dir)
        output_path = Path(output_dir) / split_name
        
        print(f"{split_name} seti için görüntüler kopyalanıyor...")
        
        for idx, row in df.iterrows():
            source_file = Path(row['image_path'])
            class_name = row['class_name']
            
            # Hedef dizin oluştur
            target_dir = output_path / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Dosyayı kopyala
            target_file = target_dir / source_file.name
            shutil.copy2(source_file, target_file)
        
        print(f"{split_name} seti için {len(df)} görüntü kopyalandı")
    
    def split_and_organize_dataset(self, 
                                  df: pd.DataFrame, 
                                  source_dir: str, 
                                  output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Veri setini böler ve dosyaları organize eder
        
        Args:
            df (pd.DataFrame): Bölünecek DataFrame
            source_dir (str): Kaynak görüntü dizini
            output_dir (str): Çıktı dizini
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
        """
        # Dizin yapısını oluştur
        self.create_directory_structure(output_dir)
        
        # Veriyi böl
        train_df, val_df, test_df = self.split_dataframe(df)
        
        # Görüntüleri kopyala
        self.copy_images_to_splits(train_df, source_dir, output_dir, 'train')
        self.copy_images_to_splits(val_df, source_dir, output_dir, 'val')
        self.copy_images_to_splits(test_df, source_dir, output_dir, 'test')
        
        # DataFrame'leri kaydet
        train_df.to_csv(Path(output_dir) / 'train_metadata.csv', index=False)
        val_df.to_csv(Path(output_dir) / 'val_metadata.csv', index=False)
        test_df.to_csv(Path(output_dir) / 'test_metadata.csv', index=False)
        
        print("Veri bölme işlemi tamamlandı!")
        
        return train_df, val_df, test_df
    
    def get_class_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sınıf dağılımını hesaplar
        
        Args:
            df (pd.DataFrame): Analiz edilecek DataFrame
            
        Returns:
            pd.DataFrame: Sınıf dağılımı
        """
        class_dist = df['class_name'].value_counts().reset_index()
        class_dist.columns = ['class_name', 'count']
        class_dist['percentage'] = (class_dist['count'] / len(df)) * 100
        
        return class_dist
    
    def print_split_summary(self, train_df: pd.DataFrame, 
                           val_df: pd.DataFrame, 
                           test_df: pd.DataFrame) -> None:
        """
        Bölme özetini yazdırır
        
        Args:
            train_df (pd.DataFrame): Eğitim seti
            val_df (pd.DataFrame): Doğrulama seti
            test_df (pd.DataFrame): Test seti
        """
        print("\n" + "="*50)
        print("VERİ BÖLME ÖZETİ")
        print("="*50)
        
        print(f"Toplam örnek sayısı: {len(train_df) + len(val_df) + len(test_df)}")
        print(f"Eğitim seti: {len(train_df)} örnek ({len(train_df)/(len(train_df) + len(val_df) + len(test_df))*100:.1f}%)")
        print(f"Doğrulama seti: {len(val_df)} örnek ({len(val_df)/(len(train_df) + len(val_df) + len(test_df))*100:.1f}%)")
        print(f"Test seti: {len(test_df)} örnek ({len(test_df)/(len(train_df) + len(val_df) + len(test_df))*100:.1f}%)")
        
        print("\nSınıf dağılımları:")
        print("-" * 30)
        
        # Her set için sınıf dağılımını göster
        for name, df in [("Eğitim", train_df), ("Doğrulama", val_df), ("Test", test_df)]:
            print(f"\n{name} seti:")
            class_dist = self.get_class_distribution(df)
            for _, row in class_dist.iterrows():
                print(f"  {row['class_name']}: {row['count']} ({row['percentage']:.1f}%)")

#!/bin/bash
# Railway build script - Git LFS dosyalarını çeker

echo "🔧 Building application..."

# Git LFS'i kur (eğer yoksa)
if ! command -v git-lfs &> /dev/null; then
    echo "📦 Installing Git LFS..."
    # Railway'de Git LFS genellikle zaten kurulu, ama kontrol ediyoruz
fi

# Git LFS dosyalarını çek
echo "📥 Fetching Git LFS files..."
git lfs pull || echo "⚠️ Git LFS pull failed, continuing..."

# Dosya varlığını kontrol et
if [ -f "models/mobilenetv2_best.keras" ]; then
    echo "✅ Model file found: $(du -h models/mobilenetv2_best.keras | cut -f1)"
else
    echo "❌ Model file NOT found!"
    echo "📂 Listing models directory:"
    ls -la models/ || echo "models/ directory does not exist"
fi

if [ -f "data/class_names.json" ]; then
    echo "✅ Class names file found"
else
    echo "❌ Class names file NOT found!"
    echo "📂 Listing data directory:"
    ls -la data/ || echo "data/ directory does not exist"
fi

echo "✅ Build script completed"


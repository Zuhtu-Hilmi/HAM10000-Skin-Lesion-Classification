import shutil
import os

# Modelleri ve split indekslerini derli toplu tutmak için bir çıktı klasörü
output_dir = './outputs/saved_models'
os.makedirs(output_dir, exist_ok=True)

# 1. Modelleri (.pth dosyaları) kopyala
kaynak_modeller = './dataset/models/'
if os.path.exists(kaynak_modeller):
    for dosya in os.listdir(kaynak_modeller):
        if dosya.endswith('.pth'):
            shutil.copy(os.path.join(kaynak_modeller, dosya), output_dir)
            print(f"✅ Model Kopyalandı: {dosya}")

# 2. İndeksleri (.npy dosyaları) kopyala
kaynak_split = './dataset/splits/'
if os.path.exists(kaynak_split):
    for dosya in os.listdir(kaynak_split):
        if dosya.endswith('.npy'):
            shutil.copy(os.path.join(kaynak_split, dosya), output_dir)
            print(f"✅ İndeks Kopyalandı: {dosya}")

print(f"\n🎉 BAŞARILI! Tüm dosyalar '{output_dir}' klasörüne kaydedildi.")
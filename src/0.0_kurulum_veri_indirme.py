# =============================================================================
# AŞAMA 0 — KURULUM & VERİ HAZIRLAMA
# Google Colab'da ilk çalıştırılacak hücre.
# Drive'dan zip dosyasını çeker, /content/dataset/ altına açar.
# =============================================================================

# ---------- HÜCRE 0-A: Gerekli kütüphanelerin kurulumu ----------
# Colab'da varsayılan olarak bulunmayan kütüphaneleri yükleyelim
!pip install -q imbalanced-learn scikit-image

# ---------- HÜCRE 0-B: Google Drive bağlantısı ve zip açma ----------
"""
Google Drive'a yüklediğiniz zip dosyasının yolunu aşağıda belirtin.
Varsayılan olarak 'Skin Cancer MNIST HAM10000' adlı zip dosyası
Drive'ın kök dizininde olduğu varsayılmaktadır.
"""

import os
import zipfile
from google.colab import drive

# Drive'ı bağla
drive.mount('/content/drive')

# ─── KONFİGÜRASYON ───
# Varsayılan olarak zip dosyasının ana dizinde 'HAM10000.zip' adıyla bulunduğu varsayılır.
# Kendi bilgisayarınızda veya Colab'da zip dosyasının yolunu buraya girin:
ZIP_PATH = './HAM10000.zip'  
DATASET_DIR = './dataset'

# Hedef klasörü oluştur (yoksa)
os.makedirs(DATASET_DIR, exist_ok=True)

# Zip dosyasını aç

if not os.path.exists(ZIP_PATH):
    print(f"❌ HATA: '{ZIP_PATH}' bulunamadı!")
    print("Lütfen HAM10000 veri setini (zip) proje ana dizinine indirin veya ZIP_PATH değişkenini güncelleyin.")
    print("Veri setini indirmek için: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
else:
    if not os.path.exists(os.path.join(DATASET_DIR, 'HAM10000_metadata.csv')):
        print(f"[INFO] Zip dosyası açılıyor: {ZIP_PATH}")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATASET_DIR)
        print("✅ Zip başarıyla açıldı.")
    else:
        print("✅ Veri seti zaten mevcut, tekrar açmaya gerek yok.")

# ---------- HÜCRE 0-C: Klasör yapısını doğrula ----------
# Beklenen dosyaları kontrol edelim
beklenen_dosyalar = [
    'HAM10000_metadata.csv',
    'hmnist_8_8_RGB.csv',
    'hmnist_28_28_L.csv',
    'HAM10000_images'
]

print("\n[KONTROL] Veri seti klasör yapısı:")
for dosya in beklenen_dosyalar:
    tam_yol = os.path.join(DATASET_DIR, dosya)
    if os.path.exists(tam_yol):
        if os.path.isdir(tam_yol):
            icerik = os.listdir(tam_yol)
            print(f"  ✅ {dosya}/ → {len(icerik)} dosya")
        else:
            boyut = os.path.getsize(tam_yol) / (1024 * 1024)  # MB
            print(f"  ✅ {dosya} → {boyut:.1f} MB")
    else:
        print(f"  ❌ {dosya} BULUNAMADI!")

# ---------- NOT ----------
# Eğer zip dosyasından çıkan klasör adı farklıysa (örneğin alt klasör varsa),
# dosya yollarını aşağıdaki şekilde düzeltmeniz gerekebilir:
#
# Örnek: Eğer zip içinde 'skin-cancer-mnist-ham10000' adlı bir alt klasör varsa:
# import shutil
# kaynak = os.path.join(DATASET_DIR, 'skin-cancer-mnist-ham10000')
# for item in os.listdir(kaynak):
#     shutil.move(os.path.join(kaynak, item), DATASET_DIR)

print("\n[BİLGİ] Aşama 0 tamamlandı. Aşama 1'e geçebilirsiniz.")

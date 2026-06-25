import os
import shutil

hedef_dizin = './dataset'

# Alt klasörleri tarayıp metadata dosyasını bulalım
bulundu = False
for root, dirs, files in os.walk(hedef_dizin):
    if 'HAM10000_metadata.csv' in files:
        bulundu = True
        if root != hedef_dizin:
            print(f"GİZLİ KLASÖR BULUNDU: {root}")
            print("Dosyalar olması gereken ana dizine taşınıyor...")
            # İçindeki tüm dosyaları ana dizine taşı
            for item in os.listdir(root):
                kaynak_yol = os.path.join(root, item)
                hedef_yol = os.path.join(hedef_dizin, item)
                shutil.move(kaynak_yol, hedef_yol)
            print("✅ TAŞIMA BAŞARILI! Tüm dosyalar /content/dataset/ klasörüne alındı.")
        else:
            print("✅ Dosyalar zaten doğru dizinde!")
        break

if not bulundu:
    print("❌ DİKKAT: Zip dosyası hiç açılmamış veya eksik inmiş olabilir.")
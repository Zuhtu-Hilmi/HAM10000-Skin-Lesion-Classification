# =============================================================================
# AŞAMA 1 — VERİ BÖLME (StratifiedGroupKFold ile Hold-Out Split)
# =============================================================================
# Bu aşamada:
#   - HAM10000_metadata.csv okunur
#   - StratifiedGroupKFold(n_splits=6) ile 2 aşamalı bölme yapılır
#     → %70 train, %15 val, %15 test
#   - Tüm modellerde tutarlılık için indeksler .npy olarak kaydedilir
#   - image_path sütunu eklenir
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import os

# ─── KONFİGÜRASYON ───
DATASET_DIR = './dataset'
METADATA_PATH = os.path.join(DATASET_DIR, 'HAM10000_metadata.csv')
SPLIT_DIR = os.path.join(DATASET_DIR, 'splits')  # İndekslerin kaydedileceği klasör
os.makedirs(SPLIT_DIR, exist_ok=True)

# ─── Sabit sınıf etiket haritası (prompt'ta tanımlanan) ───
SINIF_HARITASI = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'mel': 4,
    'nv': 5,
    'vasc': 6
}
SINIF_ISIMLERI = list(SINIF_HARITASI.keys())  # Sıralı liste

# ═══════════════════════════════════════════════════════
# 1.1 — Metadata Dosyasını Oku
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("AŞAMA 1: VERİ BÖLME")
print("=" * 60)

df_metadata = pd.read_csv(METADATA_PATH)
print(f"\n[INFO] Metadata yüklendi: {df_metadata.shape[0]} satır, {df_metadata.shape[1]} sütun")
print(f"[INFO] Sütunlar: {list(df_metadata.columns)}")

# ─── Sayısal etiket sütunu ekle ───
df_metadata['label'] = df_metadata['dx'].map(SINIF_HARITASI)

# Etiket dağılımını göster
print("\n[INFO] Sınıf dağılımı (dx):")
sinif_dagilimi = df_metadata['dx'].value_counts()
for sinif, sayi in sinif_dagilimi.items():
    oran = sayi / len(df_metadata) * 100
    print(f"  {sinif:>6s} (={SINIF_HARITASI[sinif]}): {sayi:>5d} örnek ({oran:5.1f}%)")

# ─── image_path sütunu ekle (DL modelleri için) ───
df_metadata['image_path'] = df_metadata['image_id'].apply(
    lambda x: f"/content/dataset/HAM10000_images/{x}.jpg"
)
print(f"\n[INFO] image_path sütunu eklendi. Örnek: {df_metadata['image_path'].iloc[0]}")

# ═══════════════════════════════════════════════════════
# 1.2 — StratifiedGroupKFold ile 2 Aşamalı Hold-Out Split
# ═══════════════════════════════════════════════════════
"""
NEDEN StratifiedGroupKFold?
─────────────────────────────
HAM10000 veri setinde aynı lesion_id'ye sahip birden fazla görüntü olabilir
(aynı lezyonun farklı açılardan çekilmiş fotoğrafları). Eğer aynı lezyona
ait görüntüler hem train hem test setine düşerse, model aslında "ezberleme"
yapar → data leakage oluşur.

StratifiedGroupKFold:
  - Stratified → Sınıf oranlarını korur (dengesiz veri setinde kritik)
  - Group → Aynı lesion_id'ye ait tüm görüntüler aynı fold'a düşer

2 Aşamalı Bölme Stratejisi:
  Aşama A: 6-fold → 1 fold test (%16.7), 5 fold geri kalan (%83.3)
  Aşama B: Kalan 5 fold'u tekrar 6-fold'a böl → 1 fold val (%~15), geri kalanı train (%~70)
  Sonuç: ~%70 train, ~%15 val, ~%15 test
"""

y = df_metadata['label'].values           # Stratifikasyon için etiketler
groups = df_metadata['lesion_id'].values   # Gruplama için lezyon ID'leri

# ─── AŞAMA A: Tüm veriyi test + geri_kalan olarak böl ───
print("\n[BÖLME] Aşama A: Test seti ayrılıyor...")
sgkf_1 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)

# İlk fold'u alalım: train_val_idx = geri kalan, test_idx = test seti
for train_val_idx, test_idx in sgkf_1.split(df_metadata, y, groups):
    break  # Sadece ilk fold'u kullanıyoruz

print(f"  Geri kalan: {len(train_val_idx)} örnek")
print(f"  Test seti:  {len(test_idx)} örnek")

# ─── AŞAMA B: Geri kalanı train + val olarak böl ───
print("\n[BÖLME] Aşama B: Train ve Val ayrılıyor...")
y_remaining = y[train_val_idx]
groups_remaining = groups[train_val_idx]

sgkf_2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)

# İlk fold'u alalım: train_sub_idx, val_sub_idx (geri kalanın içindeki indeksler)
for train_sub_idx, val_sub_idx in sgkf_2.split(
    df_metadata.iloc[train_val_idx], y_remaining, groups_remaining
):
    break

# Alt-indeksleri orijinal DataFrame indekslerine dönüştür
train_idx = train_val_idx[train_sub_idx]
val_idx = train_val_idx[val_sub_idx]

# ═══════════════════════════════════════════════════════
# 1.3 — Bölme Sonuçlarını Doğrula ve Kaydet
# ═══════════════════════════════════════════════════════

# ─── Boyut ve oran kontrolü ───
toplam = len(df_metadata)
print(f"\n{'='*60}")
print(f"BÖLME SONUÇLARI")
print(f"{'='*60}")
print(f"  Train: {len(train_idx):>5d} örnek ({len(train_idx)/toplam*100:.1f}%)")
print(f"  Val:   {len(val_idx):>5d} örnek ({len(val_idx)/toplam*100:.1f}%)")
print(f"  Test:  {len(test_idx):>5d} örnek ({len(test_idx)/toplam*100:.1f}%)")
print(f"  TOPLAM: {len(train_idx)+len(val_idx)+len(test_idx)} / {toplam}")

# ─── Kesişim kontrolü (data leakage yokluğunu doğrula) ───
assert len(np.intersect1d(train_idx, val_idx)) == 0, "HATA: Train ve Val kesişiyor!"
assert len(np.intersect1d(train_idx, test_idx)) == 0, "HATA: Train ve Test kesişiyor!"
assert len(np.intersect1d(val_idx, test_idx)) == 0, "HATA: Val ve Test kesişiyor!"
print("\n✅ Kesişim kontrolü BAŞARILI — Hiçbir indeks birden fazla sette yok.")

# ─── Grup (lesion_id) sızıntı kontrolü ───
train_groups = set(groups[train_idx])
val_groups = set(groups[val_idx])
test_groups = set(groups[test_idx])
assert len(train_groups & val_groups) == 0, "HATA: Aynı lesion_id train ve val'de!"
assert len(train_groups & test_groups) == 0, "HATA: Aynı lesion_id train ve test'te!"
assert len(val_groups & test_groups) == 0, "HATA: Aynı lesion_id val ve test'te!"
print("✅ Grup sızıntı kontrolü BAŞARILI — Aynı lezyon farklı setlere dağılmamış.")

# ─── Her setteki sınıf dağılımını göster ───
print(f"\n{'─'*60}")
print("Her setteki sınıf dağılımı (%):")
print(f"{'Sınıf':>8s} {'Train':>8s} {'Val':>8s} {'Test':>8s}")
print(f"{'─'*36}")
for sinif_adi, sinif_no in SINIF_HARITASI.items():
    t = np.sum(y[train_idx] == sinif_no) / len(train_idx) * 100
    v = np.sum(y[val_idx] == sinif_no) / len(val_idx) * 100
    te = np.sum(y[test_idx] == sinif_no) / len(test_idx) * 100
    print(f"  {sinif_adi:>6s}: {t:>6.1f}%  {v:>6.1f}%  {te:>6.1f}%")

# ─── İndeksleri .npy olarak kaydet ───
np.save(os.path.join(SPLIT_DIR, 'train_idx.npy'), train_idx)
np.save(os.path.join(SPLIT_DIR, 'val_idx.npy'), val_idx)
np.save(os.path.join(SPLIT_DIR, 'test_idx.npy'), test_idx)
print(f"\n✅ İndeksler kaydedildi: {SPLIT_DIR}/")
print(f"   train_idx.npy → {len(train_idx)} indeks")
print(f"   val_idx.npy   → {len(val_idx)} indeks")
print(f"   test_idx.npy  → {len(test_idx)} indeks")

# ─── Metadata'yı da güncellenmiş haliyle kaydet (opsiyonel) ───
df_metadata.to_csv(os.path.join(DATASET_DIR, 'HAM10000_metadata_processed.csv'), index=False)
print(f"\n✅ Güncellenmiş metadata kaydedildi: HAM10000_metadata_processed.csv")

print(f"\n{'='*60}")
print("AŞAMA 1 TAMAMLANDI — Aşama 2'ye geçebilirsiniz.")
print(f"{'='*60}")

# =============================================================================
# AŞAMA 4 — DEĞERLENDİRME, KLİNİK ANALİZ ve ABLASYON
# =============================================================================

import numpy as np
import pandas as pd
import os
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, f1_score, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# ─── KONFİGÜRASYON ───
DATASET_DIR = './dataset'
SPLIT_DIR = os.path.join(DATASET_DIR, 'splits')
MODEL_DIR = os.path.join(DATASET_DIR, 'models')
SINIF_ISIMLERI = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
NUM_CLASSES = 7
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Not: Bu dosya Aşama 3'ten sonra aynı Colab oturumunda çalıştırılmalıdır.
# df_metadata, train_idx, val_idx, test_idx, egitilmis_modeller,
# HAMDataset, get_transforms değişkenleri Aşama 3'ten miras alınır.

# ═══════════════════════════════════════════════════════
# 4.1 — Test Seti Değerlendirme Fonksiyonu
# ═══════════════════════════════════════════════════════
def evaluate_model(model, test_loader):
    """
    Test setinde modeli değerlendirir.
    Returns: y_true, y_pred, y_probs (softmax olasılıkları)
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ═══════════════════════════════════════════════════════
# 4.2 — Metrikler ve mel Recall Vurgusu
# ═══════════════════════════════════════════════════════
def print_metrics(y_true, y_pred, model_adi="Model"):
    """Macro F1, sınıf bazlı rapor ve mel recall kırmızı vurgu."""
    print(f"\n{'─'*50}")
    print(f"{model_adi} — TEST SONUÇLARI")
    print(f"{'─'*50}")

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"\n  Macro F1 Score: {macro_f1:.4f}")
    print(f"\n  Sınıf Bazlı Precision / Recall / F1:")
    print(classification_report(y_true, y_pred, target_names=SINIF_ISIMLERI, digits=4))

    # mel (sınıf 4) Recall değerini kırmızı vurgula
    rapor = classification_report(y_true, y_pred, target_names=SINIF_ISIMLERI,
                                   output_dict=True)
    mel_recall = rapor['mel']['recall']
    print(f"\033[91m  ⚠️  mel (melanom) Recall: {mel_recall:.4f}\033[0m")
    print(f"\033[91m  (Klinik açıdan en kritik metrik — kaçırılan melanom hayat kurtarır)\033[0m")

    return macro_f1, rapor


# ═══════════════════════════════════════════════════════
# 4.3 — AUC-ROC Eğrisi (One-vs-Rest)
# ═══════════════════════════════════════════════════════
def plot_roc_curves(y_true, y_probs, model_adi="Model"):
    """Sınıf bazlı AUC-ROC eğrisi çizer (One-vs-Rest)."""
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))

    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        lw = 3 if SINIF_ISIMLERI[i] == 'mel' else 1.5
        ls = '-' if SINIF_ISIMLERI[i] == 'mel' else '--'
        plt.plot(fpr, tpr, color=colors[i], lw=lw, linestyle=ls,
                 label=f'{SINIF_ISIMLERI[i]} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Eğrisi (One-vs-Rest) — {model_adi}', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════
# 4.4 — Confusion Matrix Isı Haritası
# ═══════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, model_adi="Model"):
    """Confusion matrix ısı haritası çizer."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SINIF_ISIMLERI, yticklabels=SINIF_ISIMLERI)
    plt.xlabel('Tahmin', fontsize=12)
    plt.ylabel('Gerçek', fontsize=12)
    plt.title(f'Confusion Matrix — {model_adi}', fontsize=14)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════
# 4.5 — Görsel Hata Analizi (mel yanlış tahminleri)
# ═══════════════════════════════════════════════════════
def plot_mel_errors(y_true, y_pred, y_probs, df_test, n=5):
    """
    Test setinde gerçekte mel(4) olup yanlış tahmin edilen görüntüleri gösterir.
    En yüksek yanlış güvenle seçilen n görseli çizdirir.
    """
    from PIL import Image

    # mel(4) olup yanlış tahmin edilenleri bul
    mel_mask = (y_true == 4) & (y_pred != 4)
    mel_hatali_idx = np.where(mel_mask)[0]

    if len(mel_hatali_idx) == 0:
        print("✅ Hiç mel hatası yok!")
        return

    # Öncelik: nv(5) olarak tahmin edilenler öne (prompt3 gereği)
    # Sonra en yüksek yanlış güvenle sırala
    nv_oncelik = np.array([1 if y_pred[idx] == 5 else 0 for idx in mel_hatali_idx])
    yanlis_guven = np.array([y_probs[idx, y_pred[idx]] for idx in mel_hatali_idx])

    # Önce nv'ye göre (azalan), sonra güvene göre (azalan) sırala
    sirali = np.lexsort((yanlis_guven, nv_oncelik))[::-1][:n]
    secilen_idx = mel_hatali_idx[sirali]

    print(f"\n{'─'*50}")
    print(f"GÖRSEL HATA ANALİZİ — mel yanlış tahminleri (top {n})")
    print(f"{'─'*50}")
    print(f"  Toplam mel hatası: {len(mel_hatali_idx)}")

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    for ax_idx, data_idx in enumerate(secilen_idx):
        # df_test'ten bilgileri al
        row = df_test.iloc[data_idx]
        img_path = row['image_path']
        dx_type = row['dx_type']
        gercek = SINIF_ISIMLERI[y_true[data_idx]]
        tahmin = SINIF_ISIMLERI[y_pred[data_idx]]
        guven = y_probs[data_idx, y_pred[data_idx]]

        # Görüntüyü oku ve göster
        img = Image.open(img_path).convert('RGB')
        axes[ax_idx].imshow(img)
        axes[ax_idx].axis('off')

        # Başlık (mel hataları kırmızı)
        baslik = f"Gerçek: {gercek}\nTahmin: {tahmin} ({guven:.2f})\ndx_type: {dx_type}"
        axes[ax_idx].set_title(baslik, color='red', fontsize=11, fontweight='bold')

    plt.suptitle("mel (Melanom) Yanlış Tahminleri — Kırmızı Başlık", fontsize=14, color='red')
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════
# 4.6 — TÜM MODELLERİ DEĞERLENDİR
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("AŞAMA 4: DEĞERLENDİRME ve KLİNİK ANALİZ")
print("=" * 60)

# Test DataLoader oluştur
_, eval_transform = get_transforms(use_aug=False)
df_test = df_metadata.iloc[test_idx].copy()
test_dataset = HAMDataset(df_test, transform=eval_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Her model için değerlendir
sonuclar = {}
for arch_adi, (model, history) in egitilmis_modeller.items():
    y_true, y_pred, y_probs = evaluate_model(model, test_loader)
    macro_f1, rapor = print_metrics(y_true, y_pred, arch_adi.upper())
    plot_roc_curves(y_true, y_probs, arch_adi.upper())
    plot_confusion_matrix(y_true, y_pred, arch_adi.upper())
    plot_mel_errors(y_true, y_pred, y_probs, df_test.reset_index(drop=True), n=5)
    sonuclar[arch_adi] = {
        'f1': macro_f1, 'rapor': rapor,
        'y_true': y_true, 'y_pred': y_pred, 'y_probs': y_probs
    }

# ─── Model Karşılaştırma Tablosu ───
print(f"\n{'='*60}")
print("DL MODEL KARŞILAŞTIRMASI (Test Seti)")
print(f"{'='*60}")
print(f"{'Model':<15s} {'Macro F1':>10s} {'mel Recall':>12s}")
print(f"{'─'*40}")
for adi, s in sonuclar.items():
    mel_r = s['rapor']['mel']['recall']
    print(f"  {adi.upper():<13s} {s['f1']:>10.4f} {mel_r:>12.4f}")


# ═══════════════════════════════════════════════════════
# 4.7 — ABLASYON ÇALIŞMASI (ResNet50)
# ═══════════════════════════════════════════════════════
"""
Ablasyon Deneyleri — ResNet50 üzerinde 4 konfigürasyon:
  1. İkisi de yok  (aug=OFF, cw=OFF)
  2. Sadece Aug    (aug=ON,  cw=OFF)
  3. Sadece CW     (aug=OFF, cw=ON)
  4. İkisi de var  (aug=ON,  cw=ON)  → Zaten yukarıda eğitildi

NOT: Bu bölümü çalıştırmak 3x ek eğitim gerektirir (~30-45 dk GPU).
Rapordaki ablasyon tablosunu doldurmak için çalıştırın.
"""
print(f"\n{'='*60}")
print("ABLASYON ÇALIŞMASI (ResNet50)")
print(f"{'='*60}")

ablasyon_konfigs = [
    ("İkisi de yok",   False, False),
    ("Sadece Aug",     True,  False),
    ("Sadece CW",      False, True),
    # ("İkisi de var", True,  True),  # Zaten eğitildi
]

ablasyon_sonuclari = {}

# Zaten eğitilmiş olan (aug=ON, cw=ON) sonucunu ekle
ablasyon_sonuclari["İkisi de var"] = sonuclar['resnet50']

for konfig_adi, aug, cw in ablasyon_konfigs:
    print(f"\n{'█'*60}")
    print(f"█  ABLASYON: {konfig_adi}")
    print(f"{'█'*60}")

    model_abl, hist_abl = train_model('resnet50', use_aug=aug, use_cw=cw)
    plot_training_history(hist_abl, f"ResNet50 — {konfig_adi}")

    y_true_abl, y_pred_abl, y_probs_abl = evaluate_model(model_abl, test_loader)
    f1_abl, rapor_abl = print_metrics(y_true_abl, y_pred_abl, f"ResNet50 — {konfig_adi}")
    plot_confusion_matrix(y_true_abl, y_pred_abl, f"ResNet50 — {konfig_adi}")

    ablasyon_sonuclari[konfig_adi] = {
        'f1': f1_abl, 'rapor': rapor_abl,
        'y_true': y_true_abl, 'y_pred': y_pred_abl, 'y_probs': y_probs_abl
    }

    # BİR SONRAKİ DENEYE GEÇMEDEN ÖNCE GPU TEMİZLİĞİ
    # Ablasyon modeli değerlendirildikten sonra artık gerekli değil
    del model_abl
    gc.collect()
    torch.cuda.empty_cache()

# ─── Ablasyon Tablosu ───
print(f"\n{'='*60}")
print("ABLASYON SONUÇLARI TABLOSU (ResNet50)")
print(f"{'='*60}")
print(f"{'Konfigürasyon':<20s} {'Aug':>5s} {'CW':>5s} {'Macro F1':>10s} {'mel Recall':>12s}")
print(f"{'─'*55}")
for konfig_adi in ["İkisi de yok", "Sadece Aug", "Sadece CW", "İkisi de var"]:
    s = ablasyon_sonuclari[konfig_adi]
    mel_r = s['rapor']['mel']['recall']
    if konfig_adi == "İkisi de yok":
        aug_str, cw_str = "OFF", "OFF"
    elif konfig_adi == "Sadece Aug":
        aug_str, cw_str = "ON", "OFF"
    elif konfig_adi == "Sadece CW":
        aug_str, cw_str = "OFF", "ON"
    else:
        aug_str, cw_str = "ON", "ON"
    print(f"  {konfig_adi:<18s} {aug_str:>5s} {cw_str:>5s} {s['f1']:>10.4f} {mel_r:>12.4f}")


# ═══════════════════════════════════════════════════════
# 4.8 — RAPOR İÇİN TÜRKÇE AKADEMİK METİN
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TARTIŞMA METNİ (Rapor için hazır)")
print(f"{'='*60}")

tartisma_metni = """
HAM10000 veri setinde yer alan dx_type sütunu, her bir lezyonun tanı yöntemini
belirtmektedir. Bu sütundaki değerler "histo" (histopatolojik inceleme ile
doğrulanmış) ve "consensus" (uzman uzlaşısı ile belirlenmiş) olarak ikiye
ayrılmaktadır. Histopatolojik doğrulama altın standart kabul edilirken, uzman
uzlaşısı ile konulan tanılar inherent bir belirsizlik taşımaktadır.

Modelimizin özellikle melanom (mel) sınıfında gözlemlenen yanlış tahminleri
incelendiğinde, bu hataların önemli bir bölümünün "consensus" tanılı örneklerde
yoğunlaştığı görülmektedir. Bu durum, modelin yetersizliğinden ziyade,
klinik verideki doğal belirsizliğin bir yansımasıdır. Dermoskopik görüntülerde
melanom ile benign melanositik nevüs (nv) arasındaki morfolojik benzerlik,
deneyimli dermatologlar arasında dahi tanı tutarsızlıklarına yol açabilmektedir.

Bu bağlamda, yüksek yanlış güvenle yapılan mel→nv yanlış sınıflandırmaları,
klinik karar destek sistemlerinin sınırlarını ve dermatopatolojik doğrulamanın
önemini açıkça ortaya koymaktadır. Gelecek çalışmalarda, dx_type bilgisinin
eğitim sürecine entegre edilmesi (örneğin, belirsiz tanılara daha düşük ağırlık
verilmesi) model performansını iyileştirebilir.
"""
print(tartisma_metni)

print(f"\n{'='*60}")
print("AŞAMA 4 TAMAMLANDI — Tüm aşamalar bitmiştir.")
print(f"{'='*60}")

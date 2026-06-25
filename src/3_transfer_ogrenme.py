# =============================================================================
# AŞAMA 3 — TRANSFER LEARNING (PyTorch)
# =============================================================================
# Bu aşamada:
#   - AlexNet, VGG16, ResNet50 transfer learning modelleri
#   - Orijinal .jpg görüntüler PyTorch Dataset ile okunur (CSV KULLANILMAZ)
#   - Augmentation sadece train'e, Normalize her sete
#   - class_weight → CrossEntropyLoss(weight=...)
#   - Manuel eğitim döngüsü: EarlyStopping + ReduceLROnPlateau + Checkpoint
#   - Ablasyon için parametrik: train_model(use_aug=True, use_cw=True)
# =============================================================================

import numpy as np
import pandas as pd
import os
import time
import copy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.utils.class_weight import compute_class_weight

# ─── GPU kontrolü ───
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Cihaz: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Bellek: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─── KONFİGÜRASYON ───
DATASET_DIR = './dataset'
SPLIT_DIR = os.path.join(DATASET_DIR, 'splits')
MODEL_DIR = os.path.join(DATASET_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

SINIF_ISIMLERI = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
NUM_CLASSES = 7

# Eğitim hiperparametreleri
BATCH_SIZE = 32
MAX_EPOCHS = 50
LR = 1e-4
PATIENCE_ES = 10       # EarlyStopping: kaç epoch iyileşme olmazsa dur
PATIENCE_LR = 4       # ReduceLROnPlateau: kaç epoch iyileşme olmazsa lr düşür
LR_FACTOR = 0.5       # Öğrenme oranı düşürme faktörü

# ImageNet normalizasyon değerleri
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ═══════════════════════════════════════════════════════
# 3.0 — Veri Yükleme
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("AŞAMA 3: TRANSFER LEARNING (PyTorch)")
print("=" * 60)

# İndeksleri yükle
train_idx = np.load(os.path.join(SPLIT_DIR, 'train_idx.npy'))
val_idx = np.load(os.path.join(SPLIT_DIR, 'val_idx.npy'))
test_idx = np.load(os.path.join(SPLIT_DIR, 'test_idx.npy'))

# Metadata yükle
df_metadata = pd.read_csv(os.path.join(DATASET_DIR, 'HAM10000_metadata.csv'))

# Sınıf etiket haritası
SINIF_HARITASI = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
df_metadata['label'] = df_metadata['dx'].map(SINIF_HARITASI)
df_metadata['image_path'] = df_metadata['image_id'].apply(
    lambda x: f"/content/dataset/HAM10000_images/{x}.jpg"
)

print(f"[INFO] Metadata: {len(df_metadata)} örnek")
print(f"[INFO] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")


# ═══════════════════════════════════════════════════════
# 3.1 — PyTorch Dataset Sınıfı
# ═══════════════════════════════════════════════════════
class HAMDataset(Dataset):
    """
    HAM10000 veri seti için özel PyTorch Dataset sınıfı.
    Orijinal .jpg dosyalarını diskten okur.

    Args:
        df: Metadata DataFrame (image_path ve label sütunları olmalı)
        transform: Uygulanacak dönüşümler (augmentation + normalize)
    """
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Görüntüyü oku
        img_path = self.df.loc[idx, 'image_path']
        image = Image.open(img_path).convert('RGB')

        # Etiketi al
        label = self.df.loc[idx, 'label']

        # Dönüşümleri uygula
        if self.transform:
            image = self.transform(image)

        return image, label


# ═══════════════════════════════════════════════════════
# 3.2 — Dönüşümler (Transforms)
# ═══════════════════════════════════════════════════════
def get_transforms(use_aug=True):
    """
    Eğitim ve değerlendirme dönüşümlerini döndürür.

    Args:
        use_aug: True ise eğitim setine augmentation uygulanır

    Returns:
        train_transform, eval_transform
    """
    # ─── Eğitim seti dönüşümleri ───
    if use_aug:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),      # Yatay çevirme
            transforms.RandomVerticalFlip(),         # Dikey çevirme
            transforms.RandomRotation(30),           # ±30° döndürme
            transforms.ColorJitter(brightness=0.2),  # Parlaklık değişimi
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # Augmentation kapalı → sadece resize + normalize
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    # ─── Val/Test dönüşümleri (augmentation YOK) ───
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, eval_transform


# ═══════════════════════════════════════════════════════
# 3.3 — Sınıf Ağırlıkları (Class Weights)
# ═══════════════════════════════════════════════════════
def compute_class_weights(y_train):
    """
    Dengesiz sınıf dağılımını telafi etmek için ağırlıkları hesaplar.

    compute_class_weight('balanced') formülü:
        w_c = n_samples / (n_classes * n_samples_c)

    Azınlık sınıflarına (df, vasc) yüksek, çoğunluk sınıflarına (nv) düşük ağırlık verir.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)

    print("\n[INFO] Sınıf ağırlıkları (balanced):")
    for sinif_no, agirlik in zip(classes, weights):
        print(f"  {SINIF_ISIMLERI[sinif_no]:>6s} ({sinif_no}): {agirlik:.4f}")

    return torch.FloatTensor(weights).to(device)


# ═══════════════════════════════════════════════════════
# 3.4 — Model Oluşturma (build_model)
# ═══════════════════════════════════════════════════════
def build_model(arch):
    """
    Transfer learning modeli oluşturur.

    Args:
        arch: 'alexnet', 'vgg16', veya 'resnet50'

    Returns:
        model: Belirtilen mimariye göre yapılandırılmış model

    Dondurma stratejisi:
        - AlexNet: Tüm conv katmanları dondur, sadece classifier eğit
        - VGG16: features[0:24] dondur, features[24:] + classifier eğit
        - ResNet50: layer1+layer2 dondur, layer3+layer4+fc eğit
    """
    print(f"\n[MODEL] {arch.upper()} oluşturuluyor...")

    if arch == 'alexnet':
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

        # Tüm conv (features) katmanlarını dondur
        for param in model.features.parameters():
            param.requires_grad = False

        # Classifier'ı yeniden tasarla (son katman 7 sınıfa)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, NUM_CLASSES
        )

        dondurulan = "features (tüm conv)"
        egitilen = "classifier"

    elif arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # features[0:24] dondur (ilk 24 katman)
        for i, layer in enumerate(model.features):
            if i < 24:
                for param in layer.parameters():
                    param.requires_grad = False

        # Classifier'ı yeniden tasarla
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, NUM_CLASSES
        )

        dondurulan = "features[0:24]"
        egitilen = "features[24:] + classifier"

    elif arch == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # layer1 ve layer2 dondur
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.bn1.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False

        # FC katmanını yeniden tasarla
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

        dondurulan = "conv1 + bn1 + layer1 + layer2"
        egitilen = "layer3 + layer4 + fc"

    else:
        raise ValueError(f"Desteklenmeyen mimari: {arch}")

    # Eğitilebilir parametre sayısını hesapla
    toplam_param = sum(p.numel() for p in model.parameters())
    egitilen_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dondurulan_param = toplam_param - egitilen_param

    print(f"  Dondurulan: {dondurulan}")
    print(f"  Eğitilen:   {egitilen}")
    print(f"  Toplam parametre:     {toplam_param:>12,}")
    print(f"  Eğitilebilir:         {egitilen_param:>12,} ({egitilen_param/toplam_param*100:.1f}%)")
    print(f"  Dondurulmuş:          {dondurulan_param:>12,} ({dondurulan_param/toplam_param*100:.1f}%)")

    model = model.to(device)
    return model


# ═══════════════════════════════════════════════════════
# 3.5 — Eğitim Döngüsü (Manuel — Parametrik Ablasyon)
# ═══════════════════════════════════════════════════════
def train_model(model_or_arch, use_aug=True, use_cw=True):
    """
    Transfer learning modelini eğitir.

    ABLASYON TASARIMI:
    ─────────────────
    Bu fonksiyon parametrik olarak tasarlanmıştır:
        use_aug=True  → Eğitim setine veri artırma (augmentation) uygula
        use_cw=True   → CrossEntropyLoss'a sınıf ağırlığı (class weight) ver

    Ablasyon Deneyleri (ResNet50 üzerinde):
        1. train_model('resnet50', use_aug=False, use_cw=False) → İkisi de yok
        2. train_model('resnet50', use_aug=True,  use_cw=False) → Sadece Aug
        3. train_model('resnet50', use_aug=False, use_cw=True)  → Sadece CW
        4. train_model('resnet50', use_aug=True,  use_cw=True)  → İkisi de var

    Manuel Callback'ler:
        - EarlyStopping: val_loss 5 epoch iyileşmezse dur
        - ReduceLROnPlateau: val_loss 3 epoch iyileşmezse lr *= 0.5
        - Checkpoint: val_accuracy en yüksek olduğunda ağırlıkları kaydet

    Args:
        model_or_arch: Mimari adı ('alexnet', 'vgg16', 'resnet50') veya
                       bu isimlerden biri. Her çağrıda model sıfırdan oluşturulur.
        use_aug: Veri artırma kullan (bool)
        use_cw: Sınıf ağırlığı kullan (bool)

    Returns:
        model: En iyi ağırlıklarla yüklenmiş model
        history: Eğitim geçmişi dict'i
    """
    # Mimari adını belirle (string veya nn.Module kabul eder)
    if isinstance(model_or_arch, str):
        arch = model_or_arch
    else:
        # Eğer model nesnesi verilmişse, sınıf adından mimariyi tahmin et
        class_name = model_or_arch.__class__.__name__.lower()
        if 'alexnet' in class_name:
            arch = 'alexnet'
        elif 'vgg' in class_name:
            arch = 'vgg16'
        elif 'resnet' in class_name:
            arch = 'resnet50'
        else:
            raise ValueError(f"Tanınmayan model sınıfı: {class_name}")
    # NOT: Her çağrıda model sıfırdan (pretrained) oluşturulur,
    # böylece önceki eğitimin ağırlıkları ablasyon sonuçlarını etkilemez.
    print(f"\n{'='*60}")
    deney_adi = f"{arch.upper()} | Aug={'ON' if use_aug else 'OFF'} | CW={'ON' if use_cw else 'OFF'}"
    print(f"EĞİTİM BAŞLIYOR: {deney_adi}")
    print(f"{'='*60}")

    # ─── 1. Dönüşümleri hazırla ───
    train_transform, eval_transform = get_transforms(use_aug=use_aug)

    # ─── 2. Dataset ve DataLoader oluştur ───
    df_train = df_metadata.iloc[train_idx].copy()
    df_val = df_metadata.iloc[val_idx].copy()

    train_dataset = HAMDataset(df_train, transform=train_transform)
    val_dataset = HAMDataset(df_val, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print(f"\n[DATA] Train: {len(train_dataset)} örnek, {len(train_loader)} batch")
    print(f"[DATA] Val: {len(val_dataset)} örnek, {len(val_loader)} batch")

    # ─── 3. Model oluştur ───
    model = build_model(arch)

    # ─── 4. Loss fonksiyonu (sınıf ağırlıklı / ağırlıksız) ───
    if use_cw:
        y_train = df_train['label'].values
        class_weights = compute_class_weights(y_train)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("[LOSS] CrossEntropyLoss(weight=balanced)")
    else:
        criterion = nn.CrossEntropyLoss()
        print("[LOSS] CrossEntropyLoss(weight=None)")

    # ─── 5. Optimizer ───
    # Sadece requires_grad=True olan parametreleri optimize et
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )
    print(f"[OPTIM] Adam(lr={LR})")

    # ─── 6. Eğitim geçmişi ve callback değişkenleri ───
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0         # Checkpoint için en iyi val accuracy
    best_val_loss = float('inf')  # EarlyStopping/LR için en iyi val loss
    best_model_wts = None      # En iyi ağırlıklar

    epochs_no_improve = 0      # EarlyStopping sayacı
    lr_no_improve = 0          # ReduceLR sayacı
    current_lr = LR

    checkpoint_path = os.path.join(
        MODEL_DIR,
        f"{arch}_aug{'ON' if use_aug else 'OFF'}_cw{'ON' if use_cw else 'OFF'}_best.pth"
    )

    # ─── 7. EĞİTİM DÖNGÜSÜ ───
    print(f"\n{'─'*60}")
    print(f"{'Epoch':>5s} | {'Train Loss':>10s} {'Train Acc':>10s} | {'Val Loss':>10s} {'Val Acc':>10s} | {'LR':>10s} | {'Durum'}")
    print(f"{'─'*60}")

    toplam_baslangic = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_baslangic = time.time()

        # ── TRAIN FAZI ──
        model.train()
        train_loss_toplam = 0.0
        train_dogru = 0
        train_toplam = 0

        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            # İleri geçiş
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)

            # Geri yayılım
            loss.backward()
            optimizer.step()

            # İstatistikler
            train_loss_toplam += loss.item() * batch_images.size(0)
            _, preds = torch.max(outputs, 1)
            train_dogru += (preds == batch_labels).sum().item()
            train_toplam += batch_images.size(0)

        train_loss = train_loss_toplam / train_toplam
        train_acc = train_dogru / train_toplam

        # ── VAL FAZI ──
        model.eval()
        val_loss_toplam = 0.0
        val_dogru = 0
        val_toplam = 0

        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)

                val_loss_toplam += loss.item() * batch_images.size(0)
                _, preds = torch.max(outputs, 1)
                val_dogru += (preds == batch_labels).sum().item()
                val_toplam += batch_images.size(0)

        val_loss = val_loss_toplam / val_toplam
        val_acc = val_dogru / val_toplam

        # Geçmişe kaydet
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        epoch_sure = time.time() - epoch_baslangic

        # ── MANUEL CALLBACK'LER ──
        durum_mesajlari = []

        # Checkpoint: Val accuracy en yüksek mi?
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, checkpoint_path)
            durum_mesajlari.append(f"✅ Checkpoint (acc={val_acc:.4f})")

        # EarlyStopping ve ReduceLR kontrolü (val_loss bazlı)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            lr_no_improve = 0
        else:
            epochs_no_improve += 1
            lr_no_improve += 1

        # ReduceLROnPlateau: 3 epoch iyileşme yoksa
        if lr_no_improve >= PATIENCE_LR:
            old_lr = current_lr
            current_lr *= LR_FACTOR
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            lr_no_improve = 0  # Sayacı sıfırla
            durum_mesajlari.append(f"📉 LR: {old_lr:.2e}→{current_lr:.2e}")

        # Epoch bilgisini yazdır
        durum_str = " | ".join(durum_mesajlari) if durum_mesajlari else f"({epoch_sure:.0f}s)"
        print(
            f"  {epoch:>3d}   | {train_loss:>10.4f} {train_acc:>10.4f} | "
            f"{val_loss:>10.4f} {val_acc:>10.4f} | {current_lr:>10.2e} | {durum_str}"
        )

        # EarlyStopping: 5 epoch iyileşme yoksa VE en az 30 epoch tamamlandıysa
        if epochs_no_improve >= PATIENCE_ES and epoch >= 30:
            print(f"\n  ⛔ EarlyStopping: En az 30 epoch tamamlandı ve {PATIENCE_ES} epoch boyunca val_loss iyileşmedi.")
            break

    toplam_sure = time.time() - toplam_baslangic

    # ─── 8. En iyi ağırlıkları yükle ───
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        print(f"\n  ✅ En iyi ağırlıklar yüklendi (val_acc={best_val_acc:.4f})")

    print(f"  ⏱️ Toplam eğitim süresi: {toplam_sure/60:.1f} dakika")
    print(f"  📁 Checkpoint: {checkpoint_path}")

    return model, history


# ═══════════════════════════════════════════════════════
# 3.6 — Eğitim Geçmişi Grafikleri
# ═══════════════════════════════════════════════════════
import matplotlib.pyplot as plt

def plot_training_history(history, title=""):
    """Eğitim ve doğrulama loss/accuracy grafiklerini çizer."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss grafiği
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Loss Eğrisi — {title}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy grafiği
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Accuracy Eğrisi — {title}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════
# 3.7 — 3 MİMARİYİ EĞİT (AlexNet, VGG16, ResNet50)
# ═══════════════════════════════════════════════════════
"""
ÖNEMLİ: Aşağıdaki hücreleri sırayla çalıştırın.
Her model ~5-15 dakika sürebilir (GPU'ya bağlı).

Ablasyon çalışması için:
  İlk olarak 3 mimariyi varsayılan (aug=ON, cw=ON) ile eğitin.
  Ardından Aşama 4'te ResNet50 ablasyon deneylerini çalıştırın.
"""

import gc  # GPU bellek temizliği için

# ──────────── AlexNet ────────────
print("\n" + "█" * 60)
print("█  ALEXNET EĞİTİMİ")
print("█" * 60)
alexnet_model, alexnet_history = train_model('alexnet', use_aug=True, use_cw=True)
plot_training_history(alexnet_history, "AlexNet")
# GPU TEMİZLİĞİ — Bir sonraki model için cache'lenmiş CUDA belleğini serbest bırak
gc.collect()
torch.cuda.empty_cache()

# ──────────── VGG16 ────────────
print("\n" + "█" * 60)
print("█  VGG16 EĞİTİMİ")
print("█" * 60)
vgg16_model, vgg16_history = train_model('vgg16', use_aug=True, use_cw=True)
plot_training_history(vgg16_history, "VGG16")
# GPU TEMİZLİĞİ
gc.collect()
torch.cuda.empty_cache()

# ──────────── ResNet50 ────────────
print("\n" + "█" * 60)
print("█  RESNET50 EĞİTİMİ")
print("█" * 60)
resnet50_model, resnet50_history = train_model('resnet50', use_aug=True, use_cw=True)
plot_training_history(resnet50_history, "ResNet50")
# GPU TEMİZLİĞİ
gc.collect()
torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════
# 3.8 — Eğitilmiş Modelleri Sözlükte Sakla
# ═══════════════════════════════════════════════════════
# (Aşama 4'te kullanılacak)
egitilmis_modeller = {
    'alexnet': (alexnet_model, alexnet_history),
    'vgg16': (vgg16_model, vgg16_history),
    'resnet50': (resnet50_model, resnet50_history)
}

print(f"\n{'='*60}")
print("AŞAMA 3 TAMAMLANDI — Aşama 4'e geçebilirsiniz.")
print(f"{'='*60}")

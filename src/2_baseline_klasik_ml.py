# =============================================================================
# AŞAMA 2 — BASELINE ML MODELLERİ (k-NN ve SVM)
# =============================================================================

import numpy as np
import pandas as pd
import os
import time
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from skimage.feature import hog
from imblearn.over_sampling import SMOTE

DATASET_DIR = './dataset'
SPLIT_DIR = os.path.join(DATASET_DIR, 'splits')
SINIF_ISIMLERI = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

print("=" * 60)
print("AŞAMA 2: BASELINE ML MODELLERİ")
print("=" * 60)

train_idx = np.load(os.path.join(SPLIT_DIR, 'train_idx.npy'))
val_idx = np.load(os.path.join(SPLIT_DIR, 'val_idx.npy'))
test_idx = np.load(os.path.join(SPLIT_DIR, 'test_idx.npy'))

print(f"\n[INFO] İndeksler yüklendi:")
print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# DOĞRU ETİKETLERİ METADATA'DAN ALMA (HAYAT KURTARAN DÜZELTME)
df_meta_dogru = pd.read_csv(os.path.join(DATASET_DIR, 'HAM10000_metadata.csv'))
SINIF_HARITASI = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
y_dogru_etiketler = df_meta_dogru['dx'].map(SINIF_HARITASI).values

# ---------------------------------------------------------
# 2.1 — k-NN MODELİ
# ---------------------------------------------------------
print(f"\n{'─'*60}")
print("2.1 — k-NN MODELİ")
print(f"{'─'*60}")

print("\n[1/5] hmnist_8_8_RGB.csv okunuyor...")
df_knn = pd.read_csv(os.path.join(DATASET_DIR, 'hmnist_8_8_RGB.csv'))

# Piksel verileri CSV'den, etiketler Metadata'dan!
X_knn_all = df_knn.iloc[:, :-1].values.astype(np.float32)
y_knn_all = y_dogru_etiketler

X_knn_train = X_knn_all[train_idx]
y_knn_train = y_knn_all[train_idx]
X_knn_val = X_knn_all[val_idx]
y_knn_val = y_knn_all[val_idx]
X_knn_test = X_knn_all[test_idx]
y_knn_test = y_knn_all[test_idx]

print(f"\n[3/5] PCA(n_components=32) uygulanıyor...")
pca_knn = PCA(n_components=32, random_state=42)
X_knn_train_pca = pca_knn.fit_transform(X_knn_train)
X_knn_val_pca = pca_knn.transform(X_knn_val)
X_knn_test_pca = pca_knn.transform(X_knn_test)

print("\n[4/5] SMOTE uygulanıyor (sadece eğitim verisi)...")
smote = SMOTE(random_state=42)
X_knn_train_resampled, y_knn_train_resampled = smote.fit_resample(X_knn_train_pca, y_knn_train)

print("\n[5/5] KNeighborsClassifier(n_neighbors=5) eğitiliyor...")
baslangic = time.time()
knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_model.fit(X_knn_train_resampled, y_knn_train_resampled)
sure = time.time() - baslangic
print(f"  Eğitim süresi: {sure:.1f} saniye")

print(f"\n{'─'*40}")
print("k-NN SONUÇLARI (Test Seti)")
print(f"{'─'*40}")
y_knn_pred = knn_model.predict(X_knn_test_pca)
knn_f1_macro = f1_score(y_knn_test, y_knn_pred, average='macro')
print(f"\n  Macro F1 Score: {knn_f1_macro:.4f}")
print(classification_report(y_knn_test, y_knn_pred, target_names=SINIF_ISIMLERI, digits=4))

y_knn_val_pred = knn_model.predict(X_knn_val_pca)
knn_val_f1 = f1_score(y_knn_val, y_knn_val_pred, average='macro')

# ---------------------------------------------------------
# 2.2 — SVM MODELİ (HOG + PCA + SVC)
# ---------------------------------------------------------
print(f"\n{'─'*60}")
print("2.2 — SVM MODELİ (HOG + PCA)")
print(f"{'─'*60}")

print("\n[1/6] hmnist_28_28_L.csv okunuyor...")
df_svm = pd.read_csv(os.path.join(DATASET_DIR, 'hmnist_28_28_L.csv'))

# Piksel verileri CSV'den, etiketler Metadata'dan!
X_svm_all = df_svm.iloc[:, :-1].values.astype(np.float32)
y_svm_all = y_dogru_etiketler

X_svm_train = X_svm_all[train_idx]
y_svm_train = y_svm_all[train_idx]
X_svm_val = X_svm_all[val_idx]
y_svm_val = y_svm_all[val_idx]
X_svm_test = X_svm_all[test_idx]
y_svm_test = y_svm_all[test_idx]

print("\n[3/6] Pikseller 28×28 görüntüye dönüştürülüyor...")
X_svm_train_2d = X_svm_train.reshape(-1, 28, 28)
X_svm_val_2d = X_svm_val.reshape(-1, 28, 28)
X_svm_test_2d = X_svm_test.reshape(-1, 28, 28)

print("\n[4/6] HOG özellikleri çıkarılıyor...")
def hog_ozellik_cikar(goruntu_dizisi):
    ozellikler = []
    for i, img in enumerate(goruntu_dizisi):
        feat = hog(img, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
        ozellikler.append(feat)
        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{len(goruntu_dizisi)} işlendi...")
    return np.array(ozellikler, dtype=np.float32)

baslangic = time.time()
X_svm_train_hog = hog_ozellik_cikar(X_svm_train_2d)
X_svm_val_hog = hog_ozellik_cikar(X_svm_val_2d)
X_svm_test_hog = hog_ozellik_cikar(X_svm_test_2d)
print(f"  HOG süresi: {time.time() - baslangic:.1f} saniye")

print("\n[5/6] PCA(n_components=128) uygulanıyor...")
pca_svm = PCA(n_components=128, random_state=42)
X_svm_train_pca = pca_svm.fit_transform(X_svm_train_hog)
X_svm_val_pca = pca_svm.transform(X_svm_val_hog)
X_svm_test_pca = pca_svm.transform(X_svm_test_hog)

print("\n[6/6] SVC(kernel='rbf', class_weight='balanced') eğitiliyor...")
baslangic = time.time()
svm_model = SVC(kernel='rbf', class_weight='balanced', random_state=42, decision_function_shape='ovr')
svm_model.fit(X_svm_train_pca, y_svm_train)
print(f"  Eğitim süresi: {time.time() - baslangic:.1f} saniye")

print(f"\n{'─'*40}")
print("SVM SONUÇLARI (Test Seti)")
print(f"{'─'*40}")
y_svm_pred = svm_model.predict(X_svm_test_pca)
svm_f1_macro = f1_score(y_svm_test, y_svm_pred, average='macro')
print(f"\n  Macro F1 Score: {svm_f1_macro:.4f}")
print(classification_report(y_svm_test, y_svm_pred, target_names=SINIF_ISIMLERI, digits=4))

y_svm_val_pred = svm_model.predict(X_svm_val_pca)
svm_val_f1 = f1_score(y_svm_val, y_svm_val_pred, average='macro')

print(f"\n{'='*60}")
print("BASELINE MODEL KARŞILAŞTIRMASI")
print(f"{'='*60}")
print(f"{'Model':<15s} {'Val F1':>10s} {'Test F1':>10s}")
print(f"{'─'*37}")
print(f"{'k-NN':<15s} {knn_val_f1:>10.4f} {knn_f1_macro:>10.4f}")
print(f"{'SVM (HOG)':<15s} {svm_val_f1:>10.4f} {svm_f1_macro:>10.4f}")
print(f"{'─'*37}")
import cv2
import numpy as np
import joblib
import os
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Fungsi ekstraksi fitur HOG
def extract_hog(img):
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    img = cv2.resize(img, winSize)
    return hog.compute(img).flatten()

# Fungsi load dataset
def load_dataset(base_path):
    data = []
    labels = []
    label_map = {}
    current_label = 0

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            print(f"Loading data dari: {folder}")
            label_map[current_label] = folder
            for file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    features = extract_hog(img)
                    data.append(features)
                    labels.append(current_label)
            current_label += 1

    return np.array(data), np.array(labels), label_map

# Load data
dataset_path = 'dataset'
X, y, label_map = load_dataset(dataset_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Model SVM ==========
model_svm = LinearSVC(max_iter=10000)
model_svm.fit(X_train, y_train)
y_pred_svm = model_svm.predict(X_test)
print("\n=== Evaluasi SVM ===")
print(classification_report(y_test, y_pred_svm, target_names=[label_map[i] for i in sorted(label_map)]))
joblib.dump(model_svm, 'model_svm.pkl')

# ========== Model Random Forest ==========
model_rf = RandomForestClassifier(n_estimators=100)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
print("\n=== Evaluasi Random Forest ===")
print(classification_report(y_test, y_pred_rf, target_names=[label_map[i] for i in sorted(label_map)]))
joblib.dump(model_rf, 'model_rf.pkl')

# ========== Model KNN ==========
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)
print("\n=== Evaluasi KNN ===")
print(classification_report(y_test, y_pred_knn, target_names=[label_map[i] for i in sorted(label_map)]))
joblib.dump(model_knn, 'model_knn.pkl')

# Simpan mapping label
joblib.dump(label_map, 'label_map.pkl')
print("Semua model dan label map berhasil disimpan.")

# Fungsi menghitung metrik umum
def get_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1_score': f1_score(y_true, y_pred, average='macro')
    }

# Hitung metrik untuk setiap model
metrics_svm = get_metrics(y_test, y_pred_svm)
metrics_rf = get_metrics(y_test, y_pred_rf)
metrics_knn = get_metrics(y_test, y_pred_knn)

# Tampilkan dalam bentuk tabel
print("\n=== Tabel Perbandingan Kinerja Model ===")
print(f"{'Model':<15}{'Akurasi':<10}{'Precision':<12}{'Recall':<10}{'F1-score':<10}")
print("-" * 57)
for name, metrik in zip(
    ['SVM', 'Random Forest', 'KNN'],
    [metrics_svm, metrics_rf, metrics_knn]
):
    print(f"{name:<15}{metrik['accuracy']:<10.2f}{metrik['precision']:<12.2f}{metrik['recall']:<10.2f}{metrik['f1_score']:<10.2f}")
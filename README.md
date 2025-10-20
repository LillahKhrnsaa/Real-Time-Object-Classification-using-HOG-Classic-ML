
# 🎥 Real-Time Object Classification using HOG + Classic ML

Proyek ini mengimplementasikan **klasifikasi objek secara *real-time*** melalui *webcam* menggunakan kombinasi:

1.  **Ekstraksi Fitur:** **HOG (Histogram of Oriented Gradients)**.
2.  **Algoritma Klasifikasi:** **SVM**, **Random Forest**, dan **K-Nearest Neighbors (KNN)**.

Proyek ini terbagi menjadi dua fase utama: **Training & Evaluation** dan **Real-Time Prediction**.

---

## 🧠 Teknologi yang Digunakan

| Teknologi | Deskripsi |
| :--- | :--- |
| 🐍 **Python 3.10+** | Bahasa utama untuk pengembangan proyek. |
| 🧩 **OpenCV** | Untuk pemrosesan gambar, ekstraksi fitur HOG, dan akses webcam. |
| 🧠 **Scikit-learn** | Untuk semua algoritma klasifikasi (SVM, RF, KNN) dan evaluasi model. |
| 📦 **Joblib** | Untuk menyimpan dan memuat model yang telah dilatih (`.pkl`). |
| 📊 **NumPy** | Untuk operasi numerik dan manipulasi array. |

---

## ⚙️ Struktur Folder

```

hog-classifier/
│
├── dataset/                    \# Folder dataset gambar
│   ├── label\_A/                \# Subfolder 1 (misal 'cat')
│   └── label\_B/                \# Subfolder 2 (misal 'dog')
│
├── train\_models.py             \# Script utama untuk training, ekstraksi HOG, & evaluasi
├── real\_time\_predict.py        \# Script untuk prediksi real-time via webcam
│
├── model\_svm.pkl               \# Model SVM hasil training
├── model\_rf.pkl                \# Model Random Forest hasil training
├── model\_knn.pkl               \# Model KNN hasil training
├── label\_map.pkl               \# Mapping nama label ke angka
│
├── requirements.txt            \# Daftar dependensi
└── README.md                   \# Dokumentasi proyek

````

---

## 🚀 Cara Menjalankan Project

### 1️⃣ Clone Repository & Setup Environment

```bash
# Clone repository
git clone [https://github.com/username/hog-classifier.git](https://github.com/username/hog-classifier.git)
cd hog-classifier

# Buat dan aktifkan virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Isi dari `requirements.txt`:

```
opencv-python
numpy
scikit-learn
joblib
```

### 3️⃣ Siapkan Dataset

Pastikan struktur dataset Anda benar, di mana nama folder dalam `dataset/` otomatis digunakan sebagai label klasifikasi.

```
dataset/
├── class_1/
│   ├── img1.png
│   └── ...
└── class_2/
    ├── img1.png
    └── ...
```

### 4️⃣ Jalankan Pelatihan Model

Jalankan skrip ini untuk mengekstraksi HOG, melatih ketiga model, dan menyimpan hasilnya.

```bash
python train_models.py
```

**Output dari skrip ini:**

1.  Menampilkan tabel perbandingan kinerja model (Akurasi, Precision, Recall, F1-score) di terminal.
2.  Menyimpan empat file model/map ke folder utama: `model_svm.pkl`, `model_rf.pkl`, `model_knn.pkl`, dan `label_map.pkl`.

#### Contoh Hasil Evaluasi Terminal:

```markdown
=== Tabel Perbandingan Kinerja Model ===
Model              Akurasi   Precision   Recall      F1-score  
---------------------------------------------------------
SVM                0.92      0.90        0.91        0.90
Random Forest      0.94      0.93        0.94        0.93
KNN                0.88      0.87        0.88        0.87
```

### 5️⃣ Jalankan Prediksi Real-Time

Setelah model berhasil dilatih, jalankan skrip prediksi *real-time*.

```bash
python real_time_predict.py
```

Anda akan diminta memilih model yang ingin digunakan (1=SVM, 2=Random Forest, 3=KNN). Program kemudian akan membuka *webcam*.

#### 🎥 Contoh Output Real-Time:

  - **Kotak deteksi** (misalnya berwarna hijau) ditampilkan di tengah *frame*.
  - Teks di pojok kiri atas menampilkan:
      - Model yang digunakan.
      - **Label hasil prediksi**.
      - **FPS** (Frame Per Second).

> 💡 **Keluar:** Tekan tombol **`q`** di keyboard untuk menutup jendela *webcam*.

-----

## 💡 Catatan Penting

  - **Input Gambar:** Gambar dalam dataset bisa berupa RGB atau *grayscale*. Skrip **`train_models.py`** akan otomatis mengkonversi ke *grayscale* sebelum ekstraksi fitur HOG.
  - **Penyimpanan Model:** Semua model machine learning dan *label map* disimpan menggunakan **Joblib** dalam format `.pkl`.
  - **Kamera:** Pastikan kamera internal atau *USB webcam* Anda dikenali oleh **OpenCV**.

-----

## 👩‍💻 Author

**Lillah Khairunisa**

  - 📧 `lillahkhairunisa02@gmail.com`
  - 💻 [github.com/LillahKhrnsaa](https://www.google.com/search?q=https://github.com/LillahKhrnsaa)

<!-- end list -->

```
```

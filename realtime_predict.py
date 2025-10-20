import cv2
import joblib
import numpy as np
import time

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

# Pilihan model
print("Pilih model yang akan digunakan:")
print("1. SVM")
print("2. Random Forest")
print("3. KNN")
pilihan = input("Masukkan angka [1-3]: ")

if pilihan == '1':
    model = joblib.load('model_svm.pkl')
    model_name = "SVM"
elif pilihan == '2':
    model = joblib.load('model_rf.pkl')
    model_name = "Random Forest"
elif pilihan == '3':
    model = joblib.load('model_knn.pkl')
    model_name = "KNN"
else:
    print("Pilihan tidak valid.")
    exit()

label_map = joblib.load('label_map.pkl')

# Mulai kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Tidak bisa membuka kamera.")
    exit()

print(f"Model {model_name} digunakan untuk prediksi real-time.")

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame.")
        break

    h, w, _ = frame.shape
    size = min(h, w)
    crop = frame[h//2 - 100:h//2 + 100, w//2 - 100:w//2 + 100]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    features = extract_hog(gray).reshape(1, -1)

    pred = model.predict(features)[0]
    label = label_map[pred]

    # Hitung FPS
    fps = 1.0 / (time.time() - start_time)

    # Tampilkan frame
    cv2.rectangle(frame, (w//2 - 100, h//2 - 100), (w//2 + 100, h//2 + 100), (0, 255, 0), 2)
    cv2.putText(frame, f"{model_name}: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Real-Time Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

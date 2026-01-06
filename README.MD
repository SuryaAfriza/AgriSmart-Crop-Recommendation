# 🌾 AgriSmart - Intelligent Crop Recommendation System

Sistem rekomendasi pertanian cerdas yang menggunakan **Machine Learning (XGBoost)** untuk membantu petani memilih jenis tanaman pangan yang paling optimal berdasarkan analisis unsur hara tanah (**N, P, K**) dan kondisi iklim mikro.

Proyek ini disusun untuk memenuhi **Tugas Besar Mata Kuliah Machine Learning I**.

---

## 📌 Latar Belakang Masalah

Kegagalan panen seringkali terjadi akibat ketidaksesuaian antara jenis tanaman dengan kondisi lahan.  
Petani membutuhkan alat bantu yang cepat dan akurat untuk mengambil keputusan tanam yang berbasis data (*data-driven decision making*).

**Solusi:**  
Aplikasi ini memprediksi tanaman terbaik dengan menganalisis **7 parameter lingkungan**.

---

## 📂 Dataset

- **Sumber:** Crop Recommendation Dataset (Public – Kaggle)  
- **Ukuran Data:** 2200 baris data (balanced, 100 sampel per kelas)

### 🔢 Fitur Input (7 Parameter)
1. **Nitrogen (N)** – Rasio nitrogen dalam tanah  
2. **Phosphorous (P)** – Rasio fosfor dalam tanah  
3. **Potassium (K)** – Rasio kalium dalam tanah  
4. **Temperature** – Suhu lingkungan (°C)  
5. **Humidity** – Kelembapan udara (%)  
6. **pH** – Tingkat keasaman tanah  
7. **Rainfall** – Curah hujan (mm)

### 🎯 Target
22 kelas tanaman:
Rice, Maize, Chickpea, Kidneybeans, Pigeonpeas, Mothbeans, Mungbean, Blackgram, Lentil,  
Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya,  
Coconut, Cotton, Jute, Coffee.

---

## 🚀 Cara Menjalankan (Instalasi Lokal)

Ikuti langkah-langkah berikut untuk menjalankan aplikasi di komputer Anda.

---

### 1. Clone Repository

```bash
git clone https://github.com/USERNAME-KAMU/AgriSmart-Crop-Recommendation.git
cd AgriSmart-Crop-Recommendation


## 🚀 Cara Menjalankan (Instalasi Lokal)

Ikuti langkah ini untuk menjalankan aplikasi di komputer Anda:

### 1. Clone Repository

```bash
git clone https://github.com/SuryaAfriza/AgriSmart-Crop-Recommendation.git
cd AgriSmart-Crop-Recommendation
```

### 2. Buat Virtual Environment (Opsional tapi Disarankan)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

Pastikan terkoneksi internet.

```bash
pip install -r requirements.txt
```

### 4. Latih Model (Training)

Jalankan script ini untuk memproses data dan menghasilkan model `.pkl`.

```bash
python train_model.py
```

*Tunggu hingga muncul pesan "✅ SELESAI!" dan file `model_xgboost.pkl` terbentuk.*

### 5. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser Anda (biasanya di `http://localhost:8501`).

## ✨ Fitur Aplikasi

1. **🏠 Home:** Dashboard informatif tentang proyek.
2. **🔍 Prediksi Manual:** Input nilai N, P, K, dll secara manual untuk analisis cepat.
3. **📂 Upload & Batch Prediction:** Upload file CSV berisi ratusan data tanah, sistem memprediksi semuanya sekaligus. Dilengkapi fitur **Download Hasil (CSV)**.
4. **📊 Laporan Evaluasi:** Menampilkan transparansi model dengan metrik (Akurasi, Precision, Recall), Confusion Matrix, dan grafik **SHAP** untuk explainability.

## 📊 Hasil Evaluasi Model

Kami membandingkan dua algoritma untuk memastikan performa terbaik.

| Model | Akurasi | Keterangan |
| ----- | ----- | ----- |
| **Naive Bayes (Baseline)** | ~94.5% | Cepat, namun kurang akurat pada pola kompleks. |
| **XGBoost (Advanced)** | **~98.7%** | Sangat akurat dan robust terhadap noise. |

**Interpretabilitas (SHAP):**
Fitur yang paling dominan mempengaruhi prediksi adalah **Curah Hujan (Rainfall)** dan **Kelembapan (Humidity)**, diikuti oleh unsur hara K dan N.

## 🛠️ Tech Stack

* **Language:** Python
* **GUI Framework:** Streamlit
* **Machine Learning:** Scikit-Learn, XGBoost
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, SHAP


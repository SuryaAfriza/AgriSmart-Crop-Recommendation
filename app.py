import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from PIL import Image

# Konfigurasi Halaman Website
st.set_page_config(page_title="AgriSmart - Rekomendasi Tani", layout="wide", page_icon="🌾")

# Custom CSS agar tampilan agak cantik
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat aset (Model, Scaler, dll)
@st.cache_resource
def load_assets():
    try:
        model_xgb = joblib.load('model_xgboost.pkl')
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')
        with open('metrics_report.json', 'r') as f:
            metrics = json.load(f)
        return model_xgb, scaler, le, metrics
    except FileNotFoundError:
        return None, None, None, None

model_xgb, scaler, le, metrics = load_assets()

# --- SIDEBAR (Menu Kiri) ---
st.sidebar.title("🌾 AgriSmart")
st.sidebar.caption("Sistem Cerdas Pertanian")
menu = st.sidebar.radio("Pilih Menu", 
    ["🏠 Home", "🔍 Prediksi Manual", "📂 Upload Excel/CSV", "📊 Laporan Evaluasi"])

# --- HALAMAN 1: HOME ---
if menu == "🏠 Home":
    st.title("Selamat Datang di AgriSmart 🌱")
    # Menggunakan gambar placeholder yang valid
    st.image("https://plus.unsplash.com/premium_photo-1661962692059-55d5a4319814?q=80&w=1000&auto=format&fit=crop", use_column_width=True)
    st.markdown("""
    ### Solusi Gagal Panen dengan AI
    Aplikasi ini membantu petani menentukan tanaman pangan yang tepat berdasarkan kondisi tanah dan iklim menggunakan Machine Learning.
    
    **Fitur Utama:**
    1.  **Analisis Tanah:** Mempertimbangkan Nitrogen, Fosfor, Kalium, dan pH.
    2.  **Analisis Cuaca:** Mempertimbangkan Suhu, Kelembapan, dan Curah Hujan.
    3.  **Akurasi Tinggi:** Menggunakan algoritma **XGBoost**.
    """)

# --- HALAMAN 2: PREDIKSI MANUAL ---
elif menu == "🔍 Prediksi Manual":
    st.header("🔍 Cek Rekomendasi Tanaman")
    st.write("Masukkan data tanah dan lingkungan secara manual:")
    
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Nitrogen (N)", 0, 150, 50)
        p = st.number_input("Phosphorous (P)", 0, 150, 50)
        k = st.number_input("Potassium (K)", 0, 210, 50)
        ph = st.number_input("pH Tanah (0-14)", 0.0, 14.0, 6.5)
    with col2:
        temp = st.number_input("Suhu (°C)", 0.0, 50.0, 26.0)
        hum = st.number_input("Kelembapan (%)", 0.0, 100.0, 80.0)
        rain = st.number_input("Curah Hujan (mm)", 0.0, 300.0, 200.0)
        
    if st.button("🌱 Analisa Sekarang", type="primary"):
        if model_xgb:
            # Siapkan data input
            input_data = np.array([[n, p, k, temp, hum, ph, rain]])
            input_scaled = scaler.transform(input_data)
            
            # Prediksi
            pred_idx = model_xgb.predict(input_scaled)[0]
            pred_label = le.inverse_transform([pred_idx])[0]
            
            st.success(f"Tanaman yang paling cocok adalah: **{pred_label}**")
            st.info(f"💡 Tips: Pastikan irigasi cukup jika memilih {pred_label}.")
        else:
            st.error("⚠️ Model belum siap! Jalankan `python train_model.py` dulu di terminal.")

# --- HALAMAN 3: UPLOAD CSV (Batch Prediction) ---
elif menu == "📂 Upload Excel/CSV":
    st.header("📂 Prediksi Banyak Data Sekaligus")
    uploaded_file = st.file_uploader("Upload file CSV data tanah", type=["csv"])
    
    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write("Preview Data:", df_upload.head(3))
            
            if st.button("Jalankan Prediksi Batch"):
                required_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
                # Cek apakah kolom lengkap (case sensitive biasanya, tapi kita asumsikan user pakai format yg sama)
                # Tips: Dataset asli pakai huruf kecil untuk temperature dkk, pastikan csv user sama.
                
                # Kita normalisasi nama kolom dulu biar aman
                df_upload.columns = [c.strip() for c in df_upload.columns]
                
                if all(col in df_upload.columns for col in required_cols):
                    X_batch = scaler.transform(df_upload[required_cols])
                    preds = model_xgb.predict(X_batch)
                    df_upload['Hasil_Rekomendasi'] = le.inverse_transform(preds)
                    
                    st.dataframe(df_upload)
                    
                    # Tombol Download
                    csv = df_upload.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Hasil (CSV)", csv, "hasil_prediksi.csv", "text/csv")
                else:
                    st.error(f"Format salah! Pastikan ada kolom: {required_cols}")
        except Exception as e:
            st.error(f"Terjadi kesalahan membaca file: {e}")

# --- HALAMAN 4: EVALUASI (Untuk Laporan) ---
elif menu == "📊 Laporan Evaluasi":
    st.header("📊 Kinerja Model AI")
    
    if metrics:
        # Metrik Angka
        c1, c2, c3 = st.columns(3)
        c1.metric("Akurasi Total", f"{metrics['accuracy']*100:.1f}%")
        c2.metric("Rata-rata Precision", f"{metrics['macro avg']['precision']*100:.1f}%")
        c3.metric("Rata-rata Recall", f"{metrics['macro avg']['recall']*100:.1f}%")
        
        st.markdown("---")
        
        # Grafik
        tab1, tab2 = st.tabs(["Confusion Matrix", "SHAP Explainability"])
        
        with tab1:
            st.write("### Seberapa tepat tebakan model?")
            try:
                st.image("confusion_matrix.png")
                st.caption("Semakin gelap warna diagonal, semakin bagus modelnya.")
            except:
                st.warning("Gambar belum dibuat.")
                
        with tab2:
            st.write("### Faktor apa yang paling penting?")
            try:
                st.image("shap_summary.png")
                st.caption("Fitur paling atas = Paling mempengaruhi keputusan AI.")
            except:
                st.warning("Gambar belum dibuat.")
    else:
        st.error("Metrik belum tersedia. Jalankan training dulu.")
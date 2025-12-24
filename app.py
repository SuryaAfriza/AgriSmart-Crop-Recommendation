import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Konfigurasi Halaman Website
st.set_page_config(page_title="AgriSmart - Rekomendasi Tani", layout="wide", page_icon="🌾")

# Custom CSS agar tampilan agak cantik
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
    .metric-container { background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;}
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
st.sidebar.caption("Sistem Pendukung Keputusan Petani")
st.sidebar.markdown("---")
menu = st.sidebar.radio("Navigasi Menu", 
    ["🏠 Home", "🔍 Prediksi Manual", "📂 Upload Data Batch", "📊 Laporan Evaluasi"])

# --- HALAMAN 1: HOME ---
if menu == "🏠 Home":
    st.title("Selamat Datang di AgriSmart 🌱")
    
    # Gambar dihapus sesuai permintaan user

    st.markdown("""
    ### Solusi Gagal Panen dengan Kecerdasan Buatan (AI)
    Aplikasi ini dirancang untuk membantu petani dan penyuluh pertanian dalam menentukan **jenis tanaman pangan yang paling optimal** untuk ditanam, menyesuaikan dengan profil kimia tanah dan kondisi iklim setempat.
    
    #### 🌟 Mengapa Aplikasi Ini Penting?
    Pertanian tradisional seringkali mengandalkan intuisi yang bisa meleset. Dengan AgriSmart, keputusan bertani didasarkan pada data historis yang akurat.
    
    #### 🔍 Parameter Analisis (Fitur Input):
    Aplikasi menganalisis 7 faktor kunci lingkungan:
    1.  **Kandungan Tanah:**
        * **Nitrogen (N):** Penting untuk pertumbuhan daun.
        * **Fosfor (P):** Penting untuk akar dan pematangan buah.
        * **Kalium (K):** Penting untuk ketahanan penyakit.
        * **pH Tanah:** Tingkat keasaman lahan.
    2.  **Kondisi Iklim:**
        * **Temperatur:** Suhu rata-rata lingkungan.
        * **Kelembapan:** Persentase uap air di udara.
        * **Curah Hujan:** Ketersediaan air alami (mm).
    
    """)

# --- HALAMAN 2: PREDIKSI MANUAL ---
elif menu == "🔍 Prediksi Manual":
    st.header("🔍 Cek Rekomendasi Tanaman")
    st.write("Masukkan data hasil uji tanah dan cuaca di bawah ini:")
    
    # Membuat form input yang rapi dengan kolom
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1️⃣ Profil Tanah")
        n = st.number_input("Kadar Nitrogen (N)", 0, 150, 50, help="Rasio kandungan Nitrogen dalam tanah")
        p = st.number_input("Kadar Fosfor (P)", 0, 150, 50, help="Rasio kandungan Fosfor dalam tanah")
        k = st.number_input("Kadar Kalium (K)", 0, 210, 50, help="Rasio kandungan Kalium dalam tanah")
        ph = st.number_input("pH Tanah (0-14)", 0.0, 14.0, 6.5, help="Tingkat keasaman tanah")
    
    with col2:
        st.subheader("2️⃣ Profil Iklim")
        temp = st.number_input("Suhu Rata-rata (°C)", 0.0, 50.0, 26.0)
        hum = st.number_input("Kelembapan Udara (%)", 0.0, 100.0, 80.0)
        rain = st.number_input("Curah Hujan (mm)", 0.0, 300.0, 200.0)
        
    st.markdown("---")
    
    if st.button("🌱 Analisa Kesesuaian Lahan", type="primary"):
        if model_xgb:
            # Siapkan data input
            input_data = np.array([[n, p, k, temp, hum, ph, rain]])
            input_scaled = scaler.transform(input_data)
            
            # Prediksi
            pred_idx = model_xgb.predict(input_scaled)[0]
            pred_label = le.inverse_transform([pred_idx])[0]
            
            st.success("Analisis Selesai! Berikut rekomendasinya:")
            st.markdown(f"""
            <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;">
                <h2 style="color: #155724; margin:0;">Tanaman Terbaik: {pred_label}</h2>
                <p>Berdasarkan kombinasi NPK dan Curah Hujan yang Anda masukkan, lahan ini paling produktif jika ditanami <b>{pred_label}</b>.</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.error("⚠️ Model belum siap! Jalankan `python train_model.py` dulu di terminal.")

# --- HALAMAN 3: UPLOAD CSV (Batch Prediction) ---
elif menu == "📂 Upload Data Batch":
    st.header("📂 Prediksi Banyak Data Sekaligus")
    st.write("Gunakan fitur ini jika Anda memiliki data lahan dari banyak lokasi dalam format Excel/CSV.")
    
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    
    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write("Preview Data:", df_upload.head(3))
            
            if st.button("Jalankan Prediksi Batch"):
                required_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
                
                # Normalisasi nama kolom (hilangkan spasi)
                df_upload.columns = [c.strip() for c in df_upload.columns]
                
                if all(col in df_upload.columns for col in required_cols):
                    # Progress bar biar keren
                    with st.spinner('Sedang menganalisis ribuan data...'):
                        X_batch = scaler.transform(df_upload[required_cols])
                        preds = model_xgb.predict(X_batch)
                        df_upload['Rekomendasi_Tanaman'] = le.inverse_transform(preds)
                    
                    st.success("✅ Prediksi Selesai!")
                    st.dataframe(df_upload.head)
                    
                    # Tombol Download
                    csv = df_upload.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Hasil Analisis (CSV)",
                        data=csv,
                        file_name="hasil_rekomendasi_pertanian.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Format kolom salah! File CSV wajib memiliki kolom berikut: {required_cols}")
        except Exception as e:
            st.error(f"Terjadi kesalahan membaca file: {e}")

# --- HALAMAN 4: EVALUASI (Untuk Laporan) ---
elif menu == "📊 Laporan Evaluasi":
    st.header("📊 Transparansi & Kinerja Model AI")
    st.write("Halaman ini menyajikan metrik evaluasi teknis")
    
    if metrics:
        # Metrik Angka dengan desain card
        col1, col2, col3 = st.columns(3)
        col1.metric("Akurasi Model", f"{metrics['accuracy']*100:.1f}%", help="Persentase tebakan benar dari total data uji")
        col2.metric("Macro Precision", f"{metrics['macro avg']['precision']*100:.1f}%", help="Tingkat ketepatan prediksi positif")
        col3.metric("Macro Recall", f"{metrics['macro avg']['recall']*100:.1f}%", help="Tingkat keberhasilan menemukan kelas positif")
        
        st.markdown("---")
        
        # Grafik
        tab1, tab2 = st.tabs(["Confusion Matrix (Ketepatan)", "SHAP Values (Interpretasi)"])
        
        with tab1:
            st.write("#### Confusion Matrix")
            st.write("Visualisasi ini menunjukkan di mana model sering melakukan kesalahan. Warna biru tua di diagonal menandakan prediksi yang sangat akurat.")
            try:
                st.image("confusion_matrix.png", caption="Confusion Matrix Model XGBoost", use_container_width=True)
            except:
                st.warning("Gambar belum tersedia.")
                
        with tab2:
            st.write("#### Feature Importance (SHAP)")
            st.write("Grafik ini menjawab pertanyaan: **'Faktor apa yang paling menentukan rekomendasi?'**")
            try:
                st.image("shap_summary.png", caption="Faktor Dominan dalam Penentuan Tanaman", use_container_width=True)
                st.info("💡 **Cara Membaca:** Fitur yang berada di urutan paling atas adalah fitur yang paling berpengaruh. Biasanya Curah Hujan atau Kelembapan menjadi faktor penentu utama.")
            except:
                st.warning("Gambar belum tersedia.")
    else:
        st.error("Metrik belum tersedia. Jalankan training dulu.")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data
try:
    df = pd.read_csv('Crop_recommendation.csv')
    print("Data Loaded Successfully!")
except:
    print("Error: File csv tidak ditemukan.")
    exit()

# 1. HEATMAP KORELASI
# Menunjukkan hubungan antar fitur (misal: apakah P dan K berkorelasi?)
plt.figure(figsize=(10, 8))
# Hapus kolom label (string) sebelum korelasi
correlation = df.drop('label', axis=1).corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriks Korelasi Fitur Tanah & Iklim')
plt.tight_layout()
plt.savefig('eda_correlation_matrix.png', dpi=300)
print("✅ Gambar Korelasi disimpan: eda_correlation_matrix.png")

# 2. DISTRIBUSI LABEL (Cek Balance/Imbalance)
# Menunjukkan apakah jumlah data per tanaman seimbang
plt.figure(figsize=(12, 6))
sns.countplot(y='label', data=df, palette='viridis')
plt.title('Distribusi Jumlah Sampel per Jenis Tanaman')
plt.xlabel('Jumlah Sampel')
plt.ylabel('Jenis Tanaman')
plt.tight_layout()
plt.savefig('eda_label_distribution.png', dpi=300)
print("✅ Gambar Distribusi disimpan: eda_label_distribution.png")

# 3. HUBUNGAN CURAH HUJAN vs TANAMAN (Boxplot)
# Menunjukkan tanaman mana yang butuh banyak air
plt.figure(figsize=(14, 7))
sns.boxplot(x='label', y='rainfall', data=df, palette='Spectral')
plt.xticks(rotation=90)
plt.title('Distribusi Kebutuhan Curah Hujan per Tanaman')
plt.ylabel('Curah Hujan (mm)')
plt.tight_layout()
plt.savefig('eda_rainfall_boxplot.png', dpi=300)
print("✅ Gambar Boxplot disimpan: eda_rainfall_boxplot.png")
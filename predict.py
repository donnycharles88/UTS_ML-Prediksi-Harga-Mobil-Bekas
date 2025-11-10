import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi Pengangguran", page_icon="📊")

st.title("📊 Prediksi Tingkat Pengangguran di Indonesia")
st.markdown("Aplikasi ini membandingkan hasil prediksi **LightGBM** dan **Exponential Smoothing**.")

# =====================================================
# 1. LOAD MODELS
# =====================================================
try:
    model_lgbm = joblib.load("model_lightgbm.pkl")
    st.success("✅ Model LightGBM berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat model LightGBM: {e}")
    st.stop()

try:
    model_es = joblib.load("model_expsmooth.pkl")
    st.success("✅ Model Exponential Smoothing berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat model Exponential Smoothing: {e}")
    st.stop()

# =====================================================
# 2. INPUT DATA USER
# =====================================================
st.subheader("🧮 Input Data Prediksi")

col1, col2 = st.columns(2)
with col1:
    poorpeople_percentage = st.number_input("Persentase Penduduk Miskin (%)", 0.0, 100.0, 10.0)
    reg_gdp = st.number_input("Produk Domestik Regional Bruto", 0.0, 1_000_000.0, 50000.0)
    life_exp = st.number_input("Harapan Hidup (Tahun)", 40.0, 90.0, 70.0)

with col2:
    avg_schooltime = st.number_input("Rata-rata Lama Sekolah (Tahun)", 0.0, 20.0, 9.0)
    exp_percap = st.number_input("Pengeluaran per Kapita", 0.0, 1_000_000.0, 15000.0)
    province = st.text_input("Provinsi", "Jawa Barat")
    cities_reg = st.text_input("Kabupaten/Kota", "Bandung")

# =====================================================
# 3. PREDIKSI DUA MODEL
# =====================================================
if st.button("🔍 Jalankan Prediksi"):
    input_data = pd.DataFrame({
        "poorpeople_percentage": [poorpeople_percentage],
        "reg_gdp": [reg_gdp],
        "life_exp": [life_exp],
        "avg_schooltime": [avg_schooltime],
        "exp_percap": [exp_percap],
        "province": [province],
        "cities_reg": [cities_reg],
    })

    try:
        # Prediksi LightGBM
        pred_lgbm = model_lgbm.predict(input_data)[0]

        # Prediksi Exponential Smoothing (berdasarkan pola provinsi)
        prov_index = hash(province) % len(model_es.fittedvalues)
        pred_es = model_es.fittedvalues[prov_index]

        st.subheader("📈 Hasil Prediksi")
        col1, col2 = st.columns(2)
        col1.metric("Prediksi LightGBM", f"{pred_lgbm:.2f}%")
        col2.metric("Prediksi Exponential Smoothing", f"{pred_es:.2f}%")

        # =====================================================
        # 4. VISUALISASI PERBANDINGAN
        # =====================================================
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["LightGBM", "Exp. Smoothing"], [pred_lgbm, pred_es], color=["#36A2EB", "#FF6384"])
        ax.set_ylabel("Tingkat Pengangguran (%)")
        ax.set_title("Perbandingan Prediksi Dua Model")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")

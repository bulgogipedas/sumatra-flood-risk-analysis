"""
Prediction Page - Input data dan dapatkan prediksi risiko banjir
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

st.set_page_config(page_title="Prediksi Risiko", page_icon="🎯", layout="wide")

PROJECT_ROOT = Path(__file__).parent.parent.parent

@st.cache_resource
def load_model():
    """Load model and scaler."""
    models_dir = PROJECT_ROOT / "models"
    
    # Get latest model
    pkl_files = sorted([f for f in models_dir.glob("*.pkl") if "scaler" not in f.name], reverse=True)
    scaler_files = sorted(models_dir.glob("*_scaler.pkl"), reverse=True)
    json_files = sorted(models_dir.glob("*_metadata.json"), reverse=True)
    
    if not pkl_files:
        return None, None, None
    
    with open(pkl_files[0], "rb") as f:
        model = pickle.load(f)
    
    scaler = None
    if scaler_files:
        with open(scaler_files[0], "rb") as f:
            scaler = pickle.load(f)
    
    metadata = None
    if json_files:
        with open(json_files[0]) as f:
            metadata = json.load(f)
    
    return model, scaler, metadata

@st.cache_data
def load_data():
    """Load reference data for ranges."""
    path = PROJECT_ROOT / "data" / "processed" / "complete_analysis_dataset.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None

st.title("Prediksi Risiko Banjir")
st.caption("Masukkan data wilayah untuk mendapatkan prediksi risiko")

model, scaler, metadata = load_model()
df = load_data()

if model is None:
    st.error("Model tidak ditemukan. Jalankan training terlebih dahulu.")
    st.code("uv run python scripts/06_train_better_model.py")
    st.stop()

st.markdown("---")

# =============================================================================
# MODEL INFO
# =============================================================================
st.header("Tentang Model")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Performa Model")
    if metadata and "training_history" in metadata:
        m = metadata["training_history"]["metrics"]
        cv = metadata["training_history"].get("cv_results", {})
        
        st.metric("Accuracy", f"{m.get('accuracy', 0):.0%}")
        st.metric("ROC-AUC", f"{m.get('roc_auc', 0):.0%}")
        
        if cv:
            st.caption(f"CV Accuracy: {cv['accuracy']['mean']:.0%} ± {cv['accuracy']['std']:.0%}")

with col2:
    st.subheader("Interpretasi")
    st.markdown("""
    Model memprediksi apakah suatu wilayah berisiko **tinggi** atau **rendah** 
    mengalami banjir berdasarkan faktor lingkungan.
    
    **High Risk**: Kemungkinan besar mengalami banjir signifikan
    
    **Low Risk**: Kemungkinan lebih rendah mengalami banjir signifikan
    """)

st.markdown("---")

# =============================================================================
# INPUT FORM
# =============================================================================
st.header("Input Data Wilayah")

st.info("""
Masukkan karakteristik wilayah yang ingin diprediksi. 
Gunakan slider untuk mengatur nilai setiap parameter.
""")

# Get feature names
if metadata and "feature_names" in metadata:
    feature_names = metadata["feature_names"]
else:
    feature_names = ["forest_loss_pct", "palm_oil_pct", "rainfall_annual_mean_mm", "rainfall_extreme_days_avg"]

# Reference statistics
if df is not None:
    stats = df.describe()

# Input columns
col1, col2 = st.columns(2)

inputs = {}

with col1:
    st.subheader("Faktor Lingkungan")
    
    # Forest loss
    if df is not None and "forest_loss_pct" in df.columns:
        min_val, max_val = float(df["forest_loss_pct"].min()), float(df["forest_loss_pct"].max())
        mean_val = float(df["forest_loss_pct"].mean())
    else:
        min_val, max_val, mean_val = 0.0, 50.0, 15.0
    
    inputs["forest_loss_pct"] = st.slider(
        "Deforestasi (%)",
        min_value=min_val,
        max_value=max_val,
        value=mean_val,
        help="Persentase kehilangan tutupan hutan"
    )
    
    # Palm oil / agriculture
    if df is not None and "palm_oil_pct" in df.columns:
        min_val, max_val = float(df["palm_oil_pct"].min()), float(df["palm_oil_pct"].max())
        mean_val = float(df["palm_oil_pct"].mean())
    else:
        min_val, max_val, mean_val = 0.0, 80.0, 25.0
    
    inputs["palm_oil_pct"] = st.slider(
        "Lahan Pertanian (%)",
        min_value=min_val,
        max_value=max_val,
        value=mean_val,
        help="Persentase lahan pertanian/perkebunan"
    )

with col2:
    st.subheader("Faktor Iklim")
    
    # Rainfall
    if df is not None and "rainfall_annual_mean_mm" in df.columns:
        min_val, max_val = float(df["rainfall_annual_mean_mm"].min()), float(df["rainfall_annual_mean_mm"].max())
        mean_val = float(df["rainfall_annual_mean_mm"].mean())
    else:
        min_val, max_val, mean_val = 1000.0, 4000.0, 2500.0
    
    inputs["rainfall_annual_mean_mm"] = st.slider(
        "Curah Hujan Tahunan (mm)",
        min_value=min_val,
        max_value=max_val,
        value=mean_val,
        help="Rata-rata curah hujan per tahun"
    )
    
    # Extreme days
    if df is not None and "rainfall_extreme_days_avg" in df.columns:
        min_val, max_val = float(df["rainfall_extreme_days_avg"].min()), float(df["rainfall_extreme_days_avg"].max())
        mean_val = float(df["rainfall_extreme_days_avg"].mean())
    else:
        min_val, max_val, mean_val = 0.0, 50.0, 15.0
    
    inputs["rainfall_extreme_days_avg"] = st.slider(
        "Hari Hujan Ekstrem/Tahun",
        min_value=min_val,
        max_value=max_val,
        value=mean_val,
        help="Rata-rata hari dengan curah hujan ekstrem per tahun"
    )

# Create derived features (same as training)
inputs["high_deforestation"] = 1.0 if inputs.get("forest_loss_pct", 0) > (df["forest_loss_pct"].quantile(0.7) if df is not None else 20) else 0.0
inputs["high_rainfall"] = 1.0 if inputs.get("rainfall_annual_mean_mm", 0) > (df["rainfall_annual_mean_mm"].quantile(0.6) if df is not None else 2500) else 0.0
inputs["high_agriculture"] = 1.0 if inputs.get("palm_oil_pct", 0) > (df["palm_oil_pct"].quantile(0.6) if df is not None else 30) else 0.0
inputs["defor_rain"] = inputs["high_deforestation"] * inputs["high_rainfall"]
inputs["agri_rain"] = inputs["high_agriculture"] * inputs["high_rainfall"]

st.markdown("---")

# =============================================================================
# PREDICTION
# =============================================================================
st.header("Hasil Prediksi")

if st.button("Prediksi Risiko", type="primary", use_container_width=True):
    # Prepare input
    X_input = pd.DataFrame([inputs])
    
    # Ensure correct feature order
    if metadata and "feature_names" in metadata:
        features = metadata["feature_names"]
        # Add missing features with 0
        for f in features:
            if f not in X_input.columns:
                X_input[f] = 0
        X_input = X_input[features]
    
    # Scale if scaler available
    if scaler is not None:
        X_scaled = scaler.transform(X_input)
    else:
        X_scaled = X_input.values
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("### 🔴 HIGH RISK")
            st.markdown("""
            Wilayah dengan karakteristik ini memiliki **risiko tinggi** mengalami 
            banjir signifikan berdasarkan model prediksi.
            """)
        else:
            st.success("### 🟢 LOW RISK")
            st.markdown("""
            Wilayah dengan karakteristik ini memiliki **risiko lebih rendah** 
            mengalami banjir signifikan berdasarkan model prediksi.
            """)
    
    with col2:
        st.subheader("Probabilitas")
        
        prob_low = probability[0] * 100
        prob_high = probability[1] * 100
        
        st.metric("Probabilitas Low Risk", f"{prob_low:.1f}%")
        st.metric("Probabilitas High Risk", f"{prob_high:.1f}%")
        
        # Confidence
        confidence = max(prob_low, prob_high)
        if confidence > 70:
            st.info(f"Model **cukup yakin** dengan prediksi ini ({confidence:.0f}%)")
        else:
            st.warning(f"Model **kurang yakin** ({confidence:.0f}%). Pertimbangkan faktor lain.")
    
    st.markdown("---")
    
    # Factor analysis
    st.subheader("Analisis Faktor")
    
    factor_analysis = []
    
    if inputs["forest_loss_pct"] > 20:
        factor_analysis.append(("⚠️", "Deforestasi tinggi", "Meningkatkan risiko banjir"))
    if inputs["rainfall_annual_mean_mm"] > 2500:
        factor_analysis.append(("⚠️", "Curah hujan tinggi", "Meningkatkan risiko banjir"))
    if inputs["palm_oil_pct"] > 30:
        factor_analysis.append(("⚠️", "Lahan pertanian luas", "Dapat mengurangi resapan air"))
    if inputs["rainfall_extreme_days_avg"] > 20:
        factor_analysis.append(("⚠️", "Banyak hari hujan ekstrem", "Meningkatkan risiko banjir"))
    
    if inputs["forest_loss_pct"] < 10:
        factor_analysis.append(("✅", "Deforestasi rendah", "Menurunkan risiko banjir"))
    if inputs["rainfall_annual_mean_mm"] < 2000:
        factor_analysis.append(("✅", "Curah hujan moderat", "Menurunkan risiko banjir"))
    
    if factor_analysis:
        for icon, factor, impact in factor_analysis:
            st.markdown(f"{icon} **{factor}**: {impact}")
    else:
        st.info("Semua faktor dalam rentang normal")

st.markdown("---")

# =============================================================================
# COMPARISON
# =============================================================================
st.header("Bandingkan dengan Data Aktual")

if df is not None:
    st.markdown("Lihat bagaimana input Anda dibandingkan dengan kabupaten yang sudah ada:")
    
    # Find similar districts
    df_compare = df.copy()
    df_compare["dist_forest"] = abs(df_compare["forest_loss_pct"] - inputs["forest_loss_pct"])
    df_compare["dist_rain"] = abs(df_compare["rainfall_annual_mean_mm"] - inputs["rainfall_annual_mean_mm"]) / 100
    df_compare["dist_total"] = df_compare["dist_forest"] + df_compare["dist_rain"]
    
    similar = df_compare.nsmallest(5, "dist_total")[["province", "kabupaten", "flood_events_total", "forest_loss_pct", "rainfall_annual_mean_mm"]]
    similar.columns = ["Provinsi", "Kabupaten", "Kejadian Banjir", "Deforestasi (%)", "Curah Hujan (mm)"]
    
    st.subheader("5 Kabupaten Paling Mirip")
    st.dataframe(similar, hide_index=True, use_container_width=True)

st.markdown("---")

# =============================================================================
# DISCLAIMER
# =============================================================================
st.warning("""
**Disclaimer:**

Prediksi ini berdasarkan model machine learning dengan akurasi ~60% dan hanya 
mempertimbangkan faktor lingkungan. Faktor lain seperti infrastruktur drainase, 
topografi detail, dan kondisi DAS tidak tercakup. Gunakan sebagai referensi awal, 
bukan keputusan final.
""")

st.markdown("---")
st.caption("IFRA | Prediksi Risiko Banjir")


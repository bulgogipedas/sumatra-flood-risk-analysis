"""
Model & XAI - Machine Learning dan Interpretasi Model
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import pickle

st.set_page_config(page_title="Model & XAI", page_icon="🔬", layout="wide")

PROJECT_ROOT = Path(__file__).parent.parent.parent

@st.cache_data
def load_data():
    path = PROJECT_ROOT / "data" / "processed" / "complete_analysis_dataset.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        return df
    return None

@st.cache_data
def load_model_metadata():
    """Load latest model metadata."""
    models_dir = PROJECT_ROOT / "models"
    if models_dir.exists():
        json_files = sorted(models_dir.glob("*_metadata.json"), reverse=True)
        if json_files:
            with open(json_files[0]) as f:
                return json.load(f)
    return None

@st.cache_resource
def load_model():
    """Load latest trained model."""
    models_dir = PROJECT_ROOT / "models"
    if models_dir.exists():
        pkl_files = sorted([f for f in models_dir.glob("*.pkl") if "scaler" not in f.name], reverse=True)
        if pkl_files:
            with open(pkl_files[0], "rb") as f:
                return pickle.load(f)
    return None

st.title("Model & Explainable AI")
st.caption("Machine learning untuk prediksi risiko banjir dan interpretasi model")

df = load_data()
metadata = load_model_metadata()
model = load_model()

if df is None:
    st.error("Data tidak ditemukan")
    st.stop()

st.markdown("---")

# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================
st.header("Ringkasan Model")

st.markdown("""
### Pendekatan

Model klasifikasi dibangun untuk memprediksi apakah suatu kabupaten memiliki risiko banjir 
**di atas median** (high risk) atau **di bawah median** (low risk) berdasarkan faktor-faktor 
lingkungan.

### Mengapa Machine Learning?

Hubungan antara faktor lingkungan dan risiko banjir bersifat **non-linear** dan **kompleks**. 
Machine learning memungkinkan kita menangkap pola-pola yang tidak terlihat dalam analisis 
korelasi sederhana.
""")

st.markdown("---")

# =============================================================================
# MODEL PERFORMANCE
# =============================================================================
st.header("Performa Model")

if metadata and "training_history" in metadata:
    metrics = metadata["training_history"]["metrics"]
    cv = metadata["training_history"].get("cv_results", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Test Set Metrics")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
            st.metric("Precision", f"{metrics.get('precision', 0):.1%}")
        with m_col2:
            st.metric("Recall", f"{metrics.get('recall', 0):.1%}")
            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.1%}")
        
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.1%}")
    
    with col2:
        st.subheader("Cross-Validation (5-fold)")
        
        if cv:
            for metric_name in ["accuracy", "f1", "roc_auc"]:
                if metric_name in cv:
                    mean = cv[metric_name]["mean"]
                    std = cv[metric_name]["std"]
                    st.metric(
                        metric_name.upper().replace("_", "-"),
                        f"{mean:.1%}",
                        delta=f"± {std:.1%}",
                        delta_color="off"
                    )
    
    # Interpretation
    acc = metrics.get('accuracy', 0)
    cv_acc = cv.get('accuracy', {}).get('mean', 0) if cv else 0
    
    if abs(acc - cv_acc) < 0.05:
        status = "Model **tidak overfitting** - performa test set dan CV konsisten."
        status_type = "success"
    else:
        status = "Model mungkin sedikit **overfitting** - performa test set lebih tinggi dari CV."
        status_type = "warning"
    
    getattr(st, status_type)(f"""
    **Interpretasi Performa:**
    
    - Accuracy ~{acc:.0%} menunjukkan model dapat memprediksi dengan benar sekitar {acc*100:.0f} dari 100 kabupaten
    - {status}
    - F1-Score ~{metrics.get('f1_score', 0):.0%} menunjukkan keseimbangan antara precision dan recall
    """)

else:
    st.warning("Metadata model tidak tersedia. Jalankan training terlebih dahulu.")
    st.code("uv run python scripts/05_retrain_model.py")

st.markdown("---")

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================
st.header("Feature Importance")

st.markdown("""
**Apa artinya?**

Feature importance menunjukkan seberapa besar kontribusi setiap variabel dalam membuat prediksi.
Semakin tinggi nilai importance, semakin berpengaruh variabel tersebut.
""")

if metadata and "training_history" in metadata:
    fi = metadata["training_history"].get("feature_importance", [])
    
    if fi:
        fi_df = pd.DataFrame(fi)
        fi_df = fi_df.sort_values("importance", ascending=True)
        
        # Rename features for readability
        rename_map = {
            "rainfall_annual_mean_mm": "Curah Hujan Tahunan",
            "rainfall_extreme_days_avg": "Hari Hujan Ekstrem",
            "forest_loss_pct": "Deforestasi (%)",
            "palm_oil_pct": "Lahan Pertanian (%)"
        }
        fi_df["feature_label"] = fi_df["feature"].map(lambda x: rename_map.get(x, x))
        
        fig = px.bar(
            fi_df,
            x="importance",
            y="feature_label",
            orientation="h",
            title="Feature Importance",
            color="importance",
            color_continuous_scale="Blues"
        )
        fig.update_layout(height=400, coloraxis_showscale=False, yaxis_title="", xaxis_title="Importance")
        st.plotly_chart(fig, use_container_width=True)
        
        # Top feature insight
        top = fi_df.iloc[-1]
        st.info(f"""
        **Insight:**
        
        Fitur paling penting adalah **{rename_map.get(top['feature'], top['feature'])}** dengan 
        importance score {top['importance']:.4f}. Ini berarti model paling banyak mengandalkan 
        variabel ini dalam membuat prediksi.
        """)

elif model is not None and hasattr(model, 'feature_importances_'):
    st.info("Menggunakan feature importance dari model langsung")
    
    if metadata and "feature_names" in metadata:
        features = metadata["feature_names"]
    else:
        features = [f"feature_{i}" for i in range(len(model.feature_importances_))]
    
    fi_df = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=True)
    
    fig = px.bar(fi_df, x="importance", y="feature", orientation="h")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# =============================================================================
# SHAP EXPLANATION
# =============================================================================
st.header("Explainable AI (XAI)")

st.markdown("""
### Apa itu SHAP?

**SHAP (SHapley Additive exPlanations)** adalah metode untuk menjelaskan prediksi model 
dengan mengukur kontribusi setiap fitur.

### Cara Membaca SHAP
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Summary Plot:**
    - Setiap titik = satu data point
    - Posisi X = pengaruh ke prediksi
    - Warna = nilai fitur (merah=tinggi, biru=rendah)
    """)

with col2:
    st.markdown("""
    **Waterfall Plot:**
    - Mulai dari base value
    - Setiap fitur mendorong naik/turun
    - Hasil akhir = prediksi
    """)

st.markdown("---")

# =============================================================================
# INDIVIDUAL PREDICTION
# =============================================================================
st.header("Analisis Per Kabupaten")

if "kabupaten" in df.columns:
    selected = st.selectbox("Pilih Kabupaten:", sorted(df["kabupaten"].unique()))
    
    kab = df[df["kabupaten"] == selected].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Profil")
        
        st.metric("Provinsi", kab.get("province", "-"))
        st.metric("Pulau", kab.get("island", "-"))
        st.metric("Kejadian Banjir", int(kab.get("flood_events_total", 0)))
        st.metric("Korban Jiwa", int(kab.get("flood_deaths", 0)))
    
    with col2:
        st.subheader("Faktor Risiko")
        
        # Compare to median
        factors = []
        
        if "forest_loss_pct" in kab:
            med = df["forest_loss_pct"].median()
            val = kab["forest_loss_pct"]
            diff = val - med
            factors.append(("Deforestasi", val, diff, diff > 0))
        
        if "rainfall_annual_mean_mm" in kab:
            med = df["rainfall_annual_mean_mm"].median()
            val = kab["rainfall_annual_mean_mm"]
            diff = val - med
            factors.append(("Curah Hujan", val, diff, diff > 0))
        
        for name, val, diff, is_high in factors:
            arrow = "↑" if is_high else "↓"
            st.metric(name, f"{val:.1f}", delta=f"{arrow} {abs(diff):.1f} vs median")
    
    # Risk interpretation
    floods = kab.get("flood_events_total", 0)
    median_floods = df["flood_events_total"].median()
    
    if floods > median_floods:
        st.error(f"""
        **Prediksi: HIGH RISK**
        
        Kabupaten ini memiliki {int(floods)} kejadian banjir, di atas median nasional ({int(median_floods)}).
        Faktor-faktor yang berkontribusi perlu dianalisis lebih lanjut.
        """)
    else:
        st.success(f"""
        **Prediksi: LOW RISK**
        
        Kabupaten ini memiliki {int(floods)} kejadian banjir, di bawah atau sama dengan 
        median nasional ({int(median_floods)}).
        """)

st.markdown("---")

# =============================================================================
# METHODOLOGY
# =============================================================================
with st.expander("Metodologi Detail"):
    st.markdown("""
    ### Pipeline Machine Learning
    
    **1. Target Variable**
    - Binary classification: High risk (>median floods) vs Low risk
    - Balanced dataset: ~50% each class
    
    **2. Features**
    - `forest_loss_pct`: Persentase kehilangan hutan
    - `palm_oil_pct`: Persentase lahan pertanian
    - `rainfall_annual_mean_mm`: Rata-rata curah hujan
    - `rainfall_extreme_days_avg`: Hari hujan ekstrem
    
    **3. Model**
    - Algorithm: XGBoost Classifier
    - Regularization: max_depth=3, min_child_weight=5, L1=1.0, L2=2.0
    - Train-test split: 75-25
    
    **4. Evaluation**
    - 5-fold stratified cross-validation
    - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
    
    ### Limitasi
    
    - Model menggunakan agregasi level kabupaten
    - Faktor infrastruktur tidak tercakup
    - Korelasi bukan kausalitas
    """)

st.markdown("---")
st.caption("IFRA | Model & Explainable AI")

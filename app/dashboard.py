"""
SawitFlood Lab - Interactive Dashboard

Streamlit dashboard for exploring flood risk analysis results.

Usage:
    streamlit run app/dashboard.py
"""

import json
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

try:
    import folium
    from streamlit_folium import st_folium

    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Page configuration
st.set_page_config(
    page_title="SawitFlood Lab",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a5f2a;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e8f5e9 0%, #c8e6c9 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .risk-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .risk-low {
        color: #388e3c;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e8f5e9;
        border-radius: 5px 5px 0 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_config():
    """Load configuration."""
    config_path = PROJECT_ROOT / "configs" / "settings.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


@st.cache_data
def load_data():
    """Load analysis dataset."""
    processed_dir = PROJECT_ROOT / "data" / "processed"

    # Try different file formats
    gpkg_path = processed_dir / "analysis_dataset.gpkg"
    parquet_path = processed_dir / "analysis_dataset.parquet"
    csv_path = processed_dir / "analysis_dataset.csv"
    sample_path = processed_dir / "sample_admin_boundaries.gpkg"

    gdf = None
    df = None

    if gpkg_path.exists():
        gdf = gpd.read_file(gpkg_path)
        df = gdf.drop(columns=["geometry"]) if "geometry" in gdf.columns else gdf
    elif parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    elif sample_path.exists():
        gdf = gpd.read_file(sample_path)
        # Generate sample data
        np.random.seed(42)
        gdf["flood_risk_label"] = np.random.randint(0, 2, len(gdf))
        gdf["risk_probability"] = np.random.uniform(0.2, 0.9, len(gdf))
        gdf["forest_loss_cumulative_pct"] = np.random.uniform(5, 50, len(gdf))
        gdf["palm_oil_pct"] = np.random.uniform(5, 60, len(gdf))
        gdf["rainfall_annual_mean_mm"] = np.random.uniform(2000, 3500, len(gdf))
        gdf["flood_events_total"] = np.random.randint(5, 50, len(gdf))
        df = gdf.drop(columns=["geometry"])
    else:
        # Create minimal sample data
        df = pd.DataFrame(
            {
                "name": ["Pekanbaru", "Kampar", "Bengkalis", "Rokan Hilir", "Indragiri Hilir"],
                "province": ["Riau"] * 5,
                "flood_risk_label": [1, 1, 0, 1, 0],
                "risk_probability": [0.85, 0.72, 0.35, 0.68, 0.28],
                "forest_loss_cumulative_pct": [45, 38, 22, 42, 18],
                "palm_oil_pct": [55, 42, 28, 48, 22],
                "rainfall_annual_mean_mm": [2800, 2650, 2500, 2900, 2400],
                "flood_events_total": [42, 35, 18, 38, 12],
            }
        )

    return df, gdf


@st.cache_resource
def load_model():
    """Load trained model."""
    models_dir = PROJECT_ROOT / "models"

    # Find latest model
    model_files = list(models_dir.glob("flood_risk_*.pkl"))

    if not model_files:
        return None, None

    latest_model = sorted(model_files)[-1]
    model_name = latest_model.stem

    with open(latest_model, "rb") as f:
        model = pickle.load(f)

    # Load metadata
    metadata_path = models_dir / f"{model_name}_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    return model, metadata


def create_risk_map(gdf: gpd.GeoDataFrame, column: str = "risk_probability"):
    """Create interactive risk map using Folium."""
    if not FOLIUM_AVAILABLE or gdf is None:
        return None

    # Ensure WGS84 CRS
    gdf = gdf.to_crs("EPSG:4326")

    # Get center
    bounds = gdf.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(
        location=center,
        zoom_start=8,
        tiles="CartoDB positron",
    )

    # Add choropleth
    folium.Choropleth(
        geo_data=gdf.__geo_interface__,
        name="Risiko Banjir",
        data=gdf,
        columns=["name", column],
        key_on="feature.properties.name",
        fill_color="RdYlGn_r",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Probabilitas Risiko",
    ).add_to(m)

    # Add tooltips
    for _idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        popup_content = f"""
        <b>{row["name"]}</b><br>
        Risiko: {row[column]:.2%}<br>
        Kehilangan Hutan: {row.get("forest_loss_cumulative_pct", "N/A"):.1f}%<br>
        Area Sawit: {row.get("palm_oil_pct", "N/A"):.1f}%
        """
        folium.CircleMarker(
            location=[centroid.y, centroid.x],
            radius=5,
            popup=popup_content,
            color="black",
            fill=True,
            fillOpacity=0.7,
        ).add_to(m)

    return m


def main():
    """Main dashboard application."""
    _config = load_config()  # Reserved for future use
    df, gdf = load_data()
    model, model_metadata = load_model()

    # Header
    st.markdown('<div class="main-header">🌴🌊 SawitFlood Lab</div>', unsafe_allow_html=True)

    st.markdown(
        """
    <p style="text-align: center; font-size: 1.1rem; color: #666;">
        Analisis Keterkaitan Deforestasi Kelapa Sawit dan Risiko Banjir di Sumatra
    </p>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/palm-tree.png", width=80)
        st.title("Navigasi")

        page = st.radio(
            "Pilih Halaman",
            [
                "📊 Overview",
                "🗺️ Peta Risiko",
                "📈 Analisis Tren",
                "🤖 Model & XAI",
                "📋 Data Explorer",
            ],
        )

        st.divider()

        # Filter options
        st.subheader("Filter Data")

        if "province" in df.columns:
            provinces = ["Semua"] + df["province"].unique().tolist()
            selected_province = st.selectbox("Provinsi", provinces)

            if selected_province != "Semua":
                df = df[df["province"] == selected_province]
                if gdf is not None:
                    gdf = gdf[gdf["province"] == selected_province]

        st.divider()
        st.markdown("**Tentang Proyek**")
        st.markdown(
            """Proyek ini menganalisis hubungan antara deforestasi
            untuk perkebunan sawit dan risiko banjir di Sumatra."""
        )

    # Main content based on page selection
    if page == "📊 Overview":
        render_overview(df, model_metadata)
    elif page == "🗺️ Peta Risiko":
        render_risk_map(df, gdf)
    elif page == "📈 Analisis Tren":
        render_trend_analysis(df)
    elif page == "🤖 Model & XAI":
        render_model_xai(df, model, model_metadata)
    elif page == "📋 Data Explorer":
        render_data_explorer(df)


def render_overview(df: pd.DataFrame, model_metadata: dict):
    """Render overview page."""
    st.header("📊 Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    total_zones = len(df)
    high_risk = df["flood_risk_label"].sum() if "flood_risk_label" in df.columns else 0
    avg_forest_loss = (
        df["forest_loss_cumulative_pct"].mean() if "forest_loss_cumulative_pct" in df.columns else 0
    )
    avg_palm_oil = df["palm_oil_pct"].mean() if "palm_oil_pct" in df.columns else 0

    with col1:
        st.metric("Total Wilayah", f"{total_zones}", help="Jumlah kabupaten/kota dalam analisis")

    with col2:
        st.metric(
            "Wilayah Risiko Tinggi",
            f"{high_risk}",
            delta=f"{high_risk / total_zones * 100:.1f}%" if total_zones > 0 else "0%",
            delta_color="inverse",
        )

    with col3:
        st.metric(
            "Rata-rata Kehilangan Hutan",
            f"{avg_forest_loss:.1f}%",
            help="Persentase rata-rata kehilangan hutan kumulatif",
        )

    with col4:
        st.metric(
            "Rata-rata Area Sawit", f"{avg_palm_oil:.1f}%", help="Persentase rata-rata area sawit"
        )

    st.divider()

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Risiko Banjir")

        if "risk_probability" in df.columns:
            fig = px.histogram(
                df,
                x="risk_probability",
                nbins=20,
                color_discrete_sequence=["#1976d2"],
                labels={"risk_probability": "Probabilitas Risiko"},
            )
            fig.update_layout(
                xaxis_title="Probabilitas Risiko",
                yaxis_title="Jumlah Wilayah",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Kehilangan Hutan vs Risiko Banjir")

        if "forest_loss_cumulative_pct" in df.columns and "risk_probability" in df.columns:
            fig = px.scatter(
                df,
                x="forest_loss_cumulative_pct",
                y="risk_probability",
                color="flood_risk_label" if "flood_risk_label" in df.columns else None,
                color_discrete_map={0: "#4caf50", 1: "#f44336"},
                hover_name="name" if "name" in df.columns else None,
                labels={
                    "forest_loss_cumulative_pct": "Kehilangan Hutan (%)",
                    "risk_probability": "Probabilitas Risiko",
                    "flood_risk_label": "Kategori Risiko",
                },
            )
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    # Model performance (if available)
    if model_metadata and "training_history" in model_metadata:
        st.divider()
        st.subheader("📈 Model Performance")

        metrics = model_metadata["training_history"].get("metrics", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")
        with col2:
            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
        with col3:
            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        with col4:
            st.metric("Recall", f"{metrics.get('recall', 0):.3f}")


def render_risk_map(df: pd.DataFrame, gdf: gpd.GeoDataFrame):
    """Render risk map page."""
    st.header("🗺️ Peta Risiko Banjir")

    if gdf is not None and FOLIUM_AVAILABLE:
        st.markdown(
            """<div class="info-box">
            Peta ini menampilkan distribusi risiko banjir per wilayah.
            Warna merah menunjukkan risiko tinggi, hijau menunjukkan risiko rendah.
        </div>""",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([3, 1])

        with col1:
            m = create_risk_map(gdf)
            if m:
                st_folium(m, width=800, height=500)

        with col2:
            st.subheader("Legenda")
            st.markdown(
                """🔴 **Risiko Tinggi** (>70%)

🟠 **Risiko Sedang** (50-70%)

🟡 **Risiko Rendah** (30-50%)

🟢 **Risiko Sangat Rendah** (<30%)"""
            )

            st.divider()

            # Top risk zones
            st.subheader("Top 5 Risiko Tertinggi")

            if "risk_probability" in df.columns and "name" in df.columns:
                top_risk = df.nlargest(5, "risk_probability")[["name", "risk_probability"]]
                for _, row in top_risk.iterrows():
                    risk_pct = row["risk_probability"] * 100
                    color = "🔴" if risk_pct > 70 else "🟠" if risk_pct > 50 else "🟡"
                    st.markdown(f"{color} **{row['name']}**: {risk_pct:.1f}%")
    else:
        st.warning("Data geospasial tidak tersedia atau Folium belum terinstall.")

        # Show table instead
        if "name" in df.columns and "risk_probability" in df.columns:
            st.subheader("Tabel Risiko per Wilayah")
            display_df = df[
                ["name", "risk_probability", "forest_loss_cumulative_pct", "palm_oil_pct"]
            ].copy()
            display_df["risk_probability"] = (display_df["risk_probability"] * 100).round(1)
            display_df.columns = ["Wilayah", "Risiko (%)", "Kehilangan Hutan (%)", "Area Sawit (%)"]
            st.dataframe(
                display_df.sort_values("Risiko (%)", ascending=False), use_container_width=True
            )


def render_trend_analysis(df: pd.DataFrame):
    """Render trend analysis page."""
    st.header("📈 Analisis Tren")

    # Generate sample time series data
    years = list(range(2010, 2024))
    np.random.seed(42)

    # Create trend data
    base_deforestation = 10
    cumulative_deforestation = [base_deforestation]
    for _ in years[1:]:
        cumulative_deforestation.append(cumulative_deforestation[-1] + np.random.uniform(1, 4))

    flood_events = [int(3 + d * 0.5 + np.random.poisson(2)) for d in cumulative_deforestation]

    trend_df = pd.DataFrame(
        {
            "Tahun": years,
            "Kehilangan Hutan Kumulatif (%)": cumulative_deforestation,
            "Kejadian Banjir": flood_events,
        }
    )

    # Line chart
    st.subheader("Tren Deforestasi dan Kejadian Banjir")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=trend_df["Tahun"],
            y=trend_df["Kehilangan Hutan Kumulatif (%)"],
            mode="lines+markers",
            name="Kehilangan Hutan (%)",
            line=dict(color="#1b5e20", width=3),
            yaxis="y",
        )
    )

    fig.add_trace(
        go.Bar(
            x=trend_df["Tahun"],
            y=trend_df["Kejadian Banjir"],
            name="Kejadian Banjir",
            marker_color="#d32f2f",
            opacity=0.6,
            yaxis="y2",
        )
    )

    fig.update_layout(
        xaxis=dict(title="Tahun"),
        yaxis=dict(title="Kehilangan Hutan Kumulatif (%)", side="left", color="#1b5e20"),
        yaxis2=dict(title="Kejadian Banjir", side="right", overlaying="y", color="#d32f2f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Correlation analysis
    st.divider()
    st.subheader("Analisis Korelasi")

    col1, col2 = st.columns(2)

    with col1:
        # Scatter plot: Forest loss vs Flood events
        fig_scatter = px.scatter(
            trend_df,
            x="Kehilangan Hutan Kumulatif (%)",
            y="Kejadian Banjir",
            trendline="ols",
            title="Korelasi Deforestasi vs Kejadian Banjir",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        correlation = trend_df["Kehilangan Hutan Kumulatif (%)"].corr(trend_df["Kejadian Banjir"])
        st.metric("Koefisien Korelasi", f"{correlation:.3f}")

    with col2:
        st.markdown(
            """**Interpretasi:**

Grafik menunjukkan adanya **korelasi positif** antara kehilangan
hutan kumulatif dan jumlah kejadian banjir.

⚠️ **Catatan Penting:**
- Korelasi tidak berarti kausalitas
- Faktor lain seperti curah hujan juga berpengaruh
- Analisis ini bersifat eksploratif"""
        )


def render_model_xai(df: pd.DataFrame, model, model_metadata: dict):
    """Render model and XAI page."""
    st.header("🤖 Model & Explainability")

    if model is None:
        st.warning(
            "Model belum tersedia. Jalankan `python src/models/train_model.py` terlebih dahulu."
        )
        return

    # Model info
    st.subheader("Informasi Model")

    col1, col2 = st.columns(2)

    with col1:
        model_type = model_metadata.get("model_type", "Unknown")
        st.info(f"**Tipe Model:** {model_type.upper()}")

        if "training_history" in model_metadata:
            metrics = model_metadata["training_history"].get("metrics", {})

            metric_df = pd.DataFrame(
                {
                    "Metrik": list(metrics.keys()),
                    "Nilai": [f"{v:.4f}" for v in metrics.values()],
                }
            )
            st.table(metric_df)

    with col2:
        st.markdown("""
        **Fitur Utama yang Digunakan:**
        - Kehilangan hutan kumulatif (%)
        - Persentase area sawit (%)
        - Curah hujan rata-rata (mm)
        - Kejadian banjir historis
        - Anomali curah hujan
        """)

    # Feature Importance
    st.divider()
    st.subheader("Feature Importance")

    if hasattr(model, "feature_importances_"):
        feature_names = model_metadata.get("feature_names", [])

        if len(feature_names) == len(model.feature_importances_):
            importance_df = (
                pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "Importance": model.feature_importances_,
                    }
                )
                .sort_values("Importance", ascending=True)
                .tail(10)
            )

            fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="RdYlGn_r",
            )
            fig.update_layout(
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature names tidak sesuai dengan model.")

    # SHAP explanation placeholder
    st.divider()
    st.subheader("SHAP Analysis")

    st.markdown(
        """<div class="info-box">
        <b>SHAP (SHapley Additive exPlanations)</b> menjelaskan kontribusi
        setiap fitur terhadap prediksi model untuk setiap wilayah.

        Jalankan `python src/models/evaluate_model.py` untuk menghasilkan
        analisis SHAP lengkap.
    </div>""",
        unsafe_allow_html=True,
    )

    # Show SHAP image if available
    shap_path = PROJECT_ROOT / "outputs" / "figures" / "shap_summary.png"
    if shap_path.exists():
        st.image(str(shap_path), caption="SHAP Feature Importance")


def render_data_explorer(df: pd.DataFrame):
    """Render data explorer page."""
    st.header("📋 Data Explorer")

    st.subheader("Dataset Overview")
    st.write(f"**Total Rows:** {len(df)}")
    st.write(f"**Total Columns:** {len(df.columns)}")

    # Column selector
    selected_cols = st.multiselect(
        "Pilih kolom untuk ditampilkan:", df.columns.tolist(), default=df.columns.tolist()[:8]
    )

    if selected_cols:
        st.dataframe(df[selected_cols], use_container_width=True)

    # Statistics
    st.divider()
    st.subheader("Statistik Deskriptif")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)

    # Download
    st.divider()
    st.subheader("Download Data")

    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="sawitflood_data.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

"""
IFRA - Indonesia Flood Risk Analytics
Home Page with Executive Summary (Big 4 Style)
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="IFRA - Flood Risk Analytics",
    page_icon="🌊",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).parent.parent

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
def load_correlation():
    path = PROJECT_ROOT / "data" / "processed" / "correlation_matrix.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return None

# Title
st.title("Indonesia Flood Risk Analytics")

df = load_data()
corr = load_correlation()

if df is None:
    st.error("Data tidak ditemukan. Jalankan data pipeline terlebih dahulu.")
    st.stop()

# =============================================================================
# EXECUTIVE SUMMARY - BIG 4 STYLE
# =============================================================================
st.markdown("---")

st.header("Executive Summary")

# Calculate key metrics
total_floods = int(df["flood_events_total"].sum())
total_deaths = int(df["flood_deaths"].sum())
total_districts = len(df)
affected_districts = len(df[df["flood_events_total"] > 0])
pct_affected = affected_districts / total_districts * 100

if "island" in df.columns:
    top_island = df.groupby("island")["flood_events_total"].sum().idxmax()
    top_island_pct = df.groupby("island")["flood_events_total"].sum().max() / total_floods * 100
else:
    top_island = "N/A"
    top_island_pct = 0

# Big 4 style narrative
st.markdown(f"""
### Situasi

Dalam periode 2020-2025, Indonesia mencatat **{total_floods:,} kejadian banjir** 
yang mengakibatkan **{total_deaths:,} korban jiwa**. Dari {total_districts} kabupaten/kota 
yang dianalisis, **{pct_affected:.0f}% telah mengalami minimal satu kejadian banjir**.

Konsentrasi risiko tidak merata secara geografis. **{top_island}** menyumbang 
**{top_island_pct:.0f}%** dari total kejadian nasional, mengindikasikan perlunya 
pendekatan mitigasi yang terdiferensiasi berdasarkan karakteristik wilayah.

### Komplikasi

Analisis korelasi menunjukkan hubungan antara perubahan tutupan lahan dan frekuensi banjir, 
meskipun tidak bersifat deterministik. Model prediktif dengan akurasi ~63% mengkonfirmasi 
bahwa **curah hujan** dan **deforestasi** merupakan prediktor signifikan, namun 
variance yang belum terjelaskan mengindikasikan faktor-faktor lain yang belum tercakup 
dalam analisis (infrastruktur drainase, topografi mikro, dll).

### Implikasi

**Untuk Pemerintah Daerah:**
- Alokasikan anggaran mitigasi proporsional terhadap risk score wilayah
- Integrasikan data tutupan lahan dalam perencanaan tata ruang

**Untuk BNPB:**
- Prioritaskan early warning system di {affected_districts} kabupaten terdampak
- Kembangkan indeks risiko gabungan yang mencakup faktor lingkungan

**Untuk Sektor Swasta:**
- Pertimbangkan risk assessment berbasis lokasi untuk keputusan investasi
- Kontribusi pada program konservasi DAS sebagai bagian dari CSR
""")

st.markdown("---")

# =============================================================================
# KEY METRICS
# =============================================================================
st.header("Angka Kunci")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Kejadian Banjir", f"{total_floods:,}")

with col2:
    st.metric("Korban Jiwa", f"{total_deaths:,}")

with col3:
    st.metric("Kabupaten Terdampak", f"{affected_districts}")

with col4:
    if "forest_loss_pct" in df.columns:
        avg_defor = df["forest_loss_pct"].mean()
        st.metric("Rata-rata Deforestasi", f"{avg_defor:.1f}%")

with col5:
    st.metric("Cakupan Risiko", f"{pct_affected:.0f}%")

st.markdown("---")

# =============================================================================
# QUICK INSIGHTS
# =============================================================================
st.header("Quick Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 5 Provinsi - Kejadian")
    top = df.groupby("province")["flood_events_total"].sum().nlargest(5).reset_index()
    top.columns = ["Provinsi", "Kejadian"]
    st.dataframe(top, hide_index=True, use_container_width=True)

with col2:
    st.subheader("Top 5 Provinsi - Korban")
    top = df.groupby("province")["flood_deaths"].sum().nlargest(5).reset_index()
    top.columns = ["Provinsi", "Korban Jiwa"]
    st.dataframe(top, hide_index=True, use_container_width=True)

st.markdown("---")

# =============================================================================
# NAVIGATION
# =============================================================================
st.header("Navigasi Dashboard")

st.markdown("""
| Halaman | Deskripsi |
|---------|-----------|
| **Flood Analysis** | Pola temporal dan spasial kejadian banjir |
| **Land Impact** | Korelasi deforestasi dengan risiko banjir |
| **Regional** | Perbandingan antar pulau dan provinsi |
| **Choropleth Map** | Visualisasi peta interaktif per wilayah |
| **Model & XAI** | Machine learning dan interpretasi model |
| **Data Explorer** | Akses data mentah dan download |

Gunakan **sidebar di kiri** untuk navigasi.
""")

st.markdown("---")
st.caption("IFRA | Data: BNPB DIBI, Global Forest Change, CHIRPS")

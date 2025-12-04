"""
Choropleth Map - Peta Interaktif Risiko Banjir
Menggunakan Folium untuk visualisasi geospatial yang proper
"""
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from pathlib import Path

st.set_page_config(page_title="Peta Interaktif", page_icon="🗺️", layout="wide")

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

# Province coordinates for Indonesia
PROVINCE_COORDS = {
    "Aceh": [4.695, 96.749],
    "Sumatera Utara": [2.115, 99.545],
    "Sumatera Barat": [-0.739, 100.800],
    "Riau": [0.293, 101.706],
    "Jambi": [-1.610, 103.607],
    "Sumatera Selatan": [-3.319, 104.914],
    "Bengkulu": [-3.792, 102.260],
    "Lampung": [-4.558, 105.406],
    "Kepulauan Bangka Belitung": [-2.741, 106.440],
    "Kepulauan Riau": [3.945, 108.142],
    "DKI Jakarta": [-6.208, 106.845],
    "Jawa Barat": [-6.914, 107.609],
    "Jawa Tengah": [-7.150, 110.140],
    "DI Yogyakarta": [-7.797, 110.370],
    "Jawa Timur": [-7.536, 112.238],
    "Banten": [-6.405, 106.064],
    "Bali": [-8.409, 115.188],
    "Nusa Tenggara Barat": [-8.650, 117.361],
    "Nusa Tenggara Timur": [-8.657, 121.079],
    "Kalimantan Barat": [-0.278, 111.475],
    "Kalimantan Tengah": [-1.681, 113.382],
    "Kalimantan Selatan": [-3.092, 115.283],
    "Kalimantan Timur": [1.693, 116.419],
    "Kalimantan Utara": [3.073, 116.911],
    "Sulawesi Utara": [0.624, 123.975],
    "Sulawesi Tengah": [-1.430, 121.445],
    "Sulawesi Selatan": [-3.668, 119.974],
    "Sulawesi Tenggara": [-4.145, 122.174],
    "Gorontalo": [0.696, 122.446],
    "Sulawesi Barat": [-2.844, 119.232],
    "Maluku": [-3.238, 130.145],
    "Maluku Utara": [1.570, 127.808],
    "Papua Barat": [-1.336, 133.174],
    "Papua": [-4.269, 138.080],
    "Papua Selatan": [-6.121, 140.303],
    "Papua Tengah": [-3.987, 136.980],
    "Papua Pegunungan": [-4.100, 138.500],
    "Papua Barat Daya": [-1.500, 132.000],
}

def get_color(value, min_val, max_val):
    """Get color based on value (red scale)."""
    if max_val == min_val:
        return "#fee0d2"
    
    ratio = (value - min_val) / (max_val - min_val)
    
    if ratio < 0.2:
        return "#fee0d2"
    elif ratio < 0.4:
        return "#fcbba1"
    elif ratio < 0.6:
        return "#fc9272"
    elif ratio < 0.8:
        return "#fb6a4a"
    else:
        return "#de2d26"

st.title("Peta Interaktif Indonesia")
st.caption("Visualisasi spasial risiko banjir per provinsi dan kabupaten")

df = load_data()

if df is None:
    st.error("Data tidak ditemukan")
    st.stop()

st.markdown("---")

# =============================================================================
# CONTROLS
# =============================================================================
st.header("Pengaturan Peta")

col1, col2, col3 = st.columns(3)

with col1:
    metric_options = {
        "flood_events_total": "Jumlah Banjir",
        "flood_deaths": "Korban Jiwa",
        "forest_loss_pct": "Deforestasi (%)",
    }
    available = {k: v for k, v in metric_options.items() if k in df.columns}
    selected_metric = st.selectbox("Metrik:", list(available.keys()), format_func=lambda x: available[x])

with col2:
    if "island" in df.columns:
        islands = ["Semua Indonesia"] + sorted(df["island"].dropna().unique().tolist())
        selected_island = st.selectbox("Filter Pulau:", islands)
    else:
        selected_island = "Semua Indonesia"

with col3:
    map_style = st.selectbox("Gaya Peta:", ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter"])

# Filter data
map_data = df.copy()
if selected_island != "Semua Indonesia":
    map_data = map_data[map_data["island"] == selected_island]

st.markdown("---")

# =============================================================================
# INTERACTIVE MAP
# =============================================================================
st.header("Peta Provinsi")

# Aggregate to province level
prov_agg = map_data.groupby("province").agg({
    selected_metric: "sum" if selected_metric in ["flood_events_total", "flood_deaths"] else "mean",
    "kabupaten": "count",
    "flood_events_total": "sum",
    "flood_deaths": "sum"
}).reset_index()
prov_agg.columns = ["province", "value", "n_districts", "total_floods", "total_deaths"]

# Add coordinates
prov_agg["lat"] = prov_agg["province"].map(lambda x: PROVINCE_COORDS.get(x, [None, None])[0])
prov_agg["lon"] = prov_agg["province"].map(lambda x: PROVINCE_COORDS.get(x, [None, None])[1])
prov_agg = prov_agg.dropna(subset=["lat", "lon"])

if len(prov_agg) == 0:
    st.warning("Tidak ada data koordinat untuk provinsi yang dipilih")
else:
    # Create map
    center_lat = prov_agg["lat"].mean()
    center_lon = prov_agg["lon"].mean()
    
    tiles = {
        "OpenStreetMap": "OpenStreetMap",
        "CartoDB positron": "CartoDB positron",
        "CartoDB dark_matter": "CartoDB dark_matter"
    }
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles=tiles[map_style]
    )
    
    # Add markers
    min_val = prov_agg["value"].min()
    max_val = prov_agg["value"].max()
    
    for _, row in prov_agg.iterrows():
        color = get_color(row["value"], min_val, max_val)
        
        # Circle size based on value
        radius = 10 + (row["value"] - min_val) / (max_val - min_val + 1) * 30
        
        popup_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4 style="margin: 0; color: #1f2937;">{row['province']}</h4>
            <hr style="margin: 5px 0;">
            <p style="margin: 3px 0;"><b>{available[selected_metric]}:</b> {row['value']:,.1f}</p>
            <p style="margin: 3px 0;"><b>Total Banjir:</b> {int(row['total_floods']):,}</p>
            <p style="margin: 3px 0;"><b>Korban Jiwa:</b> {int(row['total_deaths']):,}</p>
            <p style="margin: 3px 0;"><b>Jumlah Kabupaten:</b> {int(row['n_districts'])}</p>
        </div>
        """
        
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row['province']}: {row['value']:,.1f}"
        ).add_to(m)
    
    # Display map
    st_folium(m, width=None, height=500, use_container_width=True)
    
    # Legend
    st.markdown("""
    **Legenda Warna:**
    - 🔴 Merah tua = Nilai tinggi
    - 🟠 Oranye = Nilai menengah
    - 🟡 Kuning muda = Nilai rendah
    
    *Klik marker untuk melihat detail. Scroll untuk zoom.*
    """)

st.markdown("---")

# =============================================================================
# KABUPATEN MAP
# =============================================================================
st.header("Peta Detail Kabupaten")

st.markdown("Pilih provinsi untuk melihat detail per kabupaten:")

selected_prov = st.selectbox("Pilih Provinsi:", map_data["province"].unique())

kab_data = map_data[map_data["province"] == selected_prov].copy()

if len(kab_data) > 0 and selected_prov in PROVINCE_COORDS:
    prov_lat, prov_lon = PROVINCE_COORDS[selected_prov]
    
    m2 = folium.Map(
        location=[prov_lat, prov_lon],
        zoom_start=8,
        tiles=tiles[map_style]
    )
    
    # Add kabupaten markers with slight offset
    marker_cluster = MarkerCluster().add_to(m2)
    
    for i, (_, row) in enumerate(kab_data.iterrows()):
        # Create offset for visibility
        lat_offset = (i % 5 - 2) * 0.15
        lon_offset = (i // 5 - 2) * 0.15
        
        value = row[selected_metric]
        color = get_color(value, kab_data[selected_metric].min(), kab_data[selected_metric].max())
        
        popup_html = f"""
        <div style="font-family: Arial; width: 180px;">
            <h4 style="margin: 0; color: #1f2937;">{row['kabupaten']}</h4>
            <p style="margin: 3px 0; color: #6b7280;">{row['province']}</p>
            <hr style="margin: 5px 0;">
            <p style="margin: 3px 0;"><b>{available[selected_metric]}:</b> {value:,.1f}</p>
            <p style="margin: 3px 0;"><b>Banjir:</b> {int(row['flood_events_total']):,}</p>
            <p style="margin: 3px 0;"><b>Korban:</b> {int(row['flood_deaths']):,}</p>
        </div>
        """
        
        folium.CircleMarker(
            location=[prov_lat + lat_offset, prov_lon + lon_offset],
            radius=8 + value / (kab_data[selected_metric].max() + 1) * 15,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{row['kabupaten']}: {value:,.0f}"
        ).add_to(marker_cluster)
    
    st_folium(m2, width=None, height=450, use_container_width=True)
    
    # Table
    st.subheader(f"Data Kabupaten di {selected_prov}")
    display_cols = ["kabupaten", "flood_events_total", "flood_deaths"]
    if "forest_loss_pct" in kab_data.columns:
        display_cols.append("forest_loss_pct")
    
    st.dataframe(
        kab_data[display_cols].sort_values("flood_events_total", ascending=False),
        hide_index=True,
        use_container_width=True
    )

st.markdown("---")

# =============================================================================
# SUMMARY
# =============================================================================
st.header("Ringkasan")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Statistik Peta")
    st.metric("Provinsi Ditampilkan", len(prov_agg))
    st.metric("Total Kabupaten", len(map_data))
    st.metric(f"Total {available[selected_metric]}", f"{map_data[selected_metric].sum():,.0f}")

with col2:
    st.subheader("Top 5 Provinsi")
    top5 = prov_agg.nlargest(5, "value")[["province", "value"]]
    top5.columns = ["Provinsi", available[selected_metric]]
    st.dataframe(top5, hide_index=True, use_container_width=True)

st.markdown("---")
st.caption("IFRA | Peta Interaktif dengan Folium")

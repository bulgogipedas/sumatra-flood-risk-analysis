"""
Script untuk mempersiapkan data awal SawitFlood Lab.

1. Load GADM Indonesia dan filter ke Sumatra
2. Load DIBI data dan filter ke banjir
3. Agregasi data banjir per kabupaten
4. Merge dan simpan dataset
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

print("=" * 60)
print("SawitFlood Lab - Data Preparation")
print("=" * 60)

# ============================================================
# STEP 1: Load dan Filter GADM ke Sumatra
# ============================================================
print("\n[1/4] Loading GADM Indonesia...")

gadm_path = DATA_EXTERNAL / "admin_boundaries" / "gadm41_IDN.gpkg"

# List available layers
import fiona

layers = fiona.listlayers(str(gadm_path))
print(f"    Available layers: {layers}")

# Load level 2 (Kabupaten/Kota)
# Try different layer names
level2_layer = None
for layer in layers:
    if "2" in layer or "ADM2" in layer.upper():
        level2_layer = layer
        break

if level2_layer is None:
    level2_layer = layers[-1]  # Usually the last one is the most detailed

print(f"    Loading layer: {level2_layer}")
gdf_all = gpd.read_file(gadm_path, layer=level2_layer)

print(f"    Total kabupaten Indonesia: {len(gdf_all)}")
print(f"    Columns: {list(gdf_all.columns)}")

# Filter ke provinsi Sumatra yang fokus
sumatra_provinces = [
    "Sumatera Utara",
    "Sumatra Utara",
    "North Sumatra",
    "Riau",
    "Jambi",
    "Sumatera Barat",
    "Sumatra Barat",
    "West Sumatra",
    "Aceh",
    "Sumatera Selatan",
    "Sumatra Selatan",
    "South Sumatra",
    "Bengkulu",
    "Lampung",
    "Kepulauan Riau",
    "Riau Islands",
    "Kepulauan Bangka Belitung",
    "Bangka Belitung Islands",
]

# Find province column
prov_col = None
for col in gdf_all.columns:
    if "NAME_1" in col or "PROVINSI" in col.upper():
        prov_col = col
        break

if prov_col is None:
    print("    WARNING: Could not find province column, showing available columns:")
    print(f"    {list(gdf_all.columns)}")
    prov_col = "NAME_1"  # Default guess

print(f"    Using province column: {prov_col}")
print(f"    Unique provinces: {gdf_all[prov_col].unique()[:10]}...")

# Filter to Sumatra
gdf_sumatra = gdf_all[gdf_all[prov_col].isin(sumatra_provinces)].copy()
print(f"    Sumatra kabupaten: {len(gdf_sumatra)}")

# Jika filter tidak berhasil, coba case-insensitive
if len(gdf_sumatra) == 0:
    print("    Trying case-insensitive matching...")
    sumatra_lower = [p.lower() for p in sumatra_provinces]
    gdf_sumatra = gdf_all[gdf_all[prov_col].str.lower().isin(sumatra_lower)].copy()
    print(f"    Sumatra kabupaten (case-insensitive): {len(gdf_sumatra)}")

# Simpan Sumatra subset
sumatra_output = DATA_EXTERNAL / "admin_boundaries" / "sumatra_admin2.gpkg"
gdf_sumatra.to_file(sumatra_output, driver="GPKG")
print(f"    ✅ Saved: {sumatra_output}")

# ============================================================
# STEP 2: Load dan Filter DIBI ke Banjir
# ============================================================
print("\n[2/4] Loading DIBI flood data...")

dibi_path = DATA_RAW / "flood_events" / "dibi_banjir_raw.csv"
df_dibi = pd.read_csv(dibi_path)

print(f"    Total records: {len(df_dibi)}")
print(f"    Columns: {list(df_dibi.columns)}")
print(f"    Years range: {df_dibi['Tahun'].min()} - {df_dibi['Tahun'].max()}")

# Filter hanya banjir
banjir_types = ["Banjir", "Banjir Bandang", "Banjir dan Tanah Longsor"]
df_banjir = df_dibi[df_dibi["Jenis Bencana"].isin(banjir_types)].copy()
print(f"    Flood records only: {len(df_banjir)}")

# Lihat distribusi per provinsi
print("\n    Top 10 provinces by flood events:")
prov_counts = df_banjir.groupby("Provinsi").size().sort_values(ascending=False)
print(prov_counts.head(10))

# ============================================================
# STEP 3: Agregasi per Kabupaten
# ============================================================
print("\n[3/4] Aggregating flood data by kabupaten...")

# Agregasi statistik per kabupaten
flood_agg = (
    df_banjir.groupby(["Kode Kabupaten", "Kabupaten", "Provinsi"])
    .agg(
        {
            "Jumlah Kejadian": "sum",
            "Meninggal": "sum",
            "Hilang": "sum",
            "Luka / Sakit": "sum",
            "Menderita": "sum",
            "Mengungsi": "sum",
            "Rumah Rusak Berat": "sum",
            "Rumah Rusak Sedang": "sum",
            "Rumah Rusak Ringan": "sum",
            "Rumah Terendam": "sum",
            "Tahun": ["min", "max", "count"],  # Period info
        }
    )
    .reset_index()
)

# Flatten column names
flood_agg.columns = [
    "_".join(col).strip("_") if isinstance(col, tuple) else col for col in flood_agg.columns
]

# Rename columns
flood_agg = flood_agg.rename(
    columns={
        "Jumlah Kejadian_sum": "flood_events_total",
        "Meninggal_sum": "deaths_total",
        "Hilang_sum": "missing_total",
        "Luka / Sakit_sum": "injured_total",
        "Menderita_sum": "affected_total",
        "Mengungsi_sum": "displaced_total",
        "Rumah Rusak Berat_sum": "houses_heavy_damage",
        "Rumah Rusak Sedang_sum": "houses_medium_damage",
        "Rumah Rusak Ringan_sum": "houses_light_damage",
        "Rumah Terendam_sum": "houses_flooded",
        "Tahun_min": "year_first_flood",
        "Tahun_max": "year_last_flood",
        "Tahun_count": "years_with_floods",
    }
)

# Hitung impact score komposit
flood_agg["flood_impact_score"] = (
    flood_agg["flood_events_total"] * 1
    + flood_agg["deaths_total"] * 100
    + flood_agg["affected_total"] * 0.01
    + flood_agg["houses_flooded"] * 0.1
)

print(f"    Aggregated to {len(flood_agg)} kabupaten")
print(f"    Columns: {list(flood_agg.columns)}")

# Simpan agregasi
flood_agg_path = DATA_PROCESSED / "flood_events_aggregated.csv"
flood_agg.to_csv(flood_agg_path, index=False)
print(f"    ✅ Saved: {flood_agg_path}")

# ============================================================
# STEP 4: Filter ke provinsi fokus dan buat risk label
# ============================================================
print("\n[4/4] Creating analysis dataset for focus provinces...")

# Provinsi fokus (3 provinsi utama)
focus_provinces = ["Sumatera Utara", "Riau", "Jambi"]

# Filter flood data ke provinsi fokus
flood_focus = flood_agg[flood_agg["Provinsi"].isin(focus_provinces)].copy()
print(f"    Focus provinces flood data: {len(flood_focus)} kabupaten")

# Filter GADM ke provinsi fokus
gdf_focus = gdf_sumatra[gdf_sumatra[prov_col].isin(focus_provinces)].copy()
print(f"    Focus provinces geometry: {len(gdf_focus)} kabupaten")

# Buat flood risk label
# High risk = top 50% by impact score
threshold = flood_focus["flood_impact_score"].quantile(0.5)
flood_focus["flood_risk_label"] = (flood_focus["flood_impact_score"] >= threshold).astype(int)

high_risk = flood_focus["flood_risk_label"].sum()
low_risk = len(flood_focus) - high_risk
print(f"    Risk distribution: High={high_risk}, Low={low_risk}")

# Rename columns untuk matching dengan GADM
flood_focus = flood_focus.rename(
    columns={
        "Kode Kabupaten": "kode_kabupaten",
        "Kabupaten": "kabupaten_name",
        "Provinsi": "province",
    }
)

# Simpan dataset
analysis_path = DATA_PROCESSED / "analysis_dataset.csv"
flood_focus.to_csv(analysis_path, index=False)
print(f"    ✅ Saved: {analysis_path}")

# Simpan juga sebagai parquet
parquet_path = DATA_PROCESSED / "analysis_dataset.parquet"
flood_focus.to_parquet(parquet_path, index=False)
print(f"    ✅ Saved: {parquet_path}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETE!")
print("=" * 60)
print(f"""
Files created:
  1. {sumatra_output.name}
     - Sumatra administrative boundaries (Level 2)
     - {len(gdf_sumatra)} kabupaten/kota
     
  2. {flood_agg_path.name}
     - All Indonesia flood data aggregated by kabupaten
     - {len(flood_agg)} kabupaten/kota
     
  3. {analysis_path.name} / {parquet_path.name}
     - Focus provinces analysis dataset
     - {len(flood_focus)} kabupaten/kota
     - High risk: {high_risk}, Low risk: {low_risk}

Next steps:
  - Run EDA notebook: notebooks/01_eda_data.ipynb
  - Train model: notebooks/02_modeling_risk.ipynb
""")

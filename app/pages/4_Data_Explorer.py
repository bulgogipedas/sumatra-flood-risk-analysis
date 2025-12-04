"""
Data Explorer
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Data Explorer", page_icon="📋", layout="wide")

PROJECT_ROOT = Path(__file__).parent.parent.parent

@st.cache_data
def load_data():
    path = PROJECT_ROOT / "data" / "processed" / "complete_analysis_dataset.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None

st.title("📋 Data Explorer")
st.caption("Browse, filter, and download the dataset")

df = load_data()

if df is None:
    st.error("Data not found")
    st.stop()

st.divider()

# Info
st.header("Dataset Info")

col1, col2, col3 = st.columns(3)
col1.metric("Records", f"{len(df):,}")
col2.metric("Columns", len(df.columns))
col3.metric("Provinces", df["province"].nunique() if "province" in df.columns else 0)

st.divider()

# Filters
st.header("Filters")

col1, col2 = st.columns(2)

with col1:
    if "island" in df.columns:
        islands = ["All"] + sorted(df["island"].dropna().unique().tolist())
        sel_island = st.selectbox("Island", islands)
    else:
        sel_island = "All"

with col2:
    if sel_island != "All" and "island" in df.columns:
        provs = ["All"] + sorted(df[df["island"] == sel_island]["province"].unique().tolist())
    else:
        provs = ["All"] + sorted(df["province"].unique().tolist()) if "province" in df.columns else ["All"]
    sel_prov = st.selectbox("Province", provs)

# Apply
filt = df.copy()
if sel_island != "All" and "island" in filt.columns:
    filt = filt[filt["island"] == sel_island]
if sel_prov != "All" and "province" in filt.columns:
    filt = filt[filt["province"] == sel_prov]

st.info(f"Showing {len(filt):,} of {len(df):,} records")

st.divider()

# Columns
st.header("Select Columns")

default = ["province", "kabupaten", "flood_events_total", "flood_deaths"]
default = [c for c in default if c in df.columns]

cols = st.multiselect("Columns:", df.columns.tolist(), default=default)

if not cols:
    st.warning("Select at least one column")
    st.stop()

st.divider()

# Table
st.header("Data")

display = filt[cols].copy()

# Round numerics
for c in display.select_dtypes(include=[np.number]).columns:
    display[c] = display[c].round(2)

sort_col = st.selectbox("Sort by:", cols)
sort_asc = st.checkbox("Ascending", value=False)
display = display.sort_values(sort_col, ascending=sort_asc)

st.dataframe(display, hide_index=True, use_container_width=True, height=500)

st.divider()

# Download
st.header("Download")

col1, col2 = st.columns(2)

with col1:
    csv = display.to_csv(index=False)
    st.download_button("📥 CSV", csv, "flood_data.csv", "text/csv", use_container_width=True)

with col2:
    json = display.to_json(orient="records", indent=2)
    st.download_button("📥 JSON", json, "flood_data.json", "application/json", use_container_width=True)

st.divider()

# Stats
st.header("Statistics")

nums = display.select_dtypes(include=[np.number]).columns.tolist()
if nums:
    stats = display[nums].describe().T[["count", "mean", "std", "min", "max"]].round(2)
    st.dataframe(stats, use_container_width=True)


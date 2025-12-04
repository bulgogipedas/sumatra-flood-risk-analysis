"""
IFRA - Indonesia Flood Risk Analytics
Home Page
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
        # Only fillna on numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        return df
    return None

# Title
st.title("🌊 Indonesia Flood Risk Analytics")
st.caption("Comprehensive analysis of flood risk factors across Indonesian provinces")

st.divider()

df = load_data()

if df is None:
    st.error("⚠️ Data not found. Please run the data pipeline first.")
    st.code("""
cd sawitflood-lab
uv run python scripts/04_merge_all_data.py
    """)
    st.stop()

# Key Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Flood Events", f"{int(df['flood_events_total'].sum()):,}")

with col2:
    st.metric("Total Deaths", f"{int(df['flood_deaths'].sum()):,}")

with col3:
    if "forest_loss_pct" in df.columns:
        st.metric("Avg Forest Loss", f"{df['forest_loss_pct'].mean():.1f}%")
    else:
        st.metric("Districts", f"{len(df):,}")

with col4:
    st.metric("Districts Analyzed", f"{len(df):,}")

st.divider()

# About
st.header("About This Dashboard")

st.markdown("""
This dashboard analyzes the relationship between **land use changes**, 
**climate patterns**, and **flood risk** across Indonesia.

**Data Sources:**
- 🌊 **BNPB DIBI** - Flood events (2020-2025)
- 🌲 **Global Forest Change** - Deforestation data
- 🌧️ **CHIRPS** - Rainfall data
""")

st.info("👈 Use the **sidebar** to navigate to different analysis pages")

st.divider()

# Quick View
st.header("Top Affected Areas")

col1, col2 = st.columns(2)

with col1:
    st.subheader("By Flood Events")
    top = df.nlargest(5, "flood_events_total")[["province", "kabupaten", "flood_events_total"]]
    top.columns = ["Province", "District", "Events"]
    st.dataframe(top, hide_index=True, use_container_width=True)

with col2:
    st.subheader("By Deaths")
    top = df.nlargest(5, "flood_deaths")[["province", "kabupaten", "flood_deaths"]]
    top.columns = ["Province", "District", "Deaths"]
    st.dataframe(top, hide_index=True, use_container_width=True)

st.divider()
st.caption("IFRA | Data: BNPB, Global Forest Change, CHIRPS")

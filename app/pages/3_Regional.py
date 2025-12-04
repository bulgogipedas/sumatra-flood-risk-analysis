"""
Regional Analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Regional", page_icon="🗺️", layout="wide")

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

st.title("🗺️ Regional Analysis")
st.caption("Comparing flood risk across islands and provinces")

df = load_data()

if df is None:
    st.error("Data not found")
    st.stop()

st.divider()

# Island Overview
st.header("Island Comparison")

if "island" in df.columns:
    isl = df.groupby("island").agg({
        "flood_events_total": "sum",
        "flood_deaths": "sum",
        "kabupaten": "count"
    }).reset_index()
    isl.columns = ["Island", "Floods", "Deaths", "Districts"]
    isl = isl.sort_values("Floods", ascending=False)
    
    # Top 3
    cols = st.columns(3)
    for i, (_, row) in enumerate(isl.head(3).iterrows()):
        with cols[i]:
            st.metric(f"#{i+1} {row['Island']}", f"{int(row['Floods']):,} floods",
                     delta=f"{int(row['Deaths'])} deaths", delta_color="inverse")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(isl.sort_values("Floods"), x="Floods", y="Island", orientation="h",
                    title="Flood Events", color="Floods", color_continuous_scale="Reds")
        fig.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(isl.sort_values("Deaths"), x="Deaths", y="Island", orientation="h",
                    title="Deaths", color="Deaths", color_continuous_scale="Oranges")
        fig.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Full table
    st.subheader("Full Statistics")
    st.dataframe(isl, hide_index=True, use_container_width=True)

st.divider()

# Province Drill-down
st.header("Province Detail")

if "island" in df.columns:
    sel = st.selectbox("Select Island:", df["island"].unique())
    
    prov = df[df["island"] == sel].groupby("province").agg({
        "flood_events_total": "sum",
        "flood_deaths": "sum"
    }).reset_index().sort_values("flood_events_total", ascending=False)
    prov.columns = ["Province", "Floods", "Deaths"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(prov, x="Floods", y="Province", orientation="h",
                    title=f"Provinces in {sel}", color="Floods", color_continuous_scale="Blues")
        fig.update_layout(height=max(300, len(prov)*25), coloraxis_showscale=False,
                         yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(prov, hide_index=True, use_container_width=True, height=400)

st.divider()

# Priority
st.header("Priority Areas")

top = df.nlargest(10, "flood_events_total")[["province", "kabupaten", "island", "flood_events_total", "flood_deaths"]]
top.columns = ["Province", "District", "Island", "Floods", "Deaths"]
st.dataframe(top, hide_index=True, use_container_width=True)

st.warning("⚠️ These districts need priority attention for flood mitigation")


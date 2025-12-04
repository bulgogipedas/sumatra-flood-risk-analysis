"""
Flood Analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Flood Analysis", page_icon="📊", layout="wide")

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

st.title("📊 Flood Analysis")
st.caption("Understanding flood patterns and impacts across Indonesia")

df = load_data()

if df is None:
    st.error("Data not found")
    st.stop()

st.divider()

# Metrics
st.header("Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Events", f"{int(df['flood_events_total'].sum()):,}")
col2.metric("Deaths", f"{int(df['flood_deaths'].sum()):,}")

affected = df['flood_affected'].sum() if 'flood_affected' in df.columns else 0
col3.metric("Affected", f"{int(affected):,}")

pct = len(df[df['flood_events_total'] > 0]) / len(df) * 100
col4.metric("Districts Hit", f"{pct:.0f}%")

st.divider()

# Charts
st.header("Geographic Distribution")

col1, col2 = st.columns(2)

with col1:
    prov = df.groupby("province")["flood_events_total"].sum().reset_index()
    prov = prov.nlargest(15, "flood_events_total").sort_values("flood_events_total")
    
    fig = px.bar(prov, x="flood_events_total", y="province", orientation="h",
                 color="flood_events_total", color_continuous_scale="Blues")
    fig.update_layout(height=500, showlegend=False, coloraxis_showscale=False,
                     xaxis_title="Events", yaxis_title="", title="Top 15 Provinces")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    if "island" in df.columns:
        island = df.groupby("island")["flood_events_total"].sum().reset_index()
        fig = px.pie(island, values="flood_events_total", names="island", hole=0.4,
                    title="By Island")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# Severity
st.header("Severity")

col1, col2 = st.columns(2)

with col1:
    fig = px.scatter(
        df[df["flood_events_total"] > 0],
        x="flood_events_total", y="flood_deaths",
        color="island" if "island" in df.columns else None,
        hover_name="kabupaten", opacity=0.6,
        title="Events vs Deaths"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    deadly = df.nlargest(10, "flood_deaths")[["province", "kabupaten", "flood_deaths"]]
    deadly.columns = ["Province", "District", "Deaths"]
    st.subheader("Most Deadly Districts")
    st.dataframe(deadly, hide_index=True, use_container_width=True, height=350)


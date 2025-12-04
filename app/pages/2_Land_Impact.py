"""
Land Use Impact
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Land Impact", page_icon="🌲", layout="wide")

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
def load_corr():
    path = PROJECT_ROOT / "data" / "processed" / "correlation_matrix.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return None

st.title("🌲 Land Use Impact")
st.caption("Relationship between deforestation and flood risk")

df = load_data()
corr = load_corr()

if df is None:
    st.error("Data not found")
    st.stop()

st.divider()

# Hypothesis
st.header("The Question")
st.markdown("""
**Does deforestation increase flood risk?**

Forests absorb rainwater. When cleared:
- Water absorption ↓
- Surface runoff ↑  
- Flood risk potentially ↑
""")

st.divider()

# Forest Status
st.header("Deforestation Status")

if "forest_loss_pct" in df.columns:
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Forest Loss", f"{df['forest_loss_pct'].mean():.1f}%")
    col2.metric("High Defor Areas (>20%)", len(df[df["forest_loss_pct"] > 20]))
    if "palm_oil_pct" in df.columns:
        col3.metric("Avg Agri Land", f"{df['palm_oil_pct'].mean():.1f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x="forest_loss_pct", nbins=25,
                          title="Deforestation Distribution")
        fig.add_vline(x=df["forest_loss_pct"].mean(), line_dash="dash", line_color="red")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if "island" in df.columns:
            isl = df.groupby("island")["forest_loss_pct"].mean().reset_index()
            isl = isl.sort_values("forest_loss_pct")
            fig = px.bar(isl, x="forest_loss_pct", y="island", orientation="h",
                        title="By Island", color="forest_loss_pct", color_continuous_scale="Greens")
            fig.update_layout(height=350, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Forest data not available")

st.divider()

# Correlation
st.header("Correlation Analysis")

if "forest_loss_pct" in df.columns:
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(df, x="forest_loss_pct", y="flood_events_total",
                        trendline="ols", hover_name="kabupaten",
                        title="Forest Loss vs Floods", opacity=0.5)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if "palm_oil_pct" in df.columns:
            fig = px.scatter(df, x="palm_oil_pct", y="flood_events_total",
                            trendline="ols", hover_name="kabupaten",
                            title="Agri Land vs Floods", opacity=0.5)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation values
    if corr is not None and "forest_loss_pct" in corr.index:
        col1, col2 = st.columns(2)
        
        fc = corr.loc["forest_loss_pct", "flood_events_total"]
        col1.metric("Forest Loss ↔ Floods", f"{fc:.3f}")
        
        if "palm_oil_pct" in corr.index:
            pc = corr.loc["palm_oil_pct", "flood_events_total"]
            col2.metric("Agri Land ↔ Floods", f"{pc:.3f}")

st.divider()

# Comparison
st.header("High vs Low Deforestation")

if "forest_loss_pct" in df.columns:
    median = df["forest_loss_pct"].median()
    high = df[df["forest_loss_pct"] > median]
    low = df[df["forest_loss_pct"] <= median]
    
    col1, col2 = st.columns(2)
    col1.metric("High Defor - Avg Floods", f"{high['flood_events_total'].mean():.1f}")
    col2.metric("Low Defor - Avg Floods", f"{low['flood_events_total'].mean():.1f}")
    
    df_plot = df.copy()
    df_plot["Level"] = df_plot["forest_loss_pct"].apply(lambda x: "High" if x > median else "Low")
    
    fig = px.box(df_plot, x="Level", y="flood_events_total", color="Level",
                title="Flood Distribution by Deforestation Level")
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


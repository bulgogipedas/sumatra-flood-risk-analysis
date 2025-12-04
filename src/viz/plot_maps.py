"""
Map visualization for SawitFlood Lab.

This module handles:
1. Choropleth maps for risk visualization
2. Interactive maps with Folium
3. Trend visualizations

Usage:
    python src/viz/plot_maps.py
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from loguru import logger

try:
    import folium
    from folium import plugins

    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class MapVisualizer:
    """Create map visualizations for flood risk analysis."""

    def __init__(self, config_path: str | None = None):
        """
        Initialize MapVisualizer.

        Args:
            config_path: Path to settings.yaml
        """
        if config_path is None:
            config_path = PROJECT_ROOT / "configs" / "settings.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.processed_dir = PROJECT_ROOT / "data" / "processed"
        self.output_dir = PROJECT_ROOT / "outputs"
        self.figures_dir = self.output_dir / "figures"

        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Color scheme from config
        self.colors = self.config["dashboard"]["color_scheme"]

        # Setup logging
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        logger.add(log_dir / "visualization.log", rotation="10 MB", level="INFO")

        logger.info("Initialized MapVisualizer")

    def load_geo_data(self) -> gpd.GeoDataFrame:
        """
        Load preprocessed geo data.

        Returns:
            GeoDataFrame with zones and risk data
        """
        gpkg_path = self.processed_dir / "analysis_dataset.gpkg"

        if not gpkg_path.exists():
            # Try to create sample data
            sample_path = self.processed_dir / "sample_admin_boundaries.gpkg"
            if sample_path.exists():
                gdf = gpd.read_file(sample_path)
                # Add sample risk data
                np.random.seed(42)
                gdf["flood_risk_label"] = np.random.randint(0, 2, len(gdf))
                gdf["risk_probability"] = np.random.uniform(0.2, 0.9, len(gdf))
                return gdf
            raise FileNotFoundError(f"No geo data found at {gpkg_path}")

        gdf = gpd.read_file(gpkg_path)
        logger.info(f"Loaded geo data with {len(gdf)} zones")
        return gdf

    def plot_risk_map(
        self,
        gdf: gpd.GeoDataFrame | None = None,
        column: str = "risk_probability",
        title: str = "Flood Risk Map",
        cmap: str = "RdYlGn_r",
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Create static choropleth map of flood risk.

        Args:
            gdf: GeoDataFrame with zones (loads if None)
            column: Column to visualize
            title: Map title
            cmap: Colormap name
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if gdf is None:
            gdf = self.load_geo_data()

        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        # Plot choropleth
        gdf.plot(
            column=column,
            cmap=cmap,
            linewidth=0.5,
            edgecolor="white",
            legend=True,
            legend_kwds={
                "label": column.replace("_", " ").title(),
                "orientation": "horizontal",
                "shrink": 0.8,
                "pad": 0.05,
            },
            ax=ax,
        )

        # Add zone labels if small number of zones
        if len(gdf) <= 20 and "name" in gdf.columns:
            for idx, row in gdf.iterrows():
                centroid = row.geometry.centroid
                ax.annotate(
                    row["name"][:10],
                    xy=(centroid.x, centroid.y),
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

        # Add source note
        ax.text(
            0.01,
            0.01,
            "SawitFlood Lab | Data: GFC, BNPB",
            transform=ax.transAxes,
            fontsize=8,
            color="gray",
        )

        plt.tight_layout()

        if save_path is None:
            save_path = self.figures_dir / "risk_map_sumatra.png"

        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved risk map to {save_path}")

        return fig

    def plot_risk_categories(
        self,
        gdf: gpd.GeoDataFrame | None = None,
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Create map with discrete risk categories.

        Args:
            gdf: GeoDataFrame with zones
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if gdf is None:
            gdf = self.load_geo_data()

        # Create risk categories
        gdf = gdf.copy()
        gdf["risk_category"] = pd.cut(
            gdf["risk_probability"],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"],
        )

        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        # Custom colors
        colors = {
            "Rendah": self.colors["low_risk"],
            "Sedang": "#ffeb3b",
            "Tinggi": self.colors["medium_risk"],
            "Sangat Tinggi": self.colors["high_risk"],
        }

        # Plot each category
        for category, color in colors.items():
            subset = gdf[gdf["risk_category"] == category]
            if len(subset) > 0:
                subset.plot(
                    ax=ax,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                    label=category,
                )

        ax.legend(
            title="Kategori Risiko",
            loc="lower left",
            frameon=True,
            fancybox=True,
        )

        ax.set_title("Peta Kategori Risiko Banjir", fontsize=14, fontweight="bold")
        ax.axis("off")

        plt.tight_layout()

        if save_path is None:
            save_path = self.figures_dir / "risk_categories_map.png"

        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved risk categories map to {save_path}")

        return fig

    def create_interactive_map(
        self,
        gdf: gpd.GeoDataFrame | None = None,
        save_path: Path | None = None,
    ) -> Any | None:
        """
        Create interactive Folium map.

        Args:
            gdf: GeoDataFrame with zones
            save_path: Path to save HTML

        Returns:
            Folium map object
        """
        if not FOLIUM_AVAILABLE:
            logger.warning("Folium not available. Install with: pip install folium")
            return None

        if gdf is None:
            gdf = self.load_geo_data()

        # Ensure WGS84 CRS
        gdf = gdf.to_crs("EPSG:4326")

        # Get center point
        center = self.config["dashboard"]["map_center"]

        # Create base map
        m = folium.Map(
            location=[center["lat"], center["lon"]],
            zoom_start=self.config["dashboard"]["map_zoom"],
            tiles="CartoDB positron",
        )

        # Add choropleth layer
        if "risk_probability" in gdf.columns:
            folium.Choropleth(
                geo_data=gdf.__geo_interface__,
                name="Risiko Banjir",
                data=gdf,
                columns=["name" if "name" in gdf.columns else gdf.index.name, "risk_probability"],
                key_on="feature.properties.name" if "name" in gdf.columns else "feature.id",
                fill_color="RdYlGn_r",
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name="Probabilitas Risiko Banjir",
            ).add_to(m)

        # Add hover info
        style_function = lambda x: {
            "fillColor": "#ffffff",
            "color": "#000000",
            "fillOpacity": 0.1,
            "weight": 0.1,
        }

        highlight_function = lambda x: {
            "fillColor": "#000000",
            "color": "#000000",
            "fillOpacity": 0.50,
            "weight": 0.1,
        }

        # Create tooltip
        if "name" in gdf.columns:
            tooltip_fields = ["name"]
            tooltip_aliases = ["Kabupaten:"]

            if "risk_probability" in gdf.columns:
                gdf["risk_pct"] = (gdf["risk_probability"] * 100).round(1)
                tooltip_fields.append("risk_pct")
                tooltip_aliases.append("Risiko (%)")

            folium.features.GeoJson(
                gdf,
                style_function=style_function,
                highlight_function=highlight_function,
                tooltip=folium.features.GeoJsonTooltip(
                    fields=tooltip_fields,
                    aliases=tooltip_aliases,
                    localize=True,
                ),
            ).add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        # Add title
        title_html = """
        <div style="position: fixed; 
                    top: 10px; left: 50px; 
                    z-index: 9999; 
                    background-color: white;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.2);">
            <h4 style="margin: 0;">🌊 SawitFlood Lab - Peta Risiko Banjir</h4>
        </div>
        """
        m.get_root().html.add_child(folium.Element(title_html))

        if save_path is None:
            save_path = self.figures_dir / "interactive_risk_map.html"

        m.save(str(save_path))
        logger.info(f"Saved interactive map to {save_path}")

        return m

    def plot_deforestation_trend(
        self,
        df: pd.DataFrame | None = None,
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Plot deforestation trend over time.

        Args:
            df: DataFrame with trend data
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        logger.info("Creating deforestation trend plot...")

        # Create sample trend data if not provided
        if df is None:
            years = list(range(2010, 2024))
            np.random.seed(42)

            # Sample cumulative deforestation trend
            base_deforestation = 10
            cumulative = [base_deforestation]
            for _ in years[1:]:
                cumulative.append(cumulative[-1] + np.random.uniform(1, 4))

            # Sample flood events
            flood_events = [int(3 + d * 0.5 + np.random.poisson(2)) for d in cumulative]

            df = pd.DataFrame(
                {
                    "year": years,
                    "cumulative_deforestation_pct": cumulative,
                    "flood_events": flood_events,
                }
            )

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot deforestation
        color1 = self.colors["forest"]
        ax1.set_xlabel("Tahun", fontsize=12)
        ax1.set_ylabel("Kehilangan Hutan Kumulatif (%)", color=color1, fontsize=12)
        line1 = ax1.plot(
            df["year"],
            df["cumulative_deforestation_pct"],
            color=color1,
            linewidth=2,
            marker="o",
            label="Kehilangan Hutan",
        )
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.fill_between(
            df["year"],
            df["cumulative_deforestation_pct"],
            alpha=0.3,
            color=color1,
        )

        # Create second y-axis for flood events
        ax2 = ax1.twinx()
        color2 = self.colors["high_risk"]
        ax2.set_ylabel("Kejadian Banjir", color=color2, fontsize=12)
        line2 = ax2.bar(
            df["year"],
            df["flood_events"],
            alpha=0.5,
            color=color2,
            label="Kejadian Banjir",
        )
        ax2.tick_params(axis="y", labelcolor=color2)

        # Title and legend
        ax1.set_title(
            "Tren Deforestasi dan Kejadian Banjir",
            fontsize=14,
            fontweight="bold",
        )

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        ax1.legend(
            lines1 + [line2],
            labels1 + ["Kejadian Banjir"],
            loc="upper left",
        )

        ax1.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = self.figures_dir / "deforestation_flood_trend.png"

        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved trend plot to {save_path}")

        return fig

    def plot_feature_comparison(
        self,
        gdf: gpd.GeoDataFrame | None = None,
        features: list[str] = None,
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Plot feature comparison by risk category.

        Args:
            gdf: GeoDataFrame with zones
            features: Features to compare
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if gdf is None:
            gdf = self.load_geo_data()

        if features is None:
            features = ["forest_loss_cumulative_pct", "palm_oil_pct", "rainfall_annual_mean_mm"]
            features = [f for f in features if f in gdf.columns]

        if len(features) == 0:
            logger.warning("No features available for comparison")
            return None

        # Create risk label if not present
        if "flood_risk_label" not in gdf.columns:
            gdf["flood_risk_label"] = (gdf["risk_probability"] >= 0.5).astype(int)

        gdf["Risk Category"] = gdf["flood_risk_label"].map({0: "Low Risk", 1: "High Risk"})

        fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 5))

        if len(features) == 1:
            axes = [axes]

        colors = {"Low Risk": self.colors["low_risk"], "High Risk": self.colors["high_risk"]}

        for ax, feature in zip(axes, features):
            for category, color in colors.items():
                data = gdf[gdf["Risk Category"] == category][feature]
                ax.hist(
                    data,
                    bins=15,
                    alpha=0.6,
                    label=category,
                    color=color,
                    edgecolor="white",
                )

            ax.set_xlabel(feature.replace("_", " ").title())
            ax.set_ylabel("Jumlah Wilayah")
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle("Distribusi Fitur berdasarkan Kategori Risiko", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path is None:
            save_path = self.figures_dir / "feature_comparison.png"

        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved feature comparison to {save_path}")

        return fig

    def create_all_visualizations(self) -> dict[str, Path]:
        """
        Create all standard visualizations.

        Returns:
            Dictionary mapping visualization name to file path
        """
        logger.info("Creating all visualizations...")

        gdf = self.load_geo_data()

        paths = {}

        # Static maps
        self.plot_risk_map(gdf)
        paths["risk_map"] = self.figures_dir / "risk_map_sumatra.png"

        self.plot_risk_categories(gdf)
        paths["risk_categories"] = self.figures_dir / "risk_categories_map.png"

        # Interactive map
        if FOLIUM_AVAILABLE:
            self.create_interactive_map(gdf)
            paths["interactive_map"] = self.figures_dir / "interactive_risk_map.html"

        # Trend plot
        self.plot_deforestation_trend()
        paths["deforestation_trend"] = self.figures_dir / "deforestation_flood_trend.png"

        # Feature comparison
        self.plot_feature_comparison(gdf)
        paths["feature_comparison"] = self.figures_dir / "feature_comparison.png"

        logger.info(f"Created {len(paths)} visualizations")
        return paths


def main():
    """Main entry point for visualization."""
    parser = argparse.ArgumentParser(description="Create visualizations")
    parser.add_argument(
        "--type",
        type=str,
        choices=["risk_map", "categories", "interactive", "trend", "all"],
        default="all",
        help="Type of visualization to create",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config file")

    args = parser.parse_args()

    viz = MapVisualizer(config_path=args.config)

    if args.type == "all":
        viz.create_all_visualizations()
    elif args.type == "risk_map":
        viz.plot_risk_map()
    elif args.type == "categories":
        viz.plot_risk_categories()
    elif args.type == "interactive":
        viz.create_interactive_map()
    elif args.type == "trend":
        viz.plot_deforestation_trend()


if __name__ == "__main__":
    main()


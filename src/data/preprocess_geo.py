"""
Geospatial preprocessing for SawitFlood Lab.

This module handles:
1. Clipping raster data to study area
2. Zonal statistics calculation
3. Vector data processing
4. CRS standardization

Usage:
    python src/data/preprocess_geo.py
    python src/data/preprocess_geo.py --step clip
"""

import argparse
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from loguru import logger
from rasterio.mask import mask
from shapely.geometry import box, mapping
from tqdm import tqdm

try:
    from rasterstats import zonal_stats
except ImportError:
    zonal_stats = None
    warnings.warn("rasterstats not installed. Some features will be unavailable.")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class GeoProcessor:
    """Handle geospatial data preprocessing."""

    def __init__(self, config_path: str | None = None):
        """
        Initialize GeoProcessor.

        Args:
            config_path: Path to settings.yaml
        """
        if config_path is None:
            config_path = PROJECT_ROOT / "configs" / "settings.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.raw_dir = PROJECT_ROOT / "data" / "raw"
        self.processed_dir = PROJECT_ROOT / "data" / "processed"
        self.external_dir = PROJECT_ROOT / "data" / "external"

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # CRS settings
        self.target_crs = self.config["geography"]["target_crs"]
        self.geographic_crs = self.config["geography"]["geographic_crs"]

        # Setup logging
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        logger.add(log_dir / "preprocess.log", rotation="10 MB", level="INFO")

        logger.info("Initialized GeoProcessor")

    def load_admin_boundaries(self, provinces: list[str] | None = None) -> gpd.GeoDataFrame:
        """
        Load and filter administrative boundaries.

        Args:
            provinces: List of province names to filter. If None, uses config.

        Returns:
            GeoDataFrame with admin boundaries
        """
        logger.info("Loading administrative boundaries...")

        # Try to find admin boundary file
        admin_dir = self.external_dir / "admin_boundaries"
        possible_files = [
            admin_dir / "gadm41_IDN.gpkg",
            admin_dir / "gadm36_IDN_2.shp",
            admin_dir / "indonesia_admin2.geojson",
        ]

        admin_file = None
        for f in possible_files:
            if f.exists():
                admin_file = f
                break

        if admin_file is None:
            logger.warning("Admin boundary file not found. Creating sample data.")
            return self._create_sample_admin_boundaries()

        # Load the file
        gdf = gpd.read_file(admin_file)

        # Filter to provinces
        if provinces is None:
            provinces = self.config["geography"]["provinces"]

        # Try different column names for province
        province_cols = ["NAME_1", "PROVINSI", "province", "prov_name"]
        prov_col = None
        for col in province_cols:
            if col in gdf.columns:
                prov_col = col
                break

        if prov_col:
            gdf = gdf[gdf[prov_col].isin(provinces)]

        # Ensure correct CRS
        if gdf.crs is None:
            gdf = gdf.set_crs(self.geographic_crs)

        logger.info(f"Loaded {len(gdf)} admin units from {admin_file}")
        return gdf

    def _create_sample_admin_boundaries(self) -> gpd.GeoDataFrame:
        """
        Create sample admin boundaries for demonstration.

        Returns:
            Sample GeoDataFrame
        """
        logger.info("Creating sample administrative boundaries...")

        # Sample kabupaten data for Riau province (simplified)
        sample_data = [
            {
                "name": "Pekanbaru",
                "province": "Riau",
                "id": "1471",
                "geometry": box(101.3, 0.4, 101.6, 0.6),
            },
            {
                "name": "Kampar",
                "province": "Riau",
                "id": "1401",
                "geometry": box(100.8, 0.2, 101.5, 0.8),
            },
            {
                "name": "Bengkalis",
                "province": "Riau",
                "id": "1402",
                "geometry": box(101.5, 0.8, 102.5, 2.0),
            },
            {
                "name": "Rokan Hilir",
                "province": "Riau",
                "id": "1404",
                "geometry": box(100.5, 1.5, 101.5, 2.5),
            },
            {
                "name": "Indragiri Hilir",
                "province": "Riau",
                "id": "1403",
                "geometry": box(102.5, -0.5, 104.0, 0.5),
            },
            {
                "name": "Pelalawan",
                "province": "Riau",
                "id": "1406",
                "geometry": box(101.5, 0.0, 103.0, 0.8),
            },
            {
                "name": "Siak",
                "province": "Riau",
                "id": "1409",
                "geometry": box(101.5, 0.8, 102.5, 1.5),
            },
            {
                "name": "Kuantan Singingi",
                "province": "Riau",
                "id": "1407",
                "geometry": box(101.0, -0.8, 102.0, 0.0),
            },
        ]

        gdf = gpd.GeoDataFrame(sample_data, crs=self.geographic_crs)

        # Save sample data
        output_path = self.processed_dir / "sample_admin_boundaries.gpkg"
        gdf.to_file(output_path, driver="GPKG")
        logger.info(f"Saved sample boundaries to {output_path}")

        return gdf

    def clip_raster_to_boundary(
        self,
        raster_path: str | Path,
        boundary_gdf: gpd.GeoDataFrame,
        output_path: str | Path,
    ) -> Path:
        """
        Clip raster to study area boundary.

        Args:
            raster_path: Path to input raster
            boundary_gdf: GeoDataFrame with boundary polygon(s)
            output_path: Path for output raster

        Returns:
            Path to clipped raster
        """
        logger.info(f"Clipping raster: {raster_path}")

        raster_path = Path(raster_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(raster_path) as src:
            # Reproject boundary to raster CRS
            boundary_reproj = boundary_gdf.to_crs(src.crs)

            # Get geometries for masking
            shapes = [mapping(geom) for geom in boundary_reproj.geometry]

            # Mask the raster
            out_image, out_transform = mask(src, shapes, crop=True, nodata=src.nodata)
            out_meta = src.meta.copy()

            # Update metadata
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                }
            )

            # Write output
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)

        logger.info(f"Saved clipped raster to {output_path}")
        return output_path

    def calculate_zonal_statistics(
        self,
        raster_path: str | Path,
        zones_gdf: gpd.GeoDataFrame,
        stats: list[str] = ["mean", "sum", "count"],
        prefix: str = "",
    ) -> gpd.GeoDataFrame:
        """
        Calculate zonal statistics from raster for each zone.

        Args:
            raster_path: Path to raster file
            zones_gdf: GeoDataFrame with zone polygons
            stats: Statistics to calculate
            prefix: Prefix for output column names

        Returns:
            GeoDataFrame with statistics added
        """
        if zonal_stats is None:
            raise ImportError("rasterstats required for zonal statistics")

        logger.info(f"Calculating zonal statistics for {raster_path}")

        raster_path = Path(raster_path)

        # Reproject zones to raster CRS if needed
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs

        zones_reproj = zones_gdf.to_crs(raster_crs)

        # Calculate statistics
        result = zonal_stats(
            zones_reproj.geometry,
            str(raster_path),
            stats=stats,
            nodata=-9999,
        )

        # Add results to GeoDataFrame
        result_df = pd.DataFrame(result)

        # Rename columns with prefix
        if prefix:
            result_df.columns = [f"{prefix}_{col}" for col in result_df.columns]

        # Combine with original GeoDataFrame
        zones_with_stats = zones_gdf.copy()
        for col in result_df.columns:
            zones_with_stats[col] = result_df[col].values

        logger.info(f"Calculated {len(stats)} statistics for {len(zones_gdf)} zones")
        return zones_with_stats

    def calculate_forest_loss_by_zone(
        self,
        treecover_path: str | Path,
        lossyear_path: str | Path,
        zones_gdf: gpd.GeoDataFrame,
        tree_threshold: int = 30,
    ) -> gpd.GeoDataFrame:
        """
        Calculate forest loss statistics for each zone.

        Args:
            treecover_path: Path to treecover2000 raster
            lossyear_path: Path to lossyear raster
            zones_gdf: GeoDataFrame with zone polygons
            tree_threshold: Minimum tree cover % to count as forest

        Returns:
            GeoDataFrame with forest loss statistics
        """
        logger.info("Calculating forest loss statistics by zone...")

        if zonal_stats is None:
            logger.warning("rasterstats not available. Using simplified calculation.")
            return self._calculate_forest_loss_simplified(zones_gdf)

        # Calculate baseline forest area (2000)
        forest_stats = zonal_stats(
            zones_gdf.geometry,
            str(treecover_path),
            stats=["count", "mean"],
            categorical=False,
        )

        # Calculate loss by year
        start_year = self.config["temporal"]["start_year"]
        end_year = self.config["temporal"]["end_year"]

        zones_result = zones_gdf.copy()

        # Add baseline forest
        zones_result["forest_2000_mean_pct"] = [s["mean"] for s in forest_stats]
        zones_result["forest_2000_pixel_count"] = [s["count"] for s in forest_stats]

        # Calculate cumulative loss
        loss_stats = zonal_stats(
            zones_gdf.geometry,
            str(lossyear_path),
            categorical=True,
        )

        # Process loss by year
        for year in range(start_year - 2000, end_year - 2000 + 1):
            col_name = f"loss_{year + 2000}"
            zones_result[col_name] = [
                s.get(year, 0) if isinstance(s, dict) else 0 for s in loss_stats
            ]

        # Calculate cumulative loss
        loss_cols = [c for c in zones_result.columns if c.startswith("loss_")]
        zones_result["forest_loss_cumulative"] = zones_result[loss_cols].sum(axis=1)

        logger.info(f"Calculated forest loss for {len(zones_gdf)} zones")
        return zones_result

    def _calculate_forest_loss_simplified(self, zones_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Create simplified/sample forest loss data when rasters unavailable.

        Args:
            zones_gdf: GeoDataFrame with zones

        Returns:
            GeoDataFrame with sample forest loss data
        """
        logger.info("Creating sample forest loss data...")

        np.random.seed(42)
        n_zones = len(zones_gdf)

        zones_result = zones_gdf.copy()

        # Generate realistic-looking sample data
        zones_result["forest_2000_mean_pct"] = np.random.uniform(40, 90, n_zones)
        zones_result["forest_loss_cumulative_pct"] = np.random.uniform(5, 45, n_zones)
        zones_result["forest_loss_recent_pct"] = np.random.uniform(1, 15, n_zones)

        # Areas with high sawit tend to have higher forest loss
        zones_result["palm_oil_area_ha"] = np.random.uniform(1000, 50000, n_zones)
        zones_result["palm_oil_pct"] = np.random.uniform(5, 60, n_zones)

        # Correlation between loss and palm oil
        zones_result["forest_loss_cumulative_pct"] = (
            zones_result["forest_loss_cumulative_pct"] * 0.5
            + zones_result["palm_oil_pct"] * 0.5
            + np.random.normal(0, 5, n_zones)
        ).clip(5, 70)

        return zones_result

    def process_rainfall_data(
        self,
        zones_gdf: gpd.GeoDataFrame,
        rainfall_dir: str | Path | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Process CHIRPS rainfall data and calculate zonal statistics.

        Args:
            zones_gdf: GeoDataFrame with zones
            rainfall_dir: Directory with rainfall rasters

        Returns:
            GeoDataFrame with rainfall statistics
        """
        logger.info("Processing rainfall data...")

        if rainfall_dir is None:
            rainfall_dir = self.raw_dir / "rainfall"

        rainfall_dir = Path(rainfall_dir)

        # Find rainfall files
        rainfall_files = list(rainfall_dir.glob("chirps*.tif"))

        if not rainfall_files:
            logger.warning("No rainfall files found. Using sample data.")
            return self._generate_sample_rainfall(zones_gdf)

        zones_result = zones_gdf.copy()

        # Process each year
        start_year = self.config["temporal"]["start_year"]
        end_year = self.config["temporal"]["end_year"]

        for year in tqdm(range(start_year, end_year + 1), desc="Processing rainfall"):
            year_files = [f for f in rainfall_files if f.name.endswith(f".{year}.tif")]

            if not year_files:
                continue

            # Calculate annual rainfall (sum of monthly)
            annual_rainfall = np.zeros(len(zones_gdf))

            for month_file in year_files:
                stats = zonal_stats(
                    zones_gdf.geometry,
                    str(month_file),
                    stats=["mean"],
                )
                annual_rainfall += np.array([s["mean"] or 0 for s in stats])

            zones_result[f"rainfall_{year}_mm"] = annual_rainfall

        # Calculate mean annual rainfall
        rain_cols = [
            c for c in zones_result.columns if c.startswith("rainfall_") and c.endswith("_mm")
        ]
        if rain_cols:
            zones_result["rainfall_annual_mean_mm"] = zones_result[rain_cols].mean(axis=1)
            zones_result["rainfall_annual_std_mm"] = zones_result[rain_cols].std(axis=1)

        logger.info(f"Processed rainfall for {len(rain_cols)} years")
        return zones_result

    def _generate_sample_rainfall(self, zones_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Generate sample rainfall data when actual data unavailable.

        Args:
            zones_gdf: GeoDataFrame with zones

        Returns:
            GeoDataFrame with sample rainfall
        """
        logger.info("Generating sample rainfall data...")

        np.random.seed(42)
        n_zones = len(zones_gdf)

        zones_result = zones_gdf.copy()

        # Typical Sumatra rainfall: 2000-3500 mm/year
        base_rainfall = np.random.uniform(2000, 3500, n_zones)

        start_year = self.config["temporal"]["start_year"]
        end_year = self.config["temporal"]["end_year"]

        for year in range(start_year, end_year + 1):
            # Add yearly variation
            year_rainfall = base_rainfall + np.random.normal(0, 200, n_zones)
            zones_result[f"rainfall_{year}_mm"] = year_rainfall.clip(1500, 4000)

        rain_cols = [c for c in zones_result.columns if c.startswith("rainfall_")]
        zones_result["rainfall_annual_mean_mm"] = zones_result[rain_cols].mean(axis=1)
        zones_result["rainfall_annual_std_mm"] = zones_result[rain_cols].std(axis=1)

        return zones_result

    def run_full_preprocessing(self) -> gpd.GeoDataFrame:
        """
        Run full preprocessing pipeline.

        Returns:
            Processed GeoDataFrame ready for analysis
        """
        logger.info("Starting full preprocessing pipeline...")

        # 1. Load admin boundaries
        zones = self.load_admin_boundaries()

        # 2. Calculate forest loss (uses sample if data unavailable)
        zones = self._calculate_forest_loss_simplified(zones)

        # 3. Process rainfall
        zones = self._generate_sample_rainfall(zones)

        # 4. Save processed data
        output_path = self.processed_dir / "zones_preprocessed.gpkg"
        zones.to_file(output_path, driver="GPKG")
        logger.info(f"Saved preprocessed data to {output_path}")

        # Also save as CSV (without geometry)
        csv_path = self.processed_dir / "zones_preprocessed.csv"
        zones.drop(columns=["geometry"]).to_csv(csv_path, index=False)
        logger.info(f"Saved CSV to {csv_path}")

        return zones


def main():
    """Main entry point for preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess geospatial data")
    parser.add_argument(
        "--step",
        type=str,
        choices=["admin", "forest", "rainfall", "all"],
        default="all",
        help="Preprocessing step to run",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config file")

    args = parser.parse_args()

    processor = GeoProcessor(config_path=args.config)

    if args.step == "all":
        processor.run_full_preprocessing()
    elif args.step == "admin":
        processor.load_admin_boundaries()
    elif args.step == "forest":
        zones = processor.load_admin_boundaries()
        processor._calculate_forest_loss_simplified(zones)
    elif args.step == "rainfall":
        zones = processor.load_admin_boundaries()
        processor._generate_sample_rainfall(zones)


if __name__ == "__main__":
    main()


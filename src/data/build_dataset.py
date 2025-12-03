"""
Build analysis dataset for SawitFlood Lab.

This module combines all preprocessed data into a single analysis-ready dataset
with features and labels for flood risk classification.

Usage:
    python src/data/build_dataset.py
    python src/data/build_dataset.py --update
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DatasetBuilder:
    """Build analysis dataset from preprocessed data."""

    def __init__(self, config_path: str | None = None):
        """
        Initialize DatasetBuilder.

        Args:
            config_path: Path to settings.yaml
        """
        if config_path is None:
            config_path = PROJECT_ROOT / "configs" / "settings.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.raw_dir = PROJECT_ROOT / "data" / "raw"
        self.processed_dir = PROJECT_ROOT / "data" / "processed"

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        logger.add(log_dir / "build_dataset.log", rotation="10 MB", level="INFO")

        logger.info("Initialized DatasetBuilder")

    def load_preprocessed_zones(self) -> gpd.GeoDataFrame:
        """
        Load preprocessed zone data.

        Returns:
            GeoDataFrame with preprocessed zones
        """
        zones_path = self.processed_dir / "zones_preprocessed.gpkg"

        if not zones_path.exists():
            logger.warning("Preprocessed zones not found. Running preprocessing...")
            from src.data.preprocess_geo import GeoProcessor

            processor = GeoProcessor()
            return processor.run_full_preprocessing()

        zones = gpd.read_file(zones_path)
        logger.info(f"Loaded {len(zones)} preprocessed zones")
        return zones

    def load_flood_events(self) -> pd.DataFrame:
        """
        Load flood event data.

        Returns:
            DataFrame with flood events
        """
        flood_dir = self.raw_dir / "flood_events"
        flood_files = list(flood_dir.glob("*.csv"))

        if not flood_files:
            logger.warning("No flood data files found. Generating sample data.")
            return self._generate_sample_flood_data()

        # Combine all flood data files
        dfs = []
        for f in flood_files:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Could not load {f}: {e}")

        if not dfs:
            return self._generate_sample_flood_data()

        flood_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(flood_df)} flood event records")
        return flood_df

    def _generate_sample_flood_data(self) -> pd.DataFrame:
        """
        Generate sample flood event data.

        Returns:
            DataFrame with sample flood data
        """
        logger.info("Generating sample flood event data...")

        np.random.seed(42)

        # Sample kabupaten data
        kabupatens = [
            {"name": "Pekanbaru", "province": "Riau", "id": "1471"},
            {"name": "Kampar", "province": "Riau", "id": "1401"},
            {"name": "Bengkalis", "province": "Riau", "id": "1402"},
            {"name": "Rokan Hilir", "province": "Riau", "id": "1404"},
            {"name": "Indragiri Hilir", "province": "Riau", "id": "1403"},
            {"name": "Pelalawan", "province": "Riau", "id": "1406"},
            {"name": "Siak", "province": "Riau", "id": "1409"},
            {"name": "Kuantan Singingi", "province": "Riau", "id": "1407"},
        ]

        start_year = self.config["temporal"]["start_year"]
        end_year = self.config["temporal"]["end_year"]

        records = []
        for kab in kabupatens:
            # Base flood risk varies by location
            base_risk = np.random.uniform(0.2, 0.8)

            for year in range(start_year, end_year + 1):
                # Increasing trend over time
                year_factor = (year - start_year) / (end_year - start_year)

                # Number of flood events
                expected_events = base_risk * (3 + 5 * year_factor)
                events = max(0, int(np.random.poisson(expected_events)))

                # Impact metrics
                if events > 0:
                    affected = int(events * np.random.uniform(100, 500))
                    deaths = int(np.random.poisson(events * 0.3))
                    houses = int(events * np.random.uniform(20, 100))
                else:
                    affected = deaths = houses = 0

                records.append(
                    {
                        "province": kab["province"],
                        "kabupaten": kab["name"],
                        "kabupaten_id": kab["id"],
                        "year": year,
                        "flood_events": events,
                        "affected_people": affected,
                        "deaths": deaths,
                        "houses_damaged": houses,
                    }
                )

        flood_df = pd.DataFrame(records)

        # Save sample data
        output_path = self.raw_dir / "flood_events" / "generated_flood_data.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        flood_df.to_csv(output_path, index=False)
        logger.info(f"Saved sample flood data to {output_path}")

        return flood_df

    def aggregate_flood_by_zone(
        self,
        flood_df: pd.DataFrame,
        zones_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """
        Aggregate flood data by zone.

        Args:
            flood_df: DataFrame with flood events
            zones_gdf: GeoDataFrame with zones

        Returns:
            DataFrame with aggregated flood statistics per zone
        """
        logger.info("Aggregating flood data by zone...")

        # Aggregate by zone
        zone_flood = (
            flood_df.groupby("kabupaten_id")
            .agg(
                {
                    "flood_events": ["sum", "mean", "max"],
                    "affected_people": ["sum", "mean"],
                    "deaths": "sum",
                    "houses_damaged": "sum",
                }
            )
            .reset_index()
        )

        # Flatten column names
        zone_flood.columns = [
            "_".join(col).strip("_") if isinstance(col, tuple) else col
            for col in zone_flood.columns
        ]

        # Rename for clarity
        zone_flood = zone_flood.rename(
            columns={
                "flood_events_sum": "flood_events_total",
                "flood_events_mean": "flood_events_annual_avg",
                "flood_events_max": "flood_events_max_year",
                "affected_people_sum": "affected_total",
                "affected_people_mean": "affected_annual_avg",
                "deaths_sum": "deaths_total",
                "houses_damaged_sum": "houses_damaged_total",
            }
        )

        logger.info(f"Aggregated flood data for {len(zone_flood)} zones")
        return zone_flood

    def create_flood_risk_label(
        self,
        df: pd.DataFrame,
        method: str = "threshold",
    ) -> pd.Series:
        """
        Create binary flood risk label.

        Args:
            df: DataFrame with flood statistics
            method: Method to create labels ('threshold', 'quantile', 'composite')

        Returns:
            Series with binary risk labels
        """
        logger.info(f"Creating flood risk labels using method: {method}")

        threshold_config = self.config["analysis"]["flood_risk_threshold"]

        if method == "threshold":
            # Use configured thresholds
            high_events = df["flood_events_annual_avg"] >= threshold_config["high_risk_min_events"]
            high_impact = df["affected_annual_avg"] >= threshold_config["high_risk_min_impact"]

            risk_label = (high_events | high_impact).astype(int)

        elif method == "quantile":
            # Top 25% are high risk
            threshold = df["flood_events_total"].quantile(0.75)
            risk_label = (df["flood_events_total"] >= threshold).astype(int)

        elif method == "composite":
            # Composite score
            score = (
                df["flood_events_total"].rank(pct=True) * 0.4
                + df["affected_total"].rank(pct=True) * 0.3
                + df["deaths_total"].rank(pct=True) * 0.3
            )
            risk_label = (score >= 0.5).astype(int)

        else:
            raise ValueError(f"Unknown method: {method}")

        high_risk_count = risk_label.sum()
        logger.info(
            f"Created labels: {high_risk_count} high risk, {len(risk_label) - high_risk_count} low risk"
        )

        return risk_label

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features for modeling.

        Args:
            df: DataFrame with raw features

        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")

        df = df.copy()

        # Interaction features
        if "forest_loss_cumulative_pct" in df.columns and "palm_oil_pct" in df.columns:
            df["forest_loss_x_palm"] = df["forest_loss_cumulative_pct"] * df["palm_oil_pct"]

        # Ratio features
        if "forest_loss_recent_pct" in df.columns and "forest_loss_cumulative_pct" in df.columns:
            df["recent_loss_ratio"] = (
                df["forest_loss_recent_pct"] / df["forest_loss_cumulative_pct"].replace(0, np.nan)
            ).fillna(0)

        # Rainfall anomaly
        if "rainfall_annual_mean_mm" in df.columns and "rainfall_annual_std_mm" in df.columns:
            # Normalize rainfall
            mean_rainfall = df["rainfall_annual_mean_mm"].mean()
            df["rainfall_anomaly"] = (df["rainfall_annual_mean_mm"] - mean_rainfall) / mean_rainfall

        # Log transforms for skewed features
        for col in ["palm_oil_area_ha", "flood_events_total", "affected_total"]:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col])

        logger.info(f"Engineered features. Total columns: {len(df.columns)}")
        return df

    def build_analysis_dataset(
        self,
        include_geometry: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Build complete analysis dataset.

        Args:
            include_geometry: Whether to include geometry column

        Returns:
            GeoDataFrame ready for analysis
        """
        logger.info("Building analysis dataset...")

        # 1. Load preprocessed zones
        zones = self.load_preprocessed_zones()

        # 2. Load and aggregate flood data
        flood_df = self.load_flood_events()
        flood_agg = self.aggregate_flood_by_zone(flood_df, zones)

        # 3. Merge zone features with flood data
        # Match on name or id
        if "id" in zones.columns:
            zones["kabupaten_id"] = zones["id"]
        elif "name" in zones.columns:
            # Create ID from name
            zones["kabupaten_id"] = zones["name"].apply(
                lambda x: flood_agg[flood_agg["kabupaten_id"].str.contains(str(x)[:3], na=False)][
                    "kabupaten_id"
                ].iloc[0]
                if len(flood_agg[flood_agg["kabupaten_id"].str.contains(str(x)[:3], na=False)]) > 0
                else None
            )

        # Merge
        dataset = zones.merge(flood_agg, on="kabupaten_id", how="left")

        # Fill NaN flood data with 0
        flood_cols = [c for c in flood_agg.columns if c != "kabupaten_id"]
        for col in flood_cols:
            if col in dataset.columns:
                dataset[col] = dataset[col].fillna(0)

        # 4. Create flood risk label
        dataset["flood_risk_label"] = self.create_flood_risk_label(dataset, method="composite")

        # 5. Engineer features
        dataset = self.engineer_features(dataset)

        # 6. Save dataset
        output_path = self.processed_dir / "analysis_dataset.gpkg"
        dataset.to_file(output_path, driver="GPKG")
        logger.info(f"Saved analysis dataset to {output_path}")

        # Also save as CSV
        csv_path = self.processed_dir / "analysis_dataset.csv"
        df_csv = dataset.drop(columns=["geometry"]) if "geometry" in dataset.columns else dataset
        df_csv.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV to {csv_path}")

        # Also save as Parquet for efficient loading
        parquet_path = self.processed_dir / "analysis_dataset.parquet"
        df_csv.to_parquet(parquet_path, index=False)
        logger.info(f"Saved Parquet to {parquet_path}")

        # Print summary
        self._print_dataset_summary(dataset)

        return dataset

    def _print_dataset_summary(self, df: pd.DataFrame) -> None:
        """Print summary statistics of the dataset."""
        logger.info("\n" + "=" * 60)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total zones: {len(df)}")
        logger.info(f"Total features: {len(df.columns)}")

        if "flood_risk_label" in df.columns:
            high_risk = df["flood_risk_label"].sum()
            logger.info(f"High risk zones: {high_risk} ({high_risk / len(df) * 100:.1f}%)")
            logger.info(
                f"Low risk zones: {len(df) - high_risk} ({(len(df) - high_risk) / len(df) * 100:.1f}%)"
            )

        logger.info("\nFeature columns:")
        feature_cols = [
            c for c in df.columns if c not in ["geometry", "kabupaten_id", "id", "name", "province"]
        ]
        for col in feature_cols[:15]:
            if df[col].dtype in [np.float64, np.int64]:
                logger.info(f"  {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")

        if len(feature_cols) > 15:
            logger.info(f"  ... and {len(feature_cols) - 15} more features")

        logger.info("=" * 60)

    def get_feature_list(self) -> list[str]:
        """
        Get list of feature columns for modeling.

        Returns:
            List of feature column names
        """
        return self.config["analysis"]["features"]


def main():
    """Main entry point for dataset building."""
    parser = argparse.ArgumentParser(description="Build analysis dataset")
    parser.add_argument(
        "--update", action="store_true", help="Update existing dataset with new data"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config file")

    args = parser.parse_args()

    builder = DatasetBuilder(config_path=args.config)
    builder.build_analysis_dataset()


if __name__ == "__main__":
    main()

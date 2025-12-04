"""
Merge All Datasets for Complete Analysis

This script combines:
1. BNPB DIBI Flood Events data
2. Global Forest Change (forest cover & loss)
3. Palm Oil plantation data
4. CHIRPS Rainfall data

Output: Complete analysis dataset with all features for modeling
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_flood_data() -> pd.DataFrame:
    """Load and aggregate BNPB DIBI flood data by kabupaten."""
    logger.info("Loading flood data...")
    
    flood_path = PROJECT_ROOT / "data" / "raw" / "flood_events" / "dibi_banjir_raw.csv"
    
    if not flood_path.exists():
        logger.error(f"Flood data not found: {flood_path}")
        return None
    
    df = pd.read_csv(flood_path, low_memory=False)
    
    # Rename columns
    df = df.rename(columns={
        "Provinsi": "province",
        "Kabupaten": "kabupaten",
        "Tahun": "year",
        "Jumlah Kejadian": "events",
        "Meninggal": "deaths",
        "Hilang": "missing",
        "Luka / Sakit": "injured",
        "Menderita": "affected",
        "Mengungsi": "displaced",
        "Rumah Rusak Berat": "houses_heavy",
        "Rumah Rusak Sedang": "houses_medium",
        "Rumah Rusak Ringan": "houses_light",
        "Rumah Terendam": "houses_flooded",
    })
    
    # Aggregate by kabupaten (across all years)
    agg = df.groupby(["province", "kabupaten"]).agg({
        "events": "sum",
        "deaths": "sum",
        "missing": "sum",
        "injured": "sum",
        "affected": "sum",
        "displaced": "sum",
        "houses_heavy": "sum",
        "houses_medium": "sum",
        "houses_light": "sum",
        "houses_flooded": "sum",
        "year": ["min", "max", "nunique"],
    }).reset_index()
    
    # Flatten column names
    agg.columns = [
        "province", "kabupaten",
        "flood_events_total", "flood_deaths", "flood_missing", "flood_injured",
        "flood_affected", "flood_displaced",
        "flood_houses_heavy", "flood_houses_medium", "flood_houses_light", "flood_houses_flooded",
        "year_first", "year_last", "years_active"
    ]
    
    # Calculate flood impact score
    agg["flood_impact_score"] = (
        agg["flood_deaths"] * 100 +
        agg["flood_missing"] * 50 +
        agg["flood_injured"] * 10 +
        agg["flood_affected"] * 0.1 +
        agg["flood_houses_flooded"] * 0.5
    )
    
    # Calculate average events per year
    agg["flood_events_annual_avg"] = agg["flood_events_total"] / agg["years_active"].clip(lower=1)
    
    # Flood frequency category
    agg["flood_frequency_category"] = pd.cut(
        agg["flood_events_total"],
        bins=[0, 5, 15, 30, 50, float("inf")],
        labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )
    
    logger.info(f"Loaded flood data: {len(agg)} kabupaten")
    return agg


def load_forest_data() -> pd.DataFrame:
    """Load forest cover and loss data."""
    logger.info("Loading forest data...")
    
    forest_path = PROJECT_ROOT / "data" / "external" / "forest_cover" / "forest_palm_statistics.parquet"
    
    if not forest_path.exists():
        logger.warning(f"Forest data not found: {forest_path}")
        logger.info("Run scripts/02_download_gfc.py first!")
        return None
    
    df = pd.read_parquet(forest_path)
    logger.info(f"Loaded forest data: {len(df)} records")
    return df


def load_rainfall_data() -> pd.DataFrame:
    """Load rainfall data."""
    logger.info("Loading rainfall data...")
    
    rainfall_path = PROJECT_ROOT / "data" / "external" / "rainfall" / "rainfall_aggregated.parquet"
    
    if not rainfall_path.exists():
        logger.warning(f"Rainfall data not found: {rainfall_path}")
        logger.info("Run scripts/03_download_chirps.py first!")
        return None
    
    df = pd.read_parquet(rainfall_path)
    logger.info(f"Loaded rainfall data: {len(df)} records")
    return df


def merge_all_datasets(
    flood_df: pd.DataFrame,
    forest_df: pd.DataFrame,
    rainfall_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all datasets on kabupaten.
    
    Uses fuzzy matching for kabupaten names since they may differ slightly
    between data sources.
    """
    logger.info("Merging datasets...")
    
    # Start with flood data as base
    merged = flood_df.copy()
    
    # Standardize kabupaten names for matching
    def standardize_name(name):
        if pd.isna(name):
            return ""
        name = str(name).upper().strip()
        # Remove common prefixes
        for prefix in ["KAB.", "KABUPATEN ", "KOTA ", "KAB "]:
            name = name.replace(prefix, "")
        return name.strip()
    
    merged["kabupaten_std"] = merged["kabupaten"].apply(standardize_name)
    
    # Merge forest data
    if forest_df is not None:
        forest_df = forest_df.copy()
        forest_df["kabupaten_std"] = forest_df["kabupaten"].apply(standardize_name)
        
        # Select columns to merge
        forest_cols = [
            "kabupaten_std",
            "forest_cover_2000_pct", "forest_cover_2023_pct",
            "forest_loss_pct", "deforestation_rate_annual",
            "palm_oil_pct", "forest_to_palm_ratio"
        ]
        forest_merge = forest_df[[c for c in forest_cols if c in forest_df.columns]]
        
        merged = merged.merge(forest_merge, on="kabupaten_std", how="left")
        logger.info(f"Merged forest data: {merged['forest_cover_2000_pct'].notna().sum()} matches")
    
    # Merge rainfall data
    if rainfall_df is not None:
        rainfall_df = rainfall_df.copy()
        rainfall_df["kabupaten_std"] = rainfall_df["kabupaten"].apply(standardize_name)
        
        rainfall_cols = [
            "kabupaten_std",
            "rainfall_annual_mean_mm", "rainfall_wet_season_mm", "rainfall_dry_season_mm",
            "rainfall_extreme_days_avg", "rainfall_max_daily_mm", "rainfall_cv"
        ]
        rainfall_merge = rainfall_df[[c for c in rainfall_cols if c in rainfall_df.columns]]
        
        merged = merged.merge(rainfall_merge, on="kabupaten_std", how="left")
        logger.info(f"Merged rainfall data: {merged['rainfall_annual_mean_mm'].notna().sum()} matches")
    
    # Drop temporary column
    merged = merged.drop(columns=["kabupaten_std"])
    
    return merged


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for analysis and modeling."""
    logger.info("Adding derived features...")
    
    df = df.copy()
    
    # Province to island mapping
    province_island = {
        "Aceh": "Sumatra", "Sumatera Utara": "Sumatra", "Sumatera Barat": "Sumatra",
        "Riau": "Sumatra", "Kepulauan Riau": "Sumatra", "Jambi": "Sumatra",
        "Sumatera Selatan": "Sumatra", "Bengkulu": "Sumatra", "Lampung": "Sumatra",
        "Kepulauan Bangka Belitung": "Sumatra",
        "DKI Jakarta": "Jawa", "Jawa Barat": "Jawa", "Jawa Tengah": "Jawa",
        "Daerah Istimewa Yogyakarta": "Jawa", "Jawa Timur": "Jawa", "Banten": "Jawa",
        "Kalimantan Barat": "Kalimantan", "Kalimantan Tengah": "Kalimantan",
        "Kalimantan Selatan": "Kalimantan", "Kalimantan Timur": "Kalimantan",
        "Kalimantan Utara": "Kalimantan",
        "Sulawesi Utara": "Sulawesi", "Gorontalo": "Sulawesi", "Sulawesi Tengah": "Sulawesi",
        "Sulawesi Selatan": "Sulawesi", "Sulawesi Barat": "Sulawesi", "Sulawesi Tenggara": "Sulawesi",
        "Bali": "Bali & Nusa Tenggara", "Nusa Tenggara Barat": "Bali & Nusa Tenggara",
        "Nusa Tenggara Timur": "Bali & Nusa Tenggara",
        "Maluku": "Maluku", "Maluku Utara": "Maluku",
        "Papua": "Papua", "Papua Barat": "Papua", "Papua Barat Daya": "Papua",
        "Papua Pegunungan": "Papua", "Papua Selatan": "Papua", "Papua Tengah": "Papua",
    }
    
    df["island"] = df["province"].map(province_island).fillna("Lainnya")
    
    # Flood risk label (binary classification target)
    if "flood_impact_score" in df.columns:
        median_score = df["flood_impact_score"].median()
        df["flood_risk_label"] = (df["flood_impact_score"] > median_score).astype(int)
    
    # Flood severity index (normalized)
    if "flood_impact_score" in df.columns:
        max_score = df["flood_impact_score"].max()
        df["flood_severity_index"] = df["flood_impact_score"] / max_score
    
    # Deforestation-flood interaction
    if "forest_loss_pct" in df.columns and "flood_events_total" in df.columns:
        df["deforestation_flood_product"] = (
            df["forest_loss_pct"].fillna(0) * 
            df["flood_events_annual_avg"].fillna(0)
        )
    
    # Palm oil-flood interaction  
    if "palm_oil_pct" in df.columns and "flood_events_total" in df.columns:
        df["palm_oil_flood_product"] = (
            df["palm_oil_pct"].fillna(0) * 
            df["flood_events_annual_avg"].fillna(0)
        )
    
    # Rainfall-flood interaction
    if "rainfall_annual_mean_mm" in df.columns and "flood_events_total" in df.columns:
        df["rainfall_flood_product"] = (
            df["rainfall_annual_mean_mm"].fillna(0) * 
            df["flood_events_annual_avg"].fillna(0) / 1000
        )
    
    # High deforestation flag
    if "forest_loss_pct" in df.columns:
        df["high_deforestation"] = (df["forest_loss_pct"] > 20).astype(int)
    
    # High palm oil flag
    if "palm_oil_pct" in df.columns:
        df["high_palm_oil"] = (df["palm_oil_pct"] > 25).astype(int)
    
    # Extreme rainfall flag
    if "rainfall_extreme_days_avg" in df.columns:
        df["high_extreme_rainfall"] = (df["rainfall_extreme_days_avg"] > 20).astype(int)
    
    # Compound risk score (combination of all risk factors)
    compound_score = np.zeros(len(df))
    
    if "forest_loss_pct" in df.columns:
        compound_score += df["forest_loss_pct"].fillna(0) / 100 * 30  # Max 30 points
    
    if "palm_oil_pct" in df.columns:
        compound_score += df["palm_oil_pct"].fillna(0) / 100 * 25  # Max 25 points
    
    if "rainfall_extreme_days_avg" in df.columns:
        compound_score += df["rainfall_extreme_days_avg"].fillna(0) / 60 * 25  # Max 25 points
    
    if "flood_events_annual_avg" in df.columns:
        compound_score += np.clip(df["flood_events_annual_avg"].fillna(0) / 10, 0, 1) * 20  # Max 20 points
    
    df["compound_risk_score"] = compound_score
    
    # Risk category
    df["risk_category"] = pd.cut(
        df["compound_risk_score"],
        bins=[0, 20, 40, 60, 80, 100],
        labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )
    
    return df


def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlations between features."""
    logger.info("Calculating feature correlations...")
    
    # Select numeric columns for correlation
    numeric_cols = [
        "flood_events_total", "flood_deaths", "flood_impact_score",
        "forest_loss_pct", "palm_oil_pct", 
        "rainfall_annual_mean_mm", "rainfall_extreme_days_avg",
        "deforestation_flood_product", "palm_oil_flood_product",
        "compound_risk_score"
    ]
    
    available_cols = [c for c in numeric_cols if c in df.columns]
    
    if len(available_cols) < 2:
        logger.warning("Not enough numeric columns for correlation")
        return None
    
    corr_matrix = df[available_cols].corr()
    
    # Log key correlations
    logger.info("\nKey Correlations with Flood Events:")
    if "flood_events_total" in available_cols:
        flood_corr = corr_matrix["flood_events_total"].drop("flood_events_total")
        for col, val in flood_corr.sort_values(ascending=False).items():
            logger.info(f"  {col}: {val:.3f}")
    
    return corr_matrix


def save_complete_dataset(df: pd.DataFrame, output_dir: Path) -> None:
    """Save the complete merged dataset."""
    logger.info("Saving complete dataset...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet (efficient)
    parquet_path = output_dir / "complete_analysis_dataset.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.success(f"Saved: {parquet_path}")
    
    # Save as CSV (human readable)
    csv_path = output_dir / "complete_analysis_dataset.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")
    
    # Save summary statistics
    summary_path = output_dir / "dataset_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("COMPLETE ANALYSIS DATASET SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total Records: {len(df)}\n")
        f.write(f"Total Columns: {len(df.columns)}\n")
        f.write(f"Provinces: {df['province'].nunique()}\n")
        f.write(f"Islands: {df['island'].nunique()}\n\n")
        
        f.write("Columns:\n")
        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].notna().sum()
            f.write(f"  - {col}: {dtype} ({non_null} non-null)\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("DESCRIPTIVE STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(df.describe().to_string())
    
    logger.info(f"Saved summary: {summary_path}")


def main():
    """Main function to merge all datasets."""
    logger.info("=" * 60)
    logger.info("COMPLETE DATA MERGE PIPELINE")
    logger.info("=" * 60)
    
    # Load all datasets
    flood_df = load_flood_data()
    forest_df = load_forest_data()
    rainfall_df = load_rainfall_data()
    
    if flood_df is None:
        logger.error("Flood data is required. Exiting.")
        return
    
    # Merge datasets
    merged_df = merge_all_datasets(flood_df, forest_df, rainfall_df)
    
    # Add derived features
    final_df = add_derived_features(merged_df)
    
    # Calculate correlations
    corr_matrix = calculate_correlations(final_df)
    
    # Save results
    output_dir = PROJECT_ROOT / "data" / "processed"
    save_complete_dataset(final_df, output_dir)
    
    if corr_matrix is not None:
        corr_path = output_dir / "correlation_matrix.csv"
        corr_matrix.to_csv(corr_path)
        logger.info(f"Saved correlation matrix: {corr_path}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("MERGE COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Total records: {len(final_df)}")
    logger.info(f"Total features: {len(final_df.columns)}")
    
    # Feature counts
    flood_features = [c for c in final_df.columns if "flood" in c.lower()]
    forest_features = [c for c in final_df.columns if "forest" in c.lower() or "palm" in c.lower()]
    rainfall_features = [c for c in final_df.columns if "rain" in c.lower()]
    
    logger.info(f"Flood features: {len(flood_features)}")
    logger.info(f"Forest/Palm features: {len(forest_features)}")
    logger.info(f"Rainfall features: {len(rainfall_features)}")
    
    # High risk summary
    if "risk_category" in final_df.columns:
        logger.info("\nRisk Category Distribution:")
        for cat, count in final_df["risk_category"].value_counts().items():
            pct = count / len(final_df) * 100
            logger.info(f"  {cat}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()


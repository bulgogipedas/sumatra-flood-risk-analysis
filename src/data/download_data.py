"""
Download data from various sources for SawitFlood Lab analysis.

This script handles downloading:
1. Administrative boundaries (GADM)
2. Forest cover data (Global Forest Change)
3. Palm oil plantation maps
4. Flood event data (BNPB DIBI)
5. Rainfall data (CHIRPS)

Usage:
    python src/data/download_data.py
    python src/data/download_data.py --source admin
    python src/data/download_data.py --all
"""

import argparse
import sys
from pathlib import Path

import requests
import yaml
from loguru import logger
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DataDownloader:
    """Handle data downloads from multiple sources."""

    def __init__(self, config_path: str | None = None):
        """
        Initialize DataDownloader.

        Args:
            config_path: Path to settings.yaml. If None, uses default.
        """
        if config_path is None:
            config_path = PROJECT_ROOT / "configs" / "settings.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.raw_dir = PROJECT_ROOT / "data" / "raw"
        self.external_dir = PROJECT_ROOT / "data" / "external"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.external_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logger.add(PROJECT_ROOT / "logs" / "download.log", rotation="10 MB", level="INFO")

        logger.info(f"Initialized DataDownloader with config from {config_path}")

    def _download_file(self, url: str, output_path: Path, desc: str = "Downloading") -> bool:
        """
        Download a file with progress bar.

        Args:
            url: URL to download from
            output_path: Path to save the file
            desc: Description for progress bar

        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with (
                open(output_path, "wb") as f,
                tqdm(total=total_size, unit="iB", unit_scale=True, desc=desc) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)

            logger.info(f"Downloaded: {output_path}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Download failed for {url}: {e}")
            return False

    def download_admin_boundaries(self) -> Path:
        """
        Download administrative boundaries from GADM.

        Returns:
            Path to downloaded/extracted data
        """
        logger.info("Downloading administrative boundaries from GADM...")

        # GADM Indonesia Level 2 (Kabupaten/Kota)
        output_dir = self.external_dir / "admin_boundaries"
        output_dir.mkdir(exist_ok=True)

        # GADM download URL for Indonesia
        gadm_url = "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_IDN.gpkg"
        output_file = output_dir / "gadm41_IDN.gpkg"

        if output_file.exists():
            logger.info(f"Admin boundaries already exist at {output_file}")
            return output_file

        # Note: In practice, you might need to download from GADM website manually
        # as they require acceptance of terms
        logger.warning(
            "GADM data requires manual download from https://gadm.org/download_country.html\n"
            f"Please download Indonesia (IDN) Level 2 and save to: {output_file}"
        )

        # Create placeholder instructions
        instructions_file = output_dir / "DOWNLOAD_INSTRUCTIONS.txt"
        with open(instructions_file, "w") as f:
            f.write("""
GADM Administrative Boundaries Download Instructions
====================================================

1. Go to: https://gadm.org/download_country.html
2. Select: Indonesia
3. Download: Level 2 (Kabupaten/Kota) in GeoPackage format
4. Save the file as: gadm41_IDN.gpkg in this directory

Alternative: Use the GeoJSON version if GeoPackage is not available.
""")

        return output_dir

    def download_forest_cover_tiles(self, tiles: list[str] | None = None) -> Path:
        """
        Download Global Forest Change tiles for Sumatra.

        Args:
            tiles: List of tile IDs to download. If None, downloads Sumatra tiles.

        Returns:
            Path to download directory
        """
        logger.info("Setting up Global Forest Change data download...")

        output_dir = self.raw_dir / "forest_cover"
        output_dir.mkdir(exist_ok=True)

        # Sumatra tiles (approximate - Hansen GFC uses 10x10 degree tiles)
        # Sumatra spans roughly: Lat -6 to 6, Lon 95 to 106
        if tiles is None:
            tiles = [
                "00N_090E",
                "00N_100E",  # Equatorial tiles
                "10N_090E",
                "10N_100E",  # Northern tiles
            ]

        base_url = self.config["data_sources"]["forest_cover"]["url"]
        layers = ["treecover2000", "lossyear"]

        # Create download script instead of direct download (files are large)
        script_content = f"""#!/bin/bash
# Download Global Forest Change tiles for Sumatra
# Source: Hansen et al. 2013

BASE_URL="{base_url}"
OUTPUT_DIR="{output_dir}"

mkdir -p $OUTPUT_DIR

# Tiles covering Sumatra
TILES=({" ".join(tiles)})

for TILE in "${{TILES[@]}}"; do
    echo "Downloading tile: $TILE"
    
    # Tree cover 2000
    wget -nc -P $OUTPUT_DIR "${{BASE_URL}}Hansen_GFC-2023-v1.11_treecover2000_${{TILE}}.tif"
    
    # Forest loss year
    wget -nc -P $OUTPUT_DIR "${{BASE_URL}}Hansen_GFC-2023-v1.11_lossyear_${{TILE}}.tif"
done

echo "Download complete!"
"""

        script_path = output_dir / "download_gfc.sh"
        with open(script_path, "w") as f:
            f.write(script_content)

        logger.info(f"Created download script at {script_path}")
        logger.info("Run 'bash download_gfc.sh' to download Global Forest Change data")

        # Also create Python alternative for Windows users
        python_script = output_dir / "download_gfc.py"
        with open(python_script, "w") as f:
            f.write(f'''"""
Download Global Forest Change tiles for Sumatra.
Run this script to download the required GFC data.
"""

import os
import urllib.request
from tqdm import tqdm

BASE_URL = "{base_url}"
OUTPUT_DIR = r"{output_dir}"
TILES = {tiles}

os.makedirs(OUTPUT_DIR, exist_ok=True)

for tile in TILES:
    for layer in ["treecover2000", "lossyear"]:
        filename = f"Hansen_GFC-2023-v1.11_{{layer}}_{{tile}}.tif"
        url = BASE_URL + filename
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(output_path):
            print(f"Already exists: {{filename}}")
            continue
        
        print(f"Downloading: {{filename}}")
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"  Saved to: {{output_path}}")
        except Exception as e:
            print(f"  Error: {{e}}")

print("Download complete!")
''')

        return output_dir

    def download_palm_oil_data(self) -> Path:
        """
        Setup palm oil plantation data download.

        Returns:
            Path to download directory
        """
        logger.info("Setting up palm oil plantation data...")

        output_dir = self.raw_dir / "palm_oil"
        output_dir.mkdir(exist_ok=True)

        # Create instructions file
        instructions = output_dir / "DOWNLOAD_INSTRUCTIONS.txt"
        with open(instructions, "w") as f:
            f.write("""
Palm Oil Plantation Data Sources
=================================

Option 1: Global Palm Oil Map (Descals et al. 2021)
---------------------------------------------------
- Paper: https://doi.org/10.5194/essd-13-1211-2021
- Data: https://zenodo.org/record/4473715
- Download the Indonesia subset

Option 2: MapBiomas Indonesia
-----------------------------
- Website: https://mapbiomas.nusantara.earth/
- Download land use maps for Sumatra
- Filter for oil palm plantations

Option 3: Global Forest Watch
-----------------------------
- Website: https://www.globalforestwatch.org/
- Navigate to: Map > Commodities > Oil Palm
- Download data for Indonesia/Sumatra

After downloading, place the files in this directory.
""")

        logger.info(f"Created palm oil data instructions at {instructions}")
        return output_dir

    def download_flood_data(self) -> Path:
        """
        Setup flood event data download from BNPB.

        Returns:
            Path to download directory
        """
        logger.info("Setting up flood event data...")

        output_dir = self.raw_dir / "flood_events"
        output_dir.mkdir(exist_ok=True)

        # Create sample data structure
        sample_data = output_dir / "sample_flood_data.csv"
        with open(sample_data, "w") as f:
            f.write("""province,kabupaten,kabupaten_id,year,flood_events,affected_people,deaths,houses_damaged
Riau,Pekanbaru,1471,2020,5,2340,2,156
Riau,Kampar,1401,2020,8,4520,5,289
Riau,Bengkalis,1402,2020,6,3100,1,201
Sumatra Utara,Medan,1275,2020,12,8900,8,567
Sumatra Utara,Deli Serdang,1207,2020,15,12000,10,890
Jambi,Kota Jambi,1571,2020,4,1890,1,120
""")

        # Create instructions
        instructions = output_dir / "DOWNLOAD_INSTRUCTIONS.txt"
        with open(instructions, "w") as f:
            f.write("""
Flood Event Data Sources
========================

Primary: BNPB Data Informasi Bencana Indonesia (DIBI)
-----------------------------------------------------
- Website: https://dibi.bnpb.go.id/
- Navigate to: Data Bencana > Banjir
- Filter by province and year range
- Export to Excel/CSV

Alternative: DesInventar
------------------------
- Website: https://www.desinventar.net/
- Country: Indonesia
- Event type: Flood
- Download historical data

Alternative: EM-DAT
-------------------
- Website: https://www.emdat.be/
- Requires registration
- Filter: Country=Indonesia, Disaster=Flood

Data Format Required:
- province: Province name
- kabupaten: Kabupaten/Kota name  
- kabupaten_id: BPS code
- year: Year of event
- flood_events: Number of flood events
- affected_people: Total affected population
- deaths: Number of deaths
- houses_damaged: Number of houses damaged

Save as CSV with UTF-8 encoding.
""")

        logger.info(f"Created flood data instructions and sample at {output_dir}")
        return output_dir

    def download_rainfall_data(self) -> Path:
        """
        Setup CHIRPS rainfall data download.

        Returns:
            Path to download directory
        """
        logger.info("Setting up CHIRPS rainfall data...")

        output_dir = self.raw_dir / "rainfall"
        output_dir.mkdir(exist_ok=True)

        # CHIRPS data parameters
        start_year = self.config["temporal"]["start_year"]
        end_year = self.config["temporal"]["end_year"]

        # Create download script
        script_content = f"""#!/bin/bash
# Download CHIRPS monthly rainfall data for Indonesia region
# Source: Climate Hazards Group

BASE_URL="https://data.chc.ucsb.edu/products/CHIRPS-2.0/SEasia_monthly/tifs/"
OUTPUT_DIR="{output_dir}"

mkdir -p $OUTPUT_DIR

for YEAR in $(seq {start_year} {end_year}); do
    for MONTH in $(seq -w 1 12); do
        FILENAME="chirps-v2.0.${{YEAR}}.${{MONTH}}.tif"
        URL="${{BASE_URL}}${{FILENAME}}"
        
        if [ ! -f "$OUTPUT_DIR/$FILENAME" ]; then
            echo "Downloading: $FILENAME"
            wget -q -P $OUTPUT_DIR $URL
        fi
    done
done

echo "CHIRPS download complete!"
"""

        script_path = output_dir / "download_chirps.sh"
        with open(script_path, "w") as f:
            f.write(script_content)

        # Create Python alternative
        python_script = output_dir / "download_chirps.py"
        with open(python_script, "w") as f:
            f.write(f'''"""
Download CHIRPS monthly rainfall data for Indonesia region.
"""

import os
import urllib.request
from datetime import datetime

BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/SEasia_monthly/tifs/"
OUTPUT_DIR = r"{output_dir}"

START_YEAR = {start_year}
END_YEAR = {end_year}

os.makedirs(OUTPUT_DIR, exist_ok=True)

for year in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):
        filename = f"chirps-v2.0.{{year}}.{{month:02d}}.tif"
        url = BASE_URL + filename
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(output_path):
            print(f"Already exists: {{filename}}")
            continue
        
        print(f"Downloading: {{filename}}")
        try:
            urllib.request.urlretrieve(url, output_path)
        except Exception as e:
            print(f"  Error: {{e}}")

print("Download complete!")
''')

        logger.info(f"Created CHIRPS download scripts at {output_dir}")
        return output_dir

    def download_all(self) -> dict:
        """
        Download all required data sources.

        Returns:
            Dictionary with paths to all downloaded data
        """
        logger.info("Starting download of all data sources...")

        paths = {
            "admin_boundaries": self.download_admin_boundaries(),
            "forest_cover": self.download_forest_cover_tiles(),
            "palm_oil": self.download_palm_oil_data(),
            "flood_events": self.download_flood_data(),
            "rainfall": self.download_rainfall_data(),
        }

        logger.info("All download tasks completed!")
        logger.info("Please check DOWNLOAD_INSTRUCTIONS.txt files for manual download steps.")

        return paths


def main():
    """Main entry point for data download."""
    parser = argparse.ArgumentParser(description="Download data for SawitFlood Lab analysis")
    parser.add_argument(
        "--source",
        type=str,
        choices=["admin", "forest", "palm", "flood", "rainfall", "all"],
        default="all",
        help="Which data source to download",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config file")

    args = parser.parse_args()

    downloader = DataDownloader(config_path=args.config)

    if args.source == "all":
        downloader.download_all()
    elif args.source == "admin":
        downloader.download_admin_boundaries()
    elif args.source == "forest":
        downloader.download_forest_cover_tiles()
    elif args.source == "palm":
        downloader.download_palm_oil_data()
    elif args.source == "flood":
        downloader.download_flood_data()
    elif args.source == "rainfall":
        downloader.download_rainfall_data()


if __name__ == "__main__":
    main()


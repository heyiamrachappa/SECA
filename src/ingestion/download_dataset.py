"""
Dataset downloader and acquisition manager for the Low Carbon London SmartMeter dataset.
Source: Official Greater London Authority (London Datastore) / UK Power Networks.
"""

import os
import sys
import zipfile
import urllib.request
import time
from pathlib import Path
from typing import Optional, List
from src.utils.config import load_config, get_project_root
from src.utils.logger import get_logger

logger = get_logger("DatasetAcquisition")

OFFICIAL_DATASTORE_PAGE = "https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households"
OFFICIAL_PARTITIONED_ZIP_URL = "https://data.london.gov.uk/download/vqm0d/04feba67-f1a3-4563-98d0-f3071e3d56d1/Partitioned%20LCL%20Data.zip"
DEFAULT_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"


def download_with_progress(url: str, output_path: Path, user_agent: str = DEFAULT_USER_AGENT) -> None:
    """
    Downloads a large file over HTTP with standard User-Agent header and progress reporting.
    """
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        chunk_size = 1024 * 1024  # 1 MB chunk
        downloaded = 0
        start_time = time.time()
        last_log_time = start_time

        logger.info(f"Target size: {total_size / (1024 * 1024):.2f} MB")

        with open(output_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_log_time >= 5.0 or (total_size > 0 and downloaded == total_size):
                    last_log_time = now
                    elapsed = now - start_time
                    speed = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                    pct = (downloaded * 100.0 / total_size) if total_size > 0 else 0
                    logger.info(
                        f"Progress: {pct:.1f}% ({downloaded / (1024 * 1024):.1f}/{total_size / (1024 * 1024):.1f} MB) "
                        f"at {speed:.2f} MB/s"
                    )


def download_london_dataset(
    dest_dir: Optional[str] = None,
    extract_blocks: Optional[int] = None,
    force_download: bool = False
) -> Path:
    """
    Downloads the official Low Carbon London Partitioned ZIP dataset and extracts CSV blocks.

    Parameters:
        dest_dir (str, optional): Target directory for raw CSVs.
        extract_blocks (int, optional): If specified, extracts only the first N CSV blocks.
                                        If None, extracts all blocks.
        force_download (bool): Whether to re-download if the zip already exists.

    Returns:
        Path: Path to the raw data directory containing CSV files.
    """
    root = get_project_root()
    config = load_config()
    target_raw_dir = root / (dest_dir or config["paths"]["raw_data_dir"])
    target_raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = target_raw_dir / "Partitioned_LCL_Data.zip"

    # Step 1: Download archive if not present
    if not zip_path.exists() or force_download:
        logger.info("Downloading Low Carbon London dataset from official London Datastore source:")
        logger.info(f"Source URL: {OFFICIAL_PARTITIONED_ZIP_URL}")
        logger.info(f"Destination archive: {zip_path}")

        try:
            download_with_progress(OFFICIAL_PARTITIONED_ZIP_URL, zip_path)
            logger.info("Archive download finished successfully.")
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            if zip_path.exists():
                zip_path.unlink()
            raise
    else:
        logger.info(f"Dataset archive already present at {zip_path}. Skipping download.")

    # Step 2: Extract CSV blocks
    logger.info(f"Inspecting archive: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        csv_members = [
            m for m in zip_ref.namelist()
            if m.lower().endswith(".csv") and not m.startswith("__MACOSX")
        ]
        csv_members.sort()
        logger.info(f"Found {len(csv_members)} CSV block partitions inside archive.")

        if extract_blocks is not None and extract_blocks > 0:
            members_to_extract = csv_members[:extract_blocks]
            logger.info(f"Extracting first {extract_blocks} CSV blocks for pipeline processing...")
        else:
            members_to_extract = csv_members
            logger.info(f"Extracting all {len(csv_members)} CSV blocks...")

        for member in members_to_extract:
            extracted_filename = Path(member).name
            target_file = target_raw_dir / extracted_filename
            if not target_file.exists():
                logger.info(f"Extracting {member} -> {target_file.name}")
                with zip_ref.open(member) as source, open(target_file, "wb") as target:
                    target.write(source.read())
            else:
                logger.info(f"File {target_file.name} already exists. Skipping extraction.")

    logger.info(f"Dataset acquisition complete. CSV blocks ready in {target_raw_dir}")
    return target_raw_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Low Carbon London SmartMeter Dataset")
    parser.add_argument("--blocks", type=int, default=5, help="Number of CSV blocks to extract (default: 5)")
    parser.add_argument("--all", action="store_true", help="Extract all 168 CSV blocks")
    parser.add_argument("--force", action="store_true", help="Force redownload")
    args = parser.parse_args()

    num_blocks = None if args.all else args.blocks
    download_london_dataset(extract_blocks=num_blocks, force_download=args.force)

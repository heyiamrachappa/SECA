"""
Sample dataset generator for development, quick testing, and continuous integration.
Extracts a representative subset from raw data if present, or synthesizes authentic
half-hourly telemetry matching the exact Low Carbon London schema.
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from src.utils.config import load_config, get_project_root
from src.utils.logger import get_logger

logger = get_logger("SampleGenerator")


def create_sample_dataset(
    output_path: Optional[str] = None,
    num_households: int = 25,
    num_days: int = 30,
    force_synthetic: bool = False
) -> Path:
    """
    Creates a sample dataset for fast development and testing.

    If raw CSV files exist in dataset/raw/, extracts authentic records from block_0.csv.
    Otherwise, generates authentic synthetic half-hourly time-series matching the Low Carbon London schema.

    Parameters:
        output_path (str, optional): Target CSV path.
        num_households (int): Number of households to include.
        num_days (int): Number of days of half-hourly readings.
        force_synthetic (bool): Force synthetic generation even if raw data is present.

    Returns:
        Path: Path to generated sample CSV file.
    """
    root = get_project_root()
    config = load_config()
    sample_dir = root / config["paths"]["sample_data_dir"]
    sample_dir.mkdir(parents=True, exist_ok=True)

    dest_file = root / (output_path or (sample_dir / "london_smartmeter_sample.csv"))
    raw_dir = root / config["paths"]["raw_data_dir"]
    raw_csvs = sorted(list(raw_dir.glob("block_*.csv")))

    # Strategy 1: Extract from authentic raw block if present and not forced synthetic
    if raw_csvs and not force_synthetic:
        logger.info(f"Extracting sample from authentic raw data: {raw_csvs[0].name}")
        df_raw = pd.read_csv(raw_csvs[0], nrows=150000)
        # Standardize column checking
        col_map = {c: c.strip() for c in df_raw.columns}
        df_raw = df_raw.rename(columns=col_map)

        id_col = "LCLid" if "LCLid" in df_raw.columns else df_raw.columns[0]
        households = df_raw[id_col].unique()[:num_households]
        df_sample = df_raw[df_raw[id_col].isin(households)].copy()

        # Retain original column names
        df_sample.to_csv(dest_file, index=False)
        logger.info(f"Created authentic sample with {len(df_sample):,} rows across {len(households)} households at {dest_file}")
        return dest_file

    # Strategy 2: Generate realistic half-hourly London telemetry
    logger.info("Generating synthetic Low Carbon London sample dataset...")
    np.random.seed(42)

    # Date range: 30 days of half-hourly intervals (48 readings per day)
    date_range = pd.date_range(start="2013-01-01 00:00:00", periods=num_days * 48, freq="30min")

    records = []
    tariffs = ["Std"] * int(num_households * 0.8) + ["ToU"] * (num_households - int(num_households * 0.8))

    for i in range(num_households):
        household_id = f"MAC{i+1:06d}"
        tariff = tariffs[i]
        # Base consumption profile with diurnal peaks (morning 7-9 AM, evening 18-22 PM)
        base_load = np.random.uniform(0.15, 0.45)

        for dt in date_range:
            hour = dt.hour + (dt.minute / 60.0)
            # Diurnal multiplier
            morning_peak = np.exp(-((hour - 8.0) ** 2) / 4.0) * 0.8
            evening_peak = np.exp(-((hour - 19.5) ** 2) / 6.0) * 1.4
            night_dip = -0.1 if (hour < 6 or hour > 23) else 0.0

            noise = np.random.normal(0, 0.05)
            kwh = max(0.01, base_load + morning_peak + evening_peak + night_dip + noise)

            # Introduce small rate of missing data ('Null') like real smart meters (~0.2%)
            if np.random.random() < 0.002:
                kwh_str = "Null"
            else:
                kwh_str = f"{kwh:.4f}"

            records.append({
                "LCLid": household_id,
                "stdorToU": tariff,
                "DateTime": dt.strftime("%Y-%m-%d %H:%M:%S.0000000"),
                "KWH/hh (per half hour) ": kwh_str
            })

    df_synth = pd.DataFrame(records)
    df_synth.to_csv(dest_file, index=False)
    logger.info(f"Generated synthetic sample with {len(df_synth):,} rows across {num_households} households at {dest_file}")
    return dest_file


if __name__ == "__main__":
    create_sample_dataset()

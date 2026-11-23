"""
Dataset inspection and schema validation module.
Analyzes raw CSV files, validates column headers, checks data types,
detects missing/Null values, and produces a summary data quality report.
"""

import os
import glob
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from src.utils.config import load_config, get_project_root
from src.utils.logger import get_logger

logger = get_logger("DatasetInspector")


def find_csv_files(directory: Path) -> List[Path]:
    """Returns a sorted list of CSV files in the specified directory."""
    if not directory.exists():
        logger.warning(f"Directory {directory} does not exist.")
        return []
    files = sorted(list(directory.glob("*.csv")))
    return files


def inspect_raw_csv(file_path: Path, sample_rows: int = 10000) -> Dict[str, Any]:
    """
    Performs quick inspection on a single CSV block file.

    Parameters:
        file_path (Path): Path to CSV file.
        sample_rows (int): Number of rows to read for sample stats.

    Returns:
        dict: Inspection metrics including headers, row count, nulls, and sample rows.
    """
    logger.info(f"Inspecting CSV file: {file_path.name}")

    # Read header and first batch of rows
    df_sample = pd.read_csv(file_path, nrows=sample_rows, dtype=str)

    # Detect header columns (stripping trailing whitespace for detection)
    raw_columns = list(df_sample.columns)
    clean_columns = [c.strip() for c in raw_columns]

    # Check for known canonical columns
    has_household = any(c in clean_columns for c in ["LCLid", "household_id"])
    has_datetime = any(c in clean_columns for c in ["DateTime", "timestamp", "tstp"])
    has_kwh = any("KWH" in c or "kwh" in c or "energy" in c for c in clean_columns)

    # Quick line count estimation
    with open(file_path, "rb") as f:
        line_count = sum(1 for _ in f) - 1  # exclude header

    # Count 'Null' strings or empty values in sample
    null_counts = {}
    for col in df_sample.columns:
        null_count = (df_sample[col].isna() | (df_sample[col].str.strip() == "Null") | (df_sample[col].str.strip() == "")).sum()
        null_counts[col.strip()] = int(null_count)

    unique_households = df_sample.iloc[:, 0].nunique() if has_household else 0

    return {
        "filename": file_path.name,
        "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
        "total_lines": line_count,
        "sample_analyzed_rows": len(df_sample),
        "raw_columns": raw_columns,
        "clean_columns": clean_columns,
        "has_expected_schema": has_household and has_datetime and has_kwh,
        "unique_households_in_sample": unique_households,
        "null_counts_in_sample": null_counts,
        "sample_head": df_sample.head(3).to_dict(orient="records")
    }


def generate_inspection_report(target_dir: Optional[str] = None, output_report: str = "reports/dataset_inspection_report.md") -> Dict[str, Any]:
    """
    Inspects all CSV files in target directory and writes a markdown summary report.
    """
    root = get_project_root()
    config = load_config()
    data_dir = root / (target_dir or config["paths"]["raw_data_dir"])

    csv_files = find_csv_files(data_dir)
    if not csv_files:
        # Fallback to sample directory if raw has no CSVs yet
        sample_dir = root / config["paths"]["sample_data_dir"]
        csv_files = find_csv_files(sample_dir)
        if csv_files:
            logger.info(f"No CSVs found in {data_dir}; using sample directory {sample_dir}")
            data_dir = sample_dir
        else:
            raise FileNotFoundError(f"No CSV files found in {data_dir} or {sample_dir}")

    logger.info(f"Inspecting {len(csv_files)} CSV files in {data_dir}...")

    summaries = []
    total_records = 0
    total_size_mb = 0.0

    for file_path in csv_files:
        info = inspect_raw_csv(file_path)
        summaries.append(info)
        total_records += info["total_lines"]
        total_size_mb += info["file_size_mb"]

    report_data = {
        "data_directory": str(data_dir),
        "file_count": len(csv_files),
        "total_records": total_records,
        "total_size_mb": round(total_size_mb, 2),
        "files": summaries
    }

    # Format Markdown Report
    report_path = root / output_report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    first_file = summaries[0]
    md_content = f"""# Low Carbon London SmartMeter Dataset Inspection Report

**Generated Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Data Source Directory:** `{data_dir}`  
**Total CSV Block Files:** {len(csv_files)}  
**Total Telemetry Records:** {total_records:,}  
**Total Disk Footprint:** {round(total_size_mb, 2)} MB  

---

## 1. Schema & Column Headers Analysis

Inspection of `{first_file['filename']}`:
- **Raw Headers:** `{first_file['raw_columns']}`
- **Standardized Headers:** `{first_file['clean_columns']}`
- **Schema Validation Status:** {'PASSED (Conforms to Low Carbon London spec)' if first_file['has_expected_schema'] else 'FAILED'}

### Column Data Dictionary
1. `LCLid` (String): Unique identifier assigned to each household during the trial.
2. `stdorToU` (String): Tariff type (`Std` for Flat-rate 14.228p/kWh, `ToU` for Dynamic Time of Use).
3. `DateTime` (String): Half-hourly reading timestamp formatted as `YYYY-MM-DD HH:MM:SS.0000000`.
4. `KWH/hh (per half hour) ` (String/Float): Electricity consumed during the 30-minute interval in kilowatt-hours. Contains string `'Null'` for missing meter telemetry.

---

## 2. Partition Block Summary

| File Name | File Size (MB) | Record Count | Sample Households | Nulls in Sample |
| :--- | :--- | :--- | :--- | :--- |
"""

    for s in summaries:
        null_kwh = s['null_counts_in_sample'].get('KWH/hh (per half hour)', s['null_counts_in_sample'].get('consumption_kwh', 0))
        md_content += f"| `{s['filename']}` | {s['file_size_mb']} | {s['total_lines']:,} | {s['unique_households_in_sample']} | {null_kwh} |\n"

    md_content += """
---

## 3. Data Quality Findings & Ingestion Directives

1. **Trailing Whitespace in Headers:** Header `KWH/hh (per half hour) ` contains trailing whitespace. PySpark ingestion must sanitize and trim column names.
2. **Missing Value Representation:** Meter disconnections and transmission drops are represented as literal string `"Null"`. Preprocessing must convert these to numeric `NaN` / `0.0`.
3. **Timestamp Precision:** Timestamps include microsecond suffixes (`.0000000`). Preprocessing must parse them cleanly into UTC timestamps.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Inspection report saved to {report_path}")
    return report_data


if __name__ == "__main__":
    generate_inspection_report()

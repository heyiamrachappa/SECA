"""
Unit tests for Phase 2: Dataset Acquisition & Inspection.
Validates dataset discovery, column header detection, and schema compliance.
"""

from pathlib import Path
import pytest
from src.utils.config import load_config, get_project_root
from src.ingestion.inspect_dataset import find_csv_files, inspect_raw_csv, generate_inspection_report
from src.ingestion.sample_generator import create_sample_dataset


def test_sample_generator():
    """Verify that the sample dataset generator creates a valid CSV file."""
    root = get_project_root()
    sample_path = root / "dataset/sample/test_sample.csv"
    created = create_sample_dataset(output_path=str(sample_path), num_households=5, num_days=2, force_synthetic=True)
    assert created.exists()
    assert created.stat().st_size > 0

    # Cleanup
    if sample_path.exists():
        sample_path.unlink()


def test_inspect_csv_schema():
    """Verify that inspect_raw_csv correctly parses Low Carbon London headers and detects expected columns."""
    root = get_project_root()
    sample_file = root / "dataset/sample/london_smartmeter_sample.csv"
    if not sample_file.exists():
        create_sample_dataset()

    inspection = inspect_raw_csv(sample_file, sample_rows=100)
    assert inspection["has_expected_schema"] is True
    assert "LCLid" in inspection["clean_columns"]
    assert "DateTime" in inspection["clean_columns"]
    assert any("KWH" in col or "kwh" in col for col in inspection["clean_columns"])
    assert inspection["total_lines"] > 0


def test_inspection_report_generation():
    """Verify that generate_inspection_report creates the markdown summary."""
    root = get_project_root()
    report_file = "reports/test_inspection_report.md"
    result = generate_inspection_report(target_dir="dataset/sample", output_report=report_file)

    report_path = root / report_file
    assert report_path.exists()
    assert result["file_count"] >= 1
    assert result["total_records"] > 0

    # Cleanup
    if report_path.exists():
        report_path.unlink()

"""
Unit and Integration Tests for Custom Uploaded Dataset Pipeline.
Verifies schema auto-detection, non-standard column mapping, PySpark preprocessing,
11 SQL aggregations, adaptive ML forecasting, and anomaly detection.
"""

import json
import numpy as np
import pandas as pd
import pytest

from src.analytics.custom_pipeline import inspect_uploaded_file, run_custom_dataset_pipeline


def _make_sample_custom_csv(path, include_household=True, hours=72):
    """Generates a realistic multi-meter CSV with non-standard column names."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=hours, freq="1h")
    rows = []
    meters = ["MTR_A", "MTR_B"] if include_household else [None]

    for m in meters:
        for idx, ts in enumerate(timestamps):
            val = 1.2 + 0.6 * np.sin(2 * np.pi * ts.hour / 24.0) + rng.normal(0, 0.08)
            if idx == 20:
                val = 18.5  # Inject high consumption anomaly
            row = {
                "date_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "usage_kwh": "Null" if idx == 5 else f"{max(0.05, val):.3f}",
            }
            if include_household:
                row["meter_id"] = m
            rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)


def test_inspect_uploaded_file_custom_columns(tmp_path):
    """Verify auto-detection on non-standard electricity column names."""
    csv_path = tmp_path / "custom_energy.csv"
    _make_sample_custom_csv(csv_path, include_household=True, hours=24)

    info = inspect_uploaded_file(csv_path)
    assert info["filename"] == "custom_energy.csv"
    assert info["approx_rows"] == 48
    assert info["suggested_mapping"]["timestamp"] == "date_time"
    assert info["suggested_mapping"]["consumption"] == "usage_kwh"
    assert info["suggested_mapping"]["household_id"] == "meter_id"


def test_run_custom_dataset_pipeline_end_to_end(tmp_path):
    """Verify full analytics pipeline executes end-to-end on uploaded CSV."""
    csv_path = tmp_path / "user_upload.csv"
    summary_dir = tmp_path / "custom_summary"
    _make_sample_custom_csv(csv_path, include_household=True, hours=96)

    result = run_custom_dataset_pipeline(
        file_path=csv_path,
        timestamp_col="date_time",
        consumption_col="usage_kwh",
        household_col="meter_id",
        output_summary_dir=str(summary_dir),
    )

    assert result["overview"]["unique_households"] == 2
    assert result["overview"]["total_readings"] == 192
    assert result["forecast_metrics"] is not None
    assert result["anomaly_summary"] is not None

    for fname in [
        "overview_metrics.json",
        "hourly_consumption.csv",
        "daily_consumption.csv",
        "monthly_consumption.csv",
        "weekday_vs_weekend.csv",
        "peak_demand_periods.csv",
        "household_summary.csv",
        "forecasting_evaluation.json",
        "forecast_comparison.csv",
        "future_forecast.csv",
        "anomaly_summary.json",
        "detected_anomalies.csv",
        "custom_metadata.json",
    ]:
        assert (summary_dir / fname).exists(), f"Missing artifact: {fname}"


def test_custom_pipeline_without_household_column(tmp_path):
    """Verify single-meter CSV without household_id column is handled gracefully."""
    csv_path = tmp_path / "single_meter.csv"
    summary_dir = tmp_path / "single_summary"
    _make_sample_custom_csv(csv_path, include_household=False, hours=48)

    result = run_custom_dataset_pipeline(
        file_path=csv_path,
        timestamp_col="date_time",
        consumption_col="usage_kwh",
        household_col=None,
        output_summary_dir=str(summary_dir),
    )

    assert result["overview"]["unique_households"] == 1
    assert result["overview"]["peak_reading"]["household_id"] == "CUSTOM_METER_01"
    with open(summary_dir / "custom_metadata.json") as f:
        meta = json.load(f)
    assert meta["filename"] == "single_meter.csv"

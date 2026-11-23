"""
Unit tests for Phase 7: Anomaly Detection.
Tests statistical Z-score thresholding, Isolation Forest training, and diagnostic anomaly reporting.
"""

import pytest
import pandas as pd
from src.anomaly_detection.detector import AnomalyDetector


def test_anomaly_detector_pipeline():
    """Verify that anomaly detector loads data, applies Z-score & Isolation Forest, and flags outliers."""
    detector = AnomalyDetector()
    df = detector.load_telemetry_sample(sample_mode=True, max_rows=5000)
    assert len(df) > 0
    assert "consumption_kwh" in df.columns

    df_stat = detector.detect_statistical_zscore(df)
    assert "z_score" in df_stat.columns
    assert "stat_anomaly" in df_stat.columns

    df_anom = detector.detect_isolation_forest(df_stat)
    assert "iso_anomaly" in df_anom.columns
    assert "is_anomaly" in df_anom.columns
    assert "anomaly_category" in df_anom.columns

    summary = detector.generate_diagnostic_summary(df_anom)
    assert summary["total_records_analyzed"] == len(df)
    assert summary["total_anomalies"] >= 0
    assert "high_consumption_anomalies" in summary
    assert "low_consumption_anomalies" in summary

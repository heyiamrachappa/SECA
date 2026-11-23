"""
End-to-End Integration Tests for Smart Electricity Analytics (SECA).
Validates cross-module pipeline execution and artifact generation.
"""

import pytest
from pathlib import Path
from src.utils.config import get_project_root, load_config
from src.ingestion.pyspark_ingestion import run_ingestion
from src.preprocessing.preprocessor import run_preprocessing
from src.analytics.consumption_analytics import run_analytics
from src.forecasting.demand_forecaster import run_forecasting
from src.anomaly_detection.detector import run_anomaly_detection


def test_full_pipeline_sample_mode():
    """Verify that every module executes and generates valid artifacts in sample mode."""
    root = get_project_root()
    summary_dir = root / "dataset/processed/sample_analytics_summary"
    models_dir = root / "models"

    # Step 1: Preprocessing
    spark, df_clean, parquet_path = run_preprocessing(sample_mode=True)
    assert parquet_path.exists()
    assert df_clean.count() == 36000

    # Step 2: Analytics
    overview = run_analytics(sample_mode=True)
    spark.stop()
    assert overview["total_readings"] == 36000
    assert (summary_dir / "overview_metrics.json").exists()
    assert (summary_dir / "hourly_consumption.csv").exists()

    # Step 3: Forecasting
    fc_metrics = run_forecasting(sample_mode=True)
    assert "linear_regression" in fc_metrics
    assert "random_forest" in fc_metrics
    assert (models_dir / "random_forest.joblib").exists()
    assert (summary_dir / "forecast_comparison.csv").exists()

    # Step 4: Anomaly Detection
    anom_summary = run_anomaly_detection(sample_mode=True)
    assert anom_summary["total_anomalies"] > 0
    assert (summary_dir / "detected_anomalies.csv").exists()
    assert (models_dir / "isolation_forest.joblib").exists()

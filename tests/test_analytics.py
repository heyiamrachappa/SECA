"""
Unit tests for Phase 5: Electricity Consumption Analytics.
Tests Spark SQL aggregations, hourly, daily, peak-demand queries, and dashboard artifact export.
"""

import pytest
from src.utils.spark_session import get_spark_session
from src.analytics.consumption_analytics import ConsumptionAnalyticsEngine


@pytest.fixture(scope="module")
def spark():
    session = get_spark_session(app_name="TestAnalytics", master="local[2]", driver_memory="2g")
    yield session
    session.stop()


def test_consumption_aggregations(spark):
    """Verify that the analytics engine calculates correct global, hourly, and daily metrics."""
    engine = ConsumptionAnalyticsEngine(spark=spark)
    df_sample = engine.load_cleaned_data(sample_mode=True)

    overview = engine.compute_overview_metrics(df_sample)
    assert overview["total_readings"] == 36000
    assert overview["unique_households"] == 25
    assert overview["total_consumption_kwh"] > 0
    assert overview["avg_consumption_kwh"] > 0

    df_hourly = engine.compute_hourly_analytics(df_sample)
    assert df_hourly.count() == 24

    df_peaks = engine.compute_peak_demand_periods(df_sample, top_n=10)
    assert df_peaks.count() == 10


def test_export_artifacts(spark, tmp_path):
    """Verify that export_dashboard_artifacts writes valid files."""
    engine = ConsumptionAnalyticsEngine(spark=spark)
    df_sample = engine.load_cleaned_data(sample_mode=True)

    temp_dir = tmp_path / "analytics_test_artifacts"
    overview = engine.export_dashboard_artifacts(df_sample, output_dir=str(temp_dir))

    assert (temp_dir / "overview_metrics.json").exists()
    assert (temp_dir / "hourly_consumption.csv").exists()
    assert (temp_dir / "daily_consumption.csv").exists()
    assert (temp_dir / "peak_demand_periods.csv").exists()

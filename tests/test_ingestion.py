"""
Unit tests for Phase 3: PySpark Distributed Ingestion Pipeline.
Tests multi-file reading, column discovery, header standardization, and partition configuration.
"""

import pytest
from src.utils.spark_session import get_spark_session
from src.ingestion.pyspark_ingestion import IngestionPipeline


@pytest.fixture(scope="module")
def spark():
    """Shared SparkSession fixture for testing."""
    session = get_spark_session(app_name="TestIngestion", master="local[2]", driver_memory="2g")
    yield session
    session.stop()


def test_discover_and_normalize_columns(spark):
    """Verify that columns with whitespace or alternate casings are standardized."""
    pipeline = IngestionPipeline(spark=spark)
    raw_data = [("MAC0001", "Std", "2013-01-01 00:00:00.000", "0.25")]
    raw_columns = ["LCLid", "stdorToU", "DateTime", "KWH/hh (per half hour) "]

    raw_df = spark.createDataFrame(raw_data, raw_columns)
    normalized_df = pipeline.discover_and_normalize_columns(raw_df)

    expected_cols = ["household_id", "tariff_type", "raw_timestamp", "raw_consumption"]
    assert normalized_df.columns == expected_cols


def test_load_sample_dataset(spark):
    """Verify that sample CSV is ingested correctly and yields expected record count."""
    pipeline = IngestionPipeline(spark=spark)
    df = pipeline.load_dataset(sample_mode=True)
    summary = pipeline.get_dataset_summary(df)

    assert summary["record_count"] == 36000
    assert summary["partition_count"] >= 1
    assert "household_id" in summary["columns"]
    assert "raw_consumption" in summary["columns"]

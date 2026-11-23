"""
Unit tests for Phase 4: Data Preprocessing and Feature Engineering.
Tests missing-value handling, physical range validation, timestamp parsing,
calendar feature engineering, and Parquet IO.
"""

import pytest
from pyspark.sql.types import DoubleType, TimestampType
from src.utils.spark_session import get_spark_session
from src.preprocessing.preprocessor import DataPreprocessor


@pytest.fixture(scope="module")
def spark():
    """Shared SparkSession fixture for preprocessing tests."""
    session = get_spark_session(app_name="TestPreprocessing", master="local[2]", driver_memory="2g")
    yield session
    session.stop()


def test_clean_and_transform_features(spark):
    """Verify that preprocessing converts Nulls, filters unphysical values, and engineers features."""
    preprocessor = DataPreprocessor(spark=spark)

    test_data = [
        ("MAC0001", "Std", "2013-05-10 14:30:00.0000000", "0.452"),
        ("MAC0001", "Std", "2013-05-10 15:00:00.0000000", "Null"),  # Missing value
        ("MAC0001", "Std", "2013-05-10 15:30:00.0000000", " 0.12 "),  # Whitespace
        ("MAC0001", "Std", "2013-05-10 16:00:00.0000000", "999.0"),  # Unphysical (> 25.0)
        ("MAC0001", "Std", "2013-05-10 14:30:00.0000000", "0.452"),  # Duplicate
    ]
    columns = ["household_id", "tariff_type", "raw_timestamp", "raw_consumption"]
    raw_df = spark.createDataFrame(test_data, columns)

    df_clean, metrics = preprocessor.clean_and_transform(raw_df)

    # Assertions
    assert metrics["final_clean_records"] == 3  # 999.0 filtered, duplicate dropped
    assert "hour" in df_clean.columns
    assert "is_weekend" in df_clean.columns
    assert "week_of_year" in df_clean.columns

    # Verify 'Null' was replaced with 0.0
    null_row = df_clean.filter(df_clean.hour == 15).collect()[0]
    assert null_row["consumption_kwh"] == 0.0


def test_parquet_roundtrip(spark):
    """Verify that saved Parquet file exists, has correct schema, and can be queried."""
    preprocessor = DataPreprocessor(spark=spark)
    test_data = [("MAC0002", "ToU", "2013-06-01 08:00:00.0000000", "0.85")]
    raw_df = spark.createDataFrame(test_data, ["household_id", "tariff_type", "raw_timestamp", "raw_consumption"])

    df_clean, _ = preprocessor.clean_and_transform(raw_df)
    parquet_path = preprocessor.save_to_parquet(df_clean, output_subpath="test_temp.parquet")

    assert parquet_path.exists()
    loaded_df = spark.read.parquet(str(parquet_path))
    assert loaded_df.count() == 1
    assert loaded_df.first()["household_id"] == "MAC0002"

    # Cleanup temp parquet
    import shutil
    shutil.rmtree(parquet_path, ignore_errors=True)

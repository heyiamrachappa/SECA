"""
PySpark Ingestion Pipeline for Low Carbon London SmartMeter dataset.
Provides distributed multi-file loading, schema discovery, column standardization,
and partition management without memory saturation.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

from src.utils.config import load_config, get_project_root
from src.utils.logger import get_logger
from src.utils.spark_session import get_spark_session

logger = get_logger("PySparkIngestion")


class IngestionPipeline:
    """
    Distributed data ingestion engine utilizing Apache Spark.
    Handles multiple CSV block files, normalizes schema headers,
    and returns an optimized Spark DataFrame.
    """

    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_spark_session(app_name="SECA_Ingestion")
        self.config = load_config()

    def discover_and_normalize_columns(self, df: DataFrame) -> DataFrame:
        """
        Dynamically inspects raw DataFrame column names, strips whitespaces,
        and renames them to standard canonical names:
        - household_id
        - tariff_type
        - raw_timestamp
        - raw_consumption
        """
        raw_cols = df.columns
        logger.info(f"Raw DataFrame column headers detected: {raw_cols}")

        col_mapping = {}
        for col_name in raw_cols:
            clean_name = col_name.strip()
            # Standardize by pattern matching
            if clean_name.lower() in ["lclid", "household_id", "householdid"]:
                col_mapping[col_name] = "household_id"
            elif clean_name.lower() in ["stdortou", "tariff", "tarifftype"]:
                col_mapping[col_name] = "tariff_type"
            elif clean_name.lower() in ["datetime", "timestamp", "tstp", "date_time"]:
                col_mapping[col_name] = "raw_timestamp"
            elif any(k in clean_name.lower() for k in ["kwh", "energy", "consumption"]):
                col_mapping[col_name] = "raw_consumption"
            else:
                col_mapping[col_name] = clean_name.lower()

        logger.info(f"Standardized column mapping: {col_mapping}")

        # Rename columns in PySpark
        standardized_df = df
        for old_col, new_col in col_mapping.items():
            standardized_df = standardized_df.withColumnRenamed(old_col, new_col)

        # Validate mandatory columns exist
        required_cols = {"household_id", "raw_timestamp", "raw_consumption"}
        existing_cols = set(standardized_df.columns)
        missing_cols = required_cols - existing_cols
        if missing_cols:
            raise ValueError(f"Missing mandatory columns after normalization: {missing_cols}")

        if "tariff_type" not in existing_cols:
            logger.warning("tariff_type not present in dataset; assigning default 'Std'")
            standardized_df = standardized_df.withColumn("tariff_type", F.lit("Std"))

        # Select standard columns in deterministic order
        return standardized_df.select("household_id", "tariff_type", "raw_timestamp", "raw_consumption")

    def load_dataset(
        self,
        input_path: Optional[str] = None,
        sample_mode: bool = False,
        repartition: Optional[int] = None
    ) -> DataFrame:
        """
        Loads CSV dataset files using PySpark distributed reader.

        Parameters:
            input_path (str, optional): Custom file or directory path.
            sample_mode (bool): If True, loads sample data from dataset/sample/.
            repartition (int, optional): Explicit partition count for downstream tasks.

        Returns:
            DataFrame: Standardized PySpark DataFrame.
        """
        root = get_project_root()
        if input_path:
            target_path = Path(input_path) if os.path.isabs(input_path) else root / input_path
        elif sample_mode:
            target_path = root / self.config["paths"]["sample_data_dir"]
        else:
            target_path = root / self.config["paths"]["raw_data_dir"]

        logger.info(f"Target ingestion location: {target_path}")

        # Determine path pattern for CSVs
        if target_path.is_file():
            path_pattern = str(target_path)
        elif target_path.is_dir():
            path_pattern = str(target_path / "*.csv")
        else:
            raise FileNotFoundError(f"Provided path does not exist: {target_path}")

        logger.info(f"Reading CSV pattern with Spark: {path_pattern}")

        # Distributed read with header, all columns read as StringType initially for fault-tolerance
        df_raw = (
            self.spark.read
            .option("header", "true")
            .option("mode", "PERMISSIVE")
            .option("ignoreLeadingWhiteSpace", "true")
            .option("ignoreTrailingWhiteSpace", "true")
            .csv(path_pattern)
        )

        initial_partitions = df_raw.rdd.getNumPartitions()
        logger.info(f"Initial Spark RDD partition count: {initial_partitions}")

        # Standardize column headers
        df_standardized = self.discover_and_normalize_columns(df_raw)

        # Repartition if requested or if partitions are suboptimal
        target_partitions = repartition or int(self.config["spark"].get("sql_shuffle_partitions", 16))
        if initial_partitions < target_partitions and not sample_mode:
            logger.info(f"Repartitioning DataFrame to {target_partitions} partitions...")
            df_standardized = df_standardized.repartition(target_partitions)

        return df_standardized

    def get_dataset_summary(self, df: DataFrame) -> Dict[str, Any]:
        """
        Calculates non-blocking dataset metadata and dimensions.
        """
        partitions = df.rdd.getNumPartitions()
        columns = df.columns
        schema_dict = {f.name: str(f.dataType) for f in df.schema.fields}
        record_count = df.count()
        sample_rows = [row.asDict() for row in df.limit(5).collect()]

        summary = {
            "record_count": record_count,
            "partition_count": partitions,
            "columns": columns,
            "schema": schema_dict,
            "sample_rows": sample_rows
        }

        logger.info(f"Ingested {record_count:,} records across {partitions} partitions.")
        return summary


def run_ingestion(sample_mode: bool = False, custom_path: Optional[str] = None) -> Tuple[SparkSession, DataFrame]:
    """Helper runner function to ingest data and log summary."""
    spark = get_spark_session(app_name="SECA_Ingestion")
    pipeline = IngestionPipeline(spark=spark)
    df = pipeline.load_dataset(input_path=custom_path, sample_mode=sample_mode)
    summary = pipeline.get_dataset_summary(df)

    print("\n=== [SECA] PYSPARK INGESTION SUMMARY ===")
    print(f"Total Ingested Records: {summary['record_count']:,}")
    print(f"Partitions: {summary['partition_count']}")
    print(f"Standardized Schema: {summary['schema']}")
    print("Sample Ingested Rows:")
    for row in summary["sample_rows"][:3]:
        print(f"  {row}")
    print("=========================================\n")

    return spark, df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PySpark Data Ingestion Module")
    parser.add_argument("--sample-mode", action="store_true", help="Ingest sample dataset")
    parser.add_argument("--path", type=str, default=None, help="Custom CSV path or directory")
    args = parser.parse_args()

    spark, df = run_ingestion(sample_mode=args.sample_mode, custom_path=args.path)
    spark.stop()

"""
Data Preprocessing and Feature Engineering module for Smart Electricity Analytics.
Cleans raw telemetry, removes duplicates, casts timestamps, engineers temporal features,
and exports optimized Parquet partitions.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, TimestampType, IntegerType

from src.utils.config import load_config, get_project_root
from src.utils.logger import get_logger
from src.utils.spark_session import get_spark_session
from src.ingestion.pyspark_ingestion import IngestionPipeline

logger = get_logger("DataPreprocessor")


class DataPreprocessor:
    """
    Spark-based distributed preprocessing engine.
    Ensures data consistency, validates sensor ranges, handles 'Null' meter tokens,
    generates calendar features, and writes Snappy-compressed Parquet.
    """

    def __init__(self, spark: Optional[SparkSession] = None, max_kwh: Optional[float] = None):
        self.spark = spark or get_spark_session(app_name="SECA_Preprocessor")
        self.config = load_config()
        self.prep_cfg = self.config.get("preprocessing", {})
        self.max_kwh = float(max_kwh) if max_kwh is not None else float(self.prep_cfg.get("max_valid_kwh_half_hour", 25.0))
        self.min_kwh = float(self.prep_cfg.get("min_valid_kwh_half_hour", 0.0))
        self.null_replacement = float(self.prep_cfg.get("null_replacement", 0.0))

    def clean_and_transform(self, df_raw: DataFrame) -> Tuple[DataFrame, Dict[str, Any]]:
        """
        Executes end-to-end preprocessing pipeline on raw ingested DataFrame.

        Parameters:
            df_raw (DataFrame): Ingested raw DataFrame.

        Returns:
            Tuple[DataFrame, Dict[str, Any]]: (Cleaned DataFrame, Quality Metrics Dict)
        """
        initial_count = df_raw.count()
        logger.info(f"Beginning preprocessing on {initial_count:,} raw records...")

        # 1. Clean Consumption Column
        # Real London data contains 'Null', ' 0 ', empty spaces, or trailing spaces
        trimmed_kwh = F.trim(F.col("raw_consumption").cast("string"))
        is_null_token = (
            trimmed_kwh.isNull() |
            (trimmed_kwh == "") |
            (F.lower(trimmed_kwh) == "null") |
            (F.lower(trimmed_kwh) == "nan") |
            (F.lower(trimmed_kwh) == "na")
        )

        # Cast to double safely (tolerate malformed strings under Spark 4 ANSI mode)
        df_kwh = df_raw.withColumn(
            "consumption_kwh",
            F.when(is_null_token, F.lit(self.null_replacement))
            .otherwise(F.expr("try_cast(trim(cast(raw_consumption as string)) as double)"))
        )

        # 2. Filter unphysical / corrupted readings (e.g. negative or > max_kwh)
        df_valid_kwh = df_kwh.filter(
            (F.col("consumption_kwh") >= self.min_kwh) &
            (F.col("consumption_kwh") <= self.max_kwh) &
            F.col("consumption_kwh").isNotNull()
        )

        # 3. Timestamp conversion (uses try_to_timestamp with F.lit so Spark 4 ANSI mode falls through cleanly)
        trimmed_ts = F.trim(F.col("raw_timestamp").cast("string"))
        ts_formats = [
            "yyyy-MM-dd HH:mm:ss.SSSSSSS",
            "yyyy-MM-dd HH:mm:ss",
            "yyyy-MM-dd'T'HH:mm:ss",
            "yyyy-MM-dd HH:mm",
            "dd/MM/yyyy HH:mm:ss",
            "dd/MM/yyyy HH:mm",
            "dd-MM-yyyy HH:mm:ss",
            "MM/dd/yyyy HH:mm:ss",
            "yyyy/MM/dd HH:mm:ss",
            "yyyy-MM-dd",
        ]
        if hasattr(F, "try_to_timestamp"):
            ts_exprs = [F.try_to_timestamp(trimmed_ts, F.lit(fmt)) for fmt in ts_formats] + [F.try_to_timestamp(trimmed_ts)]
        else:
            ts_exprs = [F.to_timestamp(trimmed_ts, fmt) for fmt in ts_formats] + [F.to_timestamp(trimmed_ts)]

        df_ts = df_valid_kwh.withColumn("timestamp", F.coalesce(*ts_exprs)).filter(F.col("timestamp").isNotNull())

        # 4. Deduplication by household_id and timestamp
        df_dedup = df_ts.dropDuplicates(["household_id", "timestamp"])

        # 5. Temporal Feature Engineering
        df_features = (
            df_dedup
            .withColumn("date", F.to_date(F.col("timestamp")))
            .withColumn("year", F.year(F.col("timestamp")))
            .withColumn("month", F.month(F.col("timestamp")))
            .withColumn("day", F.dayofmonth(F.col("timestamp")))
            .withColumn("hour", F.hour(F.col("timestamp")))
            .withColumn("minute", F.minute(F.col("timestamp")))
            .withColumn("week_of_year", F.weekofyear(F.col("timestamp")))
            .withColumn("day_of_week", F.dayofweek(F.col("timestamp")))  # 1=Sun, 2=Mon, ..., 7=Sat
            .withColumn("day_name", F.date_format(F.col("timestamp"), "EEEE"))
            .withColumn(
                "is_weekend",
                F.when(F.col("day_of_week").isin(1, 7), F.lit(1)).otherwise(F.lit(0))
            )
        )

        # Select final structured schema
        final_cols = [
            "household_id",
            "tariff_type",
            "timestamp",
            "date",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "week_of_year",
            "day_of_week",
            "day_name",
            "is_weekend",
            "consumption_kwh"
        ]
        df_clean = df_features.select(final_cols)

        final_count = df_clean.count()
        dropped_count = initial_count - final_count

        quality_metrics = {
            "initial_records": initial_count,
            "final_clean_records": final_count,
            "dropped_or_deduped_records": dropped_count,
            "retention_rate_pct": round((final_count / initial_count * 100.0) if initial_count > 0 else 0, 3),
            "max_consumption_kwh": self.max_kwh,
            "min_consumption_kwh": self.min_kwh
        }

        logger.info(
            f"Preprocessing completed: {final_count:,}/{initial_count:,} records retained "
            f"({quality_metrics['retention_rate_pct']}%)"
        )
        return df_clean, quality_metrics

    def save_to_parquet(
        self,
        df: DataFrame,
        output_subpath: Optional[str] = None,
        partition_by_year_month: bool = False
    ) -> Path:
        """
        Saves cleaned DataFrame in columnar Apache Parquet format.
        """
        root = get_project_root()
        proc_dir = root / self.config["paths"]["processed_data_dir"]
        proc_dir.mkdir(parents=True, exist_ok=True)

        target_parquet = proc_dir / (output_subpath or "cleaned_energy_data.parquet")
        logger.info(f"Writing Parquet dataset to: {target_parquet}")

        writer = (
            df.write
            .mode("overwrite")
            .option("compression", "snappy")
        )

        if partition_by_year_month and "year" in df.columns and "month" in df.columns:
            writer = writer.partitionBy("year", "month")

        writer.parquet(str(target_parquet))
        logger.info(f"Parquet write completed successfully at {target_parquet}")
        return target_parquet

    def write_quality_report(
        self,
        metrics: Dict[str, Any],
        output_report: str = "reports/data_quality_report.md"
    ) -> Path:
        """Generates a structured markdown report summarizing preprocessing quality metrics."""
        root = get_project_root()
        report_path = root / output_report
        report_path.parent.mkdir(parents=True, exist_ok=True)

        content = f"""# Data Quality and Preprocessing Report

**Generated Timestamp:** {F.current_timestamp()}  
**Dataset Name:** {self.config['dataset']['name']}  

---

## 1. Quality & Cleansing Metrics

| Metric | Value |
| :--- | :--- |
| **Initial Telemetry Records** | {metrics['initial_records']:,} |
| **Cleaned & Verified Records** | {metrics['final_clean_records']:,} |
| **Filtered / Deduplicated Records** | {metrics['dropped_or_deduped_records']:,} |
| **Data Retention Rate** | **{metrics['retention_rate_pct']}%** |
| **Consumption Validation Range** | [{metrics['min_consumption_kwh']} kWh, {metrics['max_consumption_kwh']} kWh] |

---

## 2. Transformations Applied

1. **Missing Value Resolution:** String `'Null'` and blank entries converted to numeric `{self.null_replacement}`.
2. **Physical Boundary Enforcement:** Half-hourly consumption constrained to `{self.min_kwh}` - `{self.max_kwh}` kWh.
3. **Timestamp Standardization:** Converted to native Spark `TimestampType` (UTC).
4. **Duplicate Purging:** Strict deduplication on `(household_id, timestamp)`.
5. **Engineered Features:** Added `hour`, `day`, `month`, `year`, `week_of_year`, `day_of_week`, `day_name`, and `is_weekend`.
6. **Storage Optimization:** Exported to Snappy-compressed Apache Parquet format for fast vectorized downstream query execution.
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Data quality report saved to {report_path}")
        return report_path


def run_preprocessing(sample_mode: bool = False) -> Tuple[SparkSession, DataFrame, Path]:
    """Helper orchestrator function for Phase 4 preprocessing."""
    spark = get_spark_session(app_name="SECA_Preprocessing")
    ingest_pipeline = IngestionPipeline(spark=spark)

    # Ingest data
    df_raw = ingest_pipeline.load_dataset(sample_mode=sample_mode)

    # Preprocess
    preprocessor = DataPreprocessor(spark=spark)
    df_clean, metrics = preprocessor.clean_and_transform(df_raw)

    # Save to Parquet
    parquet_filename = "sample_cleaned_energy_data.parquet" if sample_mode else "cleaned_energy_data.parquet"
    parquet_path = preprocessor.save_to_parquet(df_clean, output_subpath=parquet_filename)

    # Write Quality Report
    report_filename = "reports/sample_data_quality_report.md" if sample_mode else "reports/data_quality_report.md"
    preprocessor.write_quality_report(metrics, output_report=report_filename)

    print("\n=== [SECA] PREPROCESSING COMPLETE ===")
    print(f"Cleaned Records: {metrics['final_clean_records']:,}")
    print(f"Retention Rate: {metrics['retention_rate_pct']}%")
    print(f"Parquet Path: {parquet_path}")
    print(f"Quality Report: {report_filename}\n")

    return spark, df_clean, parquet_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PySpark Data Preprocessing Module")
    parser.add_argument("--sample-mode", action="store_true", help="Preprocess sample dataset")
    args = parser.parse_args()

    spark, df_clean, parquet_path = run_preprocessing(sample_mode=args.sample_mode)
    spark.stop()

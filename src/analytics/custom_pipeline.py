"""
Custom Dataset Pipeline Runner for Smart Electricity Analytics.
Allows users to upload arbitrary CSV or Parquet electricity datasets via the dashboard,
automatically detects or maps schema columns, and executes the complete PySpark + ML +
Anomaly Detection workflow into an isolated summary directory without altering defaults.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import pandas as pd
from pyspark.sql import functions as F

from src.utils.config import get_project_root
from src.utils.logger import get_logger
from src.utils.spark_session import get_spark_session
from src.preprocessing.preprocessor import DataPreprocessor
from src.analytics.consumption_analytics import ConsumptionAnalyticsEngine
from src.forecasting.demand_forecaster import run_forecasting
from src.anomaly_detection.detector import run_anomaly_detection

logger = get_logger("CustomDatasetPipeline")

CUSTOM_PARQUET_SUBPATH = "custom_cleaned_energy_data.parquet"
CUSTOM_SUMMARY_SUBDIR = "dataset/processed/custom_analytics_summary"


def inspect_uploaded_file(file_path: Path) -> Dict[str, Any]:
    """
    Reads header columns and a lightweight preview of an uploaded CSV or Parquet file,
    returning auto-detected column mappings for timestamp, consumption, household, and tariff.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(file_path))
        columns = pf.schema_arrow.names
        preview_df = pq.read_table(str(file_path)).slice(0, 5).to_pandas()
        approx_rows = pf.metadata.num_rows
    elif suffix == ".csv":
        preview_df = pd.read_csv(file_path, nrows=5, dtype=str)
        columns = list(preview_df.columns)
        with open(file_path, "rb") as f:
            approx_rows = max(0, sum(1 for _ in f) - 1)
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Please upload a .csv or .parquet file.")

    # Auto-detect canonical roles
    def _match_col(candidates):
        for col in columns:
            c_norm = col.strip().lower()
            if c_norm in candidates:
                return col
        for col in columns:
            c_norm = col.strip().lower()
            if any(cand in c_norm for cand in candidates):
                return col
        return None

    detected_ts = _match_col(["datetime", "timestamp", "tstp", "date_time", "time", "date", "ds"])
    detected_kwh = _match_col([
        "kwh/hh (per half hour)", "consumption_kwh", "kwh", "energy",
        "consumption", "usage", "load", "power", "active_power", "value"
    ])
    detected_hh = _match_col(["lclid", "household_id", "householdid", "meter_id", "meter", "id", "home_id", "user_id"])
    detected_tariff = _match_col(["stdortou", "tariff_type", "tariff", "plan"])

    return {
        "filename": file_path.name,
        "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
        "approx_rows": int(approx_rows),
        "columns": columns,
        "preview": preview_df,
        "suggested_mapping": {
            "timestamp": detected_ts or (columns[0] if columns else None),
            "consumption": detected_kwh or (columns[-1] if columns else None),
            "household_id": detected_hh,
            "tariff_type": detected_tariff,
        },
    }


def run_custom_dataset_pipeline(
    file_path: Path,
    timestamp_col: str,
    consumption_col: str,
    household_col: Optional[str] = None,
    tariff_col: Optional[str] = None,
    max_kwh: float = 10000.0,
    output_summary_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> Dict[str, Any]:
    """
    Executes end-to-end Ingestion -> Preprocessing -> Spark SQL Analytics ->
    Demand Forecasting -> Anomaly Detection for a user-uploaded dataset.
    """
    root = get_project_root()
    summary_dir_rel = output_summary_dir or CUSTOM_SUMMARY_SUBDIR
    summary_dir = root / summary_dir_rel if not Path(summary_dir_rel).is_absolute() else Path(summary_dir_rel)
    summary_dir.mkdir(parents=True, exist_ok=True)

    def _notify(msg: str, pct: int):
        logger.info(f"[{pct}%] {msg}")
        if progress_callback:
            progress_callback(msg, pct)

    _notify("Initializing Apache Spark & ingesting uploaded file...", 10)
    spark = get_spark_session(app_name="SECA_CustomDataset")

    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        df_raw = spark.read.parquet(str(file_path))
    else:
        df_raw = (
            spark.read
            .option("header", "true")
            .option("mode", "PERMISSIVE")
            .option("ignoreLeadingWhiteSpace", "true")
            .option("ignoreTrailingWhiteSpace", "true")
            .csv(str(file_path))
        )

    # Map user-selected columns to canonical ingestion schema
    cols_available = set(df_raw.columns)
    if timestamp_col not in cols_available:
        raise ValueError(f"Selected timestamp column '{timestamp_col}' not found in dataset.")
    if consumption_col not in cols_available:
        raise ValueError(f"Selected consumption column '{consumption_col}' not found in dataset.")

    df_mapped = (
        df_raw
        .withColumn("raw_timestamp", F.col(f"`{timestamp_col}`"))
        .withColumn("raw_consumption", F.col(f"`{consumption_col}`"))
    )

    if household_col and household_col in cols_available:
        df_mapped = df_mapped.withColumn("household_id", F.col(f"`{household_col}`").cast("string"))
    else:
        df_mapped = df_mapped.withColumn("household_id", F.lit("CUSTOM_METER_01"))

    if tariff_col and tariff_col in cols_available:
        df_mapped = df_mapped.withColumn("tariff_type", F.col(f"`{tariff_col}`").cast("string"))
    else:
        df_mapped = df_mapped.withColumn("tariff_type", F.lit("Custom"))

    df_standardized = df_mapped.select("household_id", "tariff_type", "raw_timestamp", "raw_consumption")

    _notify("Running PySpark cleaning, timestamp casting & feature engineering...", 30)
    preprocessor = DataPreprocessor(spark=spark, max_kwh=max_kwh)
    df_clean, quality_metrics = preprocessor.clean_and_transform(df_standardized)

    if quality_metrics["final_clean_records"] == 0:
        raise ValueError(
            "Zero valid rows remained after cleaning. Please verify the selected Timestamp and Consumption columns."
        )

    parquet_path = preprocessor.save_to_parquet(df_clean, output_subpath=CUSTOM_PARQUET_SUBPATH)

    _notify("Computing 11 Spark SQL consumption aggregations...", 55)
    analytics_engine = ConsumptionAnalyticsEngine(spark=spark)
    overview = analytics_engine.export_dashboard_artifacts(df_clean, output_dir=str(summary_dir))

    _notify("Training Linear Regression & Random Forest demand forecasters...", 75)
    warnings_list = []
    fc_metrics = None
    try:
        fc_metrics = run_forecasting(
            custom_parquet=parquet_path,
            output_dir=str(summary_dir),
            save_models=False
        )
    except Exception as fc_err:
        logger.warning(f"Forecasting skipped or partial for custom dataset: {fc_err}")
        warnings_list.append(f"Forecasting skipped: {fc_err}")
        # Clean stale forecast artifacts if any
        for stale in ["forecasting_evaluation.json", "forecast_comparison.csv", "future_forecast.csv", "feature_importance.csv"]:
            p = summary_dir / stale
            if p.exists():
                p.unlink()

    _notify("Executing Z-Score & Isolation Forest anomaly detection...", 90)
    anom_summary = None
    try:
        anom_summary = run_anomaly_detection(
            custom_parquet=parquet_path,
            output_dir=str(summary_dir),
            save_model=False
        )
    except Exception as anom_err:
        logger.warning(f"Anomaly detection skipped for custom dataset: {anom_err}")
        warnings_list.append(f"Anomaly detection skipped: {anom_err}")

    # Save custom metadata for dashboard context
    custom_meta = {
        "filename": file_path.name,
        "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
        "column_mapping": {
            "timestamp": timestamp_col,
            "consumption": consumption_col,
            "household_id": household_col or "(Synthesized: CUSTOM_METER_01)",
            "tariff_type": tariff_col or "(Default: Custom)",
        },
        "quality_metrics": quality_metrics,
        "warnings": warnings_list,
    }
    with open(summary_dir / "custom_metadata.json", "w", encoding="utf-8") as f:
        json.dump(custom_meta, f, indent=2)

    _notify("Custom dataset analytics complete!", 100)
    return {
        "overview": overview,
        "quality_metrics": quality_metrics,
        "forecast_metrics": fc_metrics,
        "anomaly_summary": anom_summary,
        "custom_meta": custom_meta,
        "summary_dir": str(summary_dir),
    }

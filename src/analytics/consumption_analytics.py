"""
Electricity Consumption Analytics and Spark SQL Aggregation Engine.
Executes 11 core analytical queries on cleaned Parquet telemetry
and exports compact, dashboard-ready analytical summaries.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from src.utils.config import load_config, get_project_root
from src.utils.logger import get_logger
from src.utils.spark_session import get_spark_session

logger = get_logger("ConsumptionAnalytics")


class ConsumptionAnalyticsEngine:
    """
    High-performance analytical engine powered by PySpark DataFrame API and Spark SQL.
    Computes grid-level and household-level temporal aggregations, peak demand indicators,
    and diurnal load shapes.
    """

    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_spark_session(app_name="SECA_Analytics")
        self.config = load_config()

    def load_cleaned_data(self, sample_mode: bool = False, custom_parquet: Optional[str] = None) -> DataFrame:
        """Loads cleaned Parquet dataset into a Spark DataFrame."""
        root = get_project_root()
        proc_dir = root / self.config["paths"]["processed_data_dir"]

        if custom_parquet:
            parquet_path = Path(custom_parquet) if os.path.isabs(custom_parquet) else root / custom_parquet
        elif sample_mode:
            parquet_path = proc_dir / "sample_cleaned_energy_data.parquet"
        else:
            parquet_path = proc_dir / "cleaned_energy_data.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(f"Cleaned Parquet dataset not found at {parquet_path}. Run preprocessing first.")

        logger.info(f"Loading cleaned Parquet data from: {parquet_path}")
        df = self.spark.read.parquet(str(parquet_path))
        # Register temporary view for Spark SQL queries
        df.createOrReplaceTempView("smart_meter_readings")
        return df

    def compute_overview_metrics(self, df: DataFrame) -> Dict[str, Any]:
        """Query 1, 2, 3: Total, average, min, max consumption and scope metrics."""
        logger.info("Computing global consumption metrics...")
        agg_row = df.select(
            F.count("*").alias("total_readings"),
            F.countDistinct("household_id").alias("unique_households"),
            F.sum("consumption_kwh").alias("total_consumption_kwh"),
            F.avg("consumption_kwh").alias("avg_consumption_kwh"),
            F.min("consumption_kwh").alias("min_consumption_kwh"),
            F.max("consumption_kwh").alias("max_consumption_kwh"),
            F.min("timestamp").alias("min_timestamp"),
            F.max("timestamp").alias("max_timestamp")
        ).collect()[0]

        # Identify absolute peak demand record
        peak_row = df.orderBy(F.col("consumption_kwh").desc()).first()

        metrics = {
            "total_readings": int(agg_row["total_readings"]),
            "unique_households": int(agg_row["unique_households"]),
            "total_consumption_kwh": round(float(agg_row["total_consumption_kwh"] or 0), 2),
            "avg_consumption_kwh": round(float(agg_row["avg_consumption_kwh"] or 0), 4),
            "min_consumption_kwh": round(float(agg_row["min_consumption_kwh"] or 0), 4),
            "max_consumption_kwh": round(float(agg_row["max_consumption_kwh"] or 0), 4),
            "date_start": str(agg_row["min_timestamp"]),
            "date_end": str(agg_row["max_timestamp"]),
            "peak_reading": {
                "household_id": peak_row["household_id"] if peak_row else None,
                "timestamp": str(peak_row["timestamp"]) if peak_row else None,
                "kwh": round(float(peak_row["consumption_kwh"]), 4) if peak_row else None
            }
        }
        return metrics

    def compute_hourly_analytics(self, df: DataFrame) -> DataFrame:
        """Query 8: Hourly consumption profile across the 24-hour cycle."""
        logger.info("Computing diurnal hourly profile (Query 8)...")
        return (
            df.groupBy("hour")
            .agg(
                F.avg("consumption_kwh").alias("avg_kwh"),
                F.sum("consumption_kwh").alias("total_kwh"),
                F.min("consumption_kwh").alias("min_kwh"),
                F.max("consumption_kwh").alias("max_kwh"),
                F.stddev("consumption_kwh").alias("std_kwh")
            )
            .orderBy("hour")
        )

    def compute_daily_analytics(self, df: DataFrame) -> DataFrame:
        """Query 5: Daily consumption trend over time."""
        logger.info("Computing daily consumption aggregations (Query 5)...")
        return (
            df.groupBy("date", "day_name", "is_weekend")
            .agg(
                F.sum("consumption_kwh").alias("total_kwh"),
                F.avg("consumption_kwh").alias("avg_kwh"),
                F.max("consumption_kwh").alias("peak_half_hour_kwh"),
                F.countDistinct("household_id").alias("active_households")
            )
            .orderBy("date")
        )

    def compute_weekly_analytics(self, df: DataFrame) -> DataFrame:
        """Query 6: Weekly consumption patterns."""
        logger.info("Computing weekly aggregations (Query 6)...")
        return (
            df.groupBy("year", "week_of_year")
            .agg(
                F.sum("consumption_kwh").alias("total_kwh"),
                F.avg("consumption_kwh").alias("avg_kwh"),
                F.countDistinct("household_id").alias("active_households")
            )
            .orderBy("year", "week_of_year")
        )

    def compute_monthly_analytics(self, df: DataFrame) -> DataFrame:
        """Query 7: Monthly consumption trends."""
        logger.info("Computing monthly aggregations (Query 7)...")
        return (
            df.groupBy("year", "month")
            .agg(
                F.sum("consumption_kwh").alias("total_kwh"),
                F.avg("consumption_kwh").alias("avg_kwh"),
                F.max("consumption_kwh").alias("peak_kwh"),
                F.countDistinct("household_id").alias("active_households")
            )
            .orderBy("year", "month")
        )

    def compute_weekday_vs_weekend(self, df: DataFrame) -> DataFrame:
        """Query 9: Weekday versus Weekend hourly load comparison."""
        logger.info("Computing weekday vs weekend profiles (Query 9)...")
        return (
            df.withColumn(
                "day_type",
                F.when(F.col("is_weekend") == 1, F.lit("Weekend")).otherwise(F.lit("Weekday"))
            )
            .groupBy("day_type", "hour")
            .agg(
                F.avg("consumption_kwh").alias("avg_kwh"),
                F.sum("consumption_kwh").alias("total_kwh"),
                F.stddev("consumption_kwh").alias("std_kwh")
            )
            .orderBy("day_type", "hour")
        )

    def compute_peak_demand_periods(self, df: DataFrame, top_n: int = 50) -> DataFrame:
        """Query 10: Identify absolute peak-demand intervals across the grid."""
        logger.info("Computing peak demand periods (Query 10)...")
        return (
            df.groupBy("timestamp", "hour", "day_name")
            .agg(
                F.sum("consumption_kwh").alias("grid_total_kwh"),
                F.avg("consumption_kwh").alias("grid_avg_kwh"),
                F.countDistinct("household_id").alias("reporting_households")
            )
            .orderBy(F.col("grid_total_kwh").desc())
            .limit(top_n)
        )

    def compute_household_analytics(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """Query 4 & 11: Household-wise consumption summaries and comparisons."""
        logger.info("Computing household-level aggregations and profiles (Query 4 & 11)...")
        hh_summary = (
            df.groupBy("household_id", "tariff_type")
            .agg(
                F.sum("consumption_kwh").alias("total_kwh"),
                F.avg("consumption_kwh").alias("avg_kwh"),
                F.min("consumption_kwh").alias("min_kwh"),
                F.max("consumption_kwh").alias("max_kwh"),
                F.stddev("consumption_kwh").alias("std_kwh"),
                F.count("consumption_kwh").alias("reading_count")
            )
            .orderBy(F.col("total_kwh").desc())
        )

        hh_hourly = (
            df.groupBy("household_id", "hour")
            .agg(
                F.avg("consumption_kwh").alias("avg_kwh"),
                F.sum("consumption_kwh").alias("total_kwh")
            )
            .orderBy("household_id", "hour")
        )

        return hh_summary, hh_hourly

    def export_dashboard_artifacts(
        self,
        df: DataFrame,
        output_dir: str = "dataset/processed/analytics_summary"
    ) -> Dict[str, Any]:
        """
        Executes all 11 analytical queries and exports compact CSV/JSON artifacts
        so the Streamlit dashboard loads instantly without repeated Spark scans.
        """
        root = get_project_root()
        target_dir = root / output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating and exporting analytical artifacts to: {target_dir}")

        # 1. Overview
        overview = self.compute_overview_metrics(df)
        with open(target_dir / "overview_metrics.json", "w", encoding="utf-8") as f:
            json.dump(overview, f, indent=2)

        # 2. Hourly Profile
        df_hourly = self.compute_hourly_analytics(df).toPandas()
        df_hourly.to_csv(target_dir / "hourly_consumption.csv", index=False)

        # 3. Daily Trend
        df_daily = self.compute_daily_analytics(df).toPandas()
        df_daily.to_csv(target_dir / "daily_consumption.csv", index=False)

        # 4. Weekly Trend
        df_weekly = self.compute_weekly_analytics(df).toPandas()
        df_weekly.to_csv(target_dir / "weekly_consumption.csv", index=False)

        # 5. Monthly Trend
        df_monthly = self.compute_monthly_analytics(df).toPandas()
        df_monthly.to_csv(target_dir / "monthly_consumption.csv", index=False)

        # 6. Weekday vs Weekend
        df_wday = self.compute_weekday_vs_weekend(df).toPandas()
        df_wday.to_csv(target_dir / "weekday_vs_weekend.csv", index=False)

        # 7. Peak Demand Periods
        df_peaks = self.compute_peak_demand_periods(df, top_n=50).toPandas()
        df_peaks.to_csv(target_dir / "peak_demand_periods.csv", index=False)

        # 8 & 9. Household Summary & Diurnal Profile
        hh_summary, hh_hourly = self.compute_household_analytics(df)
        # Limit household summary to top 500 households for lightweight dashboard rendering
        df_hh_summary = hh_summary.limit(500).toPandas()
        df_hh_summary.to_csv(target_dir / "household_summary.csv", index=False)

        # Limit household hourly profiles to top 50 households
        top_50_ids = [r["household_id"] for r in df_hh_summary.head(50).to_dict(orient="records")]
        df_hh_hourly = hh_hourly.filter(F.col("household_id").isin(top_50_ids)).toPandas()
        df_hh_hourly.to_csv(target_dir / "household_hourly_profile.csv", index=False)

        logger.info("All 11 analytical summaries successfully exported for dashboard consumption.")
        return overview


def run_analytics(sample_mode: bool = False, custom_parquet: Optional[str] = None) -> Dict[str, Any]:
    """Helper orchestrator function for Phase 5 analytics."""
    spark = get_spark_session(app_name="SECA_Analytics")
    engine = ConsumptionAnalyticsEngine(spark=spark)

    df_clean = engine.load_cleaned_data(sample_mode=sample_mode, custom_parquet=custom_parquet)
    out_dir = "dataset/processed/sample_analytics_summary" if sample_mode else "dataset/processed/analytics_summary"
    overview = engine.export_dashboard_artifacts(df_clean, output_dir=out_dir)

    print("\n=== [SECA] CONSUMPTION ANALYTICS SUMMARY ===")
    print(f"Total Readings: {overview['total_readings']:,}")
    print(f"Total Households: {overview['unique_households']}")
    print(f"Total Consumption: {overview['total_consumption_kwh']:,} kWh")
    print(f"Average Consumption: {overview['avg_consumption_kwh']} kWh/half-hour")
    print(f"Time Horizon: {overview['date_start']} to {overview['date_end']}")
    print(f"Peak Half-Hour: {overview['peak_reading']['kwh']} kWh by {overview['peak_reading']['household_id']} at {overview['peak_reading']['timestamp']}")
    print(f"Artifacts exported to: {out_dir}\n")

    return overview


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Electricity Consumption Analytics Module")
    parser.add_argument("--sample-mode", action="store_true", help="Run analytics on sample Parquet")
    args = parser.parse_args()

    spark = get_spark_session(app_name="SECA_Analytics")
    run_analytics(sample_mode=args.sample_mode)
    spark.stop()

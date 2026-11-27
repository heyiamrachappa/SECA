"""
Anomaly Detection Module for Smart Electricity Analytics.
Implements dual-method detection:
1. Statistical Z-Score Thresholding (diurnal baseline normalization).
2. Machine Learning Isolation Forest (Scikit-Learn).
Identifies unusually high surges and abnormal consumption drops with diagnostic metadata.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

from src.utils.config import load_config, get_project_root
from src.utils.logger import get_logger

logger = get_logger("AnomalyDetector")


class AnomalyDetector:
    """
    Hybrid statistical and machine learning anomaly detection engine.
    Detects irregular consumption patterns without falsely attributing
    all anomalies to equipment failure or theft.
    """

    def __init__(self):
        self.config = load_config()
        self.anom_cfg = self.config.get("anomaly_detection", {})
        self.z_threshold = float(self.anom_cfg.get("z_score_threshold", 3.0))
        iso_cfg = self.anom_cfg.get("isolation_forest", {})
        self.contamination = float(iso_cfg.get("contamination", 0.02))
        self.n_estimators = int(iso_cfg.get("n_estimators", 100))
        self.random_state = int(iso_cfg.get("random_state", 42))

        self.iso_model: Optional[IsolationForest] = None

    def load_telemetry_sample(
        self,
        sample_mode: bool = False,
        max_rows: int = 100000,
        custom_parquet: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Loads cleaned parquet telemetry for anomaly detection.
        Subsamples if necessary for responsive analysis while preserving temporal ordering.
        """
        root = get_project_root()
        if custom_parquet is not None:
            parquet_file = Path(custom_parquet) if os.path.isabs(str(custom_parquet)) else root / custom_parquet
        else:
            proc_dir = root / self.config["paths"]["processed_data_dir"]
            parquet_file = proc_dir / ("sample_cleaned_energy_data.parquet" if sample_mode else "cleaned_energy_data.parquet")

        if not parquet_file.exists():
            raise FileNotFoundError(f"Parquet data not found at {parquet_file}. Run preprocessing first.")

        logger.info(f"Loading data for anomaly detection: {parquet_file}")
        import pyarrow.parquet as pq
        table = pq.read_table(
            str(parquet_file),
            columns=["household_id", "timestamp", "hour", "day_of_week", "is_weekend", "consumption_kwh"]
        )
        pdf = table.to_pandas()

        if len(pdf) > max_rows:
            logger.info(f"Subsampling {max_rows:,} records from {len(pdf):,} total records for anomaly detection...")
            pdf = pdf.sample(n=max_rows, random_state=self.random_state)

        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
        pdf = pdf.sort_values(["household_id", "timestamp"]).reset_index(drop=True)
        return pdf

    def detect_statistical_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Statistical Z-Score detection relative to household-specific diurnal hourly baselines.
        Z = (x - mean(hour)) / std(hour)
        """
        logger.info(f"Executing Statistical Z-Score detection (threshold={self.z_threshold})...")
        df_stat = df.copy()

        # Compute mean and std per household and per hour
        grouped = df_stat.groupby(["household_id", "hour"])["consumption_kwh"].agg(["mean", "std"]).reset_index()
        grouped.columns = ["household_id", "hour", "hh_hour_mean", "hh_hour_std"]
        grouped["hh_hour_std"] = grouped["hh_hour_std"].fillna(0.01).replace(0.0, 0.01)

        merged = pd.merge(df_stat, grouped, on=["household_id", "hour"], how="left")
        merged["z_score"] = (merged["consumption_kwh"] - merged["hh_hour_mean"]) / merged["hh_hour_std"]

        # Flag high and low statistical anomalies
        merged["stat_anomaly"] = 0
        merged["stat_anomaly_type"] = "NORMAL"

        high_mask = merged["z_score"] > self.z_threshold
        low_mask = (merged["z_score"] < -self.z_threshold) | ((merged["consumption_kwh"] == 0.0) & (merged["hh_hour_mean"] > 0.3))

        merged.loc[high_mask, "stat_anomaly"] = 1
        merged.loc[high_mask, "stat_anomaly_type"] = "HIGH_CONSUMPTION"

        merged.loc[low_mask, "stat_anomaly"] = 1
        merged.loc[low_mask, "stat_anomaly_type"] = "LOW_CONSUMPTION"

        logger.info(
            f"Statistical anomalies flagged: {merged['stat_anomaly'].sum():,} "
            f"(High: {high_mask.sum():,}, Low: {low_mask.sum():,})"
        )
        return merged

    def detect_isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Unsupervised Machine Learning Anomaly Detection using Isolation Forest.
        Features: consumption_kwh, hour, day_of_week, is_weekend, z_score
        """
        logger.info(f"Training Isolation Forest (contamination={self.contamination})...")
        features = ["consumption_kwh", "hour", "day_of_week", "is_weekend", "z_score"]

        X = df[features].fillna(0.0)
        self.iso_model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.iso_model.fit(X)

        df_iso = df.copy()
        # -1 indicates anomaly in Scikit-learn IsolationForest, 1 is normal
        iso_preds = self.iso_model.predict(X)
        iso_scores = self.iso_model.decision_function(X)

        df_iso["iso_anomaly"] = (iso_preds == -1).astype(int)
        df_iso["anomaly_score"] = np.round(-iso_scores, 4)  # Higher score = more anomalous

        # Composite anomaly indicator (flagged by either or both methods)
        df_iso["is_anomaly"] = ((df_iso["stat_anomaly"] == 1) | (df_iso["iso_anomaly"] == 1)).astype(int)

        # Categorize anomaly type
        conditions = [
            (df_iso["is_anomaly"] == 1) & (df_iso["consumption_kwh"] > df_iso["hh_hour_mean"]),
            (df_iso["is_anomaly"] == 1) & (df_iso["consumption_kwh"] <= df_iso["hh_hour_mean"])
        ]
        choices = ["HIGH_CONSUMPTION", "LOW_CONSUMPTION"]
        df_iso["anomaly_category"] = np.select(conditions, choices, default="NORMAL")

        logger.info(
            f"Isolation Forest completed: {df_iso['iso_anomaly'].sum():,} flagged. "
            f"Total combined anomalies: {df_iso['is_anomaly'].sum():,}"
        )
        return df_iso

    def generate_diagnostic_summary(self, df_anom: pd.DataFrame) -> Dict[str, Any]:
        """Calculates breakdown metrics and diagnostics across detected anomalies."""
        total_records = len(df_anom)
        total_anomalies = int(df_anom["is_anomaly"].sum())
        high_anomalies = int((df_anom["anomaly_category"] == "HIGH_CONSUMPTION").sum())
        low_anomalies = int((df_anom["anomaly_category"] == "LOW_CONSUMPTION").sum())

        stat_count = int(df_anom["stat_anomaly"].sum())
        iso_count = int(df_anom["iso_anomaly"].sum())
        both_count = int(((df_anom["stat_anomaly"] == 1) & (df_anom["iso_anomaly"] == 1)).sum())

        summary = {
            "total_records_analyzed": total_records,
            "total_anomalies": total_anomalies,
            "anomaly_rate_pct": round((total_anomalies / total_records * 100.0) if total_records > 0 else 0, 3),
            "high_consumption_anomalies": high_anomalies,
            "low_consumption_anomalies": low_anomalies,
            "statistical_zscore_count": stat_count,
            "isolation_forest_count": iso_count,
            "consensus_anomalies_count": both_count,
            "detection_thresholds": {
                "z_score_threshold": self.z_threshold,
                "isolation_forest_contamination": self.contamination
            },
            "interpretation_disclaimer": (
                "Anomalies represent statistical deviations from standard household diurnal load profiles. "
                "High anomalies indicate potential appliance surges, HVAC operation, or technical sensor faults. "
                "Low anomalies suggest possible unoccupied periods, local outages, or communication packet drops. "
                "They should not be automatically attributed to electricity theft."
            )
        }
        return summary

    def save_artifacts(
        self,
        df_anom: pd.DataFrame,
        summary: Dict[str, Any],
        output_dir: str = "dataset/processed/analytics_summary",
        save_model: bool = True
    ) -> None:
        """Saves model binary, summary JSON, and top anomalous telemetry records."""
        root = get_project_root()
        models_dir = root / self.config["paths"]["models_dir"]

        target_dir = root / output_dir if not os.path.isabs(str(output_dir)) else Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save Isolation Forest Model (only when requested so custom runs don't overwrite defaults)
        if save_model and self.iso_model:
            models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.iso_model, models_dir / "isolation_forest.joblib")

        # 2. Save Summary JSON
        with open(target_dir / "anomaly_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # 3. Save Anomalous Records for Dashboard inspection (up to 5,000 records)
        anom_records = df_anom[df_anom["is_anomaly"] == 1].sort_values("anomaly_score", ascending=False)
        export_cols = [
            "household_id",
            "timestamp",
            "hour",
            "day_of_week",
            "consumption_kwh",
            "hh_hour_mean",
            "z_score",
            "anomaly_score",
            "anomaly_category",
            "stat_anomaly",
            "iso_anomaly"
        ]
        anom_export = anom_records[export_cols].head(5000)
        anom_export.to_csv(target_dir / "detected_anomalies.csv", index=False)

        # 4. Save Hourly Anomaly Distribution
        hourly_dist = (
            df_anom.groupby(["hour", "anomaly_category"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        hourly_dist.to_csv(target_dir / "anomaly_hourly_distribution.csv", index=False)

        logger.info(f"Anomaly artifacts exported to {target_dir} and models/.")


def run_anomaly_detection(
    sample_mode: bool = False,
    max_rows: int = 100000,
    custom_parquet: Optional[Path] = None,
    output_dir: Optional[str] = None,
    save_model: bool = True
) -> Dict[str, Any]:
    """Helper runner function to execute dual anomaly detection and export artifacts."""
    detector = AnomalyDetector()
    df = detector.load_telemetry_sample(
        sample_mode=sample_mode,
        max_rows=max_rows,
        custom_parquet=custom_parquet
    )
    df_stat = detector.detect_statistical_zscore(df)
    df_anom = detector.detect_isolation_forest(df_stat)
    summary = detector.generate_diagnostic_summary(df_anom)

    summary_dir = output_dir or ("dataset/processed/sample_analytics_summary" if sample_mode else "dataset/processed/analytics_summary")
    detector.save_artifacts(df_anom, summary, output_dir=summary_dir, save_model=save_model)

    print("\n=== [SECA] ANOMALY DETECTION SUMMARY ===")
    print(f"Total Analyzed Records: {summary['total_records_analyzed']:,}")
    print(f"Total Detected Anomalies: {summary['total_anomalies']:,} ({summary['anomaly_rate_pct']}%)")
    print(f"  - High Consumption Anomalies (Surges): {summary['high_consumption_anomalies']:,}")
    print(f"  - Low Consumption Anomalies (Drops):   {summary['low_consumption_anomalies']:,}")
    print(f"  - Statistical Z-Score Flags:           {summary['statistical_zscore_count']:,}")
    print(f"  - Isolation Forest Flags:              {summary['isolation_forest_count']:,}")
    print(f"  - Consensus (Dual Detected):           {summary['consensus_anomalies_count']:,}")
    print(f"Artifacts exported to: {summary_dir} and models/\n")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Anomaly Detection Module")
    parser.add_argument("--sample-mode", action="store_true", help="Run on sample dataset")
    parser.add_argument("--rows", type=int, default=100000, help="Max rows to analyze (default: 100,000)")
    args = parser.parse_args()

    run_anomaly_detection(sample_mode=args.sample_mode, max_rows=args.rows)

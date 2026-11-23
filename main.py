"""
Main execution entry point for Smart Electricity Analytics.
Provides a unified command-line interface for running individual pipeline phases
or orchestrating the complete end-to-end Big Data workflow.
"""

import argparse
import sys
import subprocess
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.config import load_config, get_project_root
from src.ingestion.inspect_dataset import generate_inspection_report
from src.ingestion.pyspark_ingestion import run_ingestion
from src.preprocessing.preprocessor import run_preprocessing
from src.analytics.consumption_analytics import run_analytics
from src.forecasting.demand_forecaster import run_forecasting
from src.anomaly_detection.detector import run_anomaly_detection

logger = get_logger("SECA_Orchestrator")


def run_dashboard_server():
    """Launches the Streamlit interactive dashboard process."""
    root = get_project_root()
    dashboard_script = root / "dashboard/app.py"
    logger.info("Starting Streamlit Dashboard server...")
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_script), "--server.headless", "true"]
    print(f"\n[SECA] Launching Dashboard on http://localhost:8501 ...")
    print("[SECA] Press Ctrl+C in terminal to stop dashboard server.\n")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Dashboard server terminated by user.")


def run_full_pipeline(sample_mode: bool = False):
    """Executes all pipeline phases in strict chronological and dependency order."""
    logger.info(f"=== Starting Full SECA Pipeline (sample_mode={sample_mode}) ===")

    print("\n" + "="*70)
    print("⚡ SECA END-TO-END BIG DATA ANALYTICS & FORECASTING PIPELINE")
    print(f"   Execution Mode: {'SAMPLE MODE (Fast CI)' if sample_mode else 'FULL PRODUCTION DATASET'}")
    print("="*70 + "\n")

    # Step 1: Preprocessing & Cleaning
    logger.info("[Step 1/4] Running Distributed PySpark Preprocessing...")
    spark, df_clean, parquet_path = run_preprocessing(sample_mode=sample_mode)

    # Step 2: Aggregations & Analytics
    logger.info("[Step 2/4] Computing 11 Spark SQL Consumption Analytics...")
    overview = run_analytics(sample_mode=sample_mode)

    # Stop spark session after aggregations to release executor memory for ML
    spark.stop()

    # Step 3: Machine Learning Demand Forecasting
    logger.info("[Step 3/4] Training Linear Regression & Random Forest Forecasters...")
    metrics = run_forecasting(sample_mode=sample_mode)

    # Step 4: Anomaly Detection Engine
    logger.info("[Step 4/4] Executing Statistical Z-Score & Isolation Forest Anomaly Detectors...")
    anomaly_summary = run_anomaly_detection(sample_mode=sample_mode)

    print("\n" + "="*70)
    print("✅ SECA PIPELINE EXECUTED SUCCESSFULLY")
    print(f"   Total Processed Readings: {overview['total_readings']:,}")
    print(f"   Total Households:         {overview['unique_households']}")
    print(f"   Baseline Model (LR R²):   {metrics['linear_regression']['r2_score']:.4f}")
    print(f"   Comparison Model (RF R²): {metrics['random_forest']['r2_score']:.4f}")
    print(f"   Anomalies Flagged:        {anomaly_summary['total_anomalies']:,} ({anomaly_summary['anomaly_rate_pct']}%)")
    print("   To launch dashboard:      python main.py --phase dashboard")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Smart Electricity Consumption Analytics and Demand Forecasting CLI"
    )
    parser.add_argument(
        "--phase",
        choices=["setup", "inspect", "ingest", "preprocess", "analytics", "forecast", "anomaly", "dashboard", "all"],
        default="setup",
        help="Specify the pipeline phase to execute."
    )
    parser.add_argument(
        "--sample-mode",
        action="store_true",
        help="Run in sample mode using lightweight data for fast verification."
    )

    args = parser.parse_args()
    config = load_config()

    logger.info(f"Initialized SECA CLI with phase: '{args.phase}' (sample_mode={args.sample_mode})")

    if args.phase == "setup":
        print("\n[SECA] Project environment verified successfully.")
        print(f"[SECA] Target Dataset: {config['dataset']['name']}")
        print(f"[SECA] Configured Spark Master: {config['spark']['master']}\n")

    elif args.phase == "inspect":
        target = config["paths"]["sample_data_dir"] if args.sample_mode else config["paths"]["raw_data_dir"]
        report = generate_inspection_report(target_dir=target)
        print("\n=== [SECA] DATASET INSPECTION SUMMARY ===")
        print(f"Directory: {report['data_directory']}")
        print(f"Total Block Files: {report['file_count']}")
        print(f"Total Telemetry Records: {report['total_records']:,}")
        print(f"Total Disk Size: {report['total_size_mb']} MB\n")

    elif args.phase == "ingest":
        spark, df = run_ingestion(sample_mode=args.sample_mode)
        spark.stop()

    elif args.phase == "preprocess":
        spark, df_clean, parquet_path = run_preprocessing(sample_mode=args.sample_mode)
        spark.stop()

    elif args.phase == "analytics":
        spark = run_analytics(sample_mode=args.sample_mode)

    elif args.phase == "forecast":
        run_forecasting(sample_mode=args.sample_mode)

    elif args.phase == "anomaly":
        run_anomaly_detection(sample_mode=args.sample_mode)

    elif args.phase == "dashboard":
        run_dashboard_server()

    elif args.phase == "all":
        run_full_pipeline(sample_mode=args.sample_mode)


if __name__ == "__main__":
    main()

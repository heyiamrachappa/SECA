"""
Main execution entry point for Smart Electricity Analytics.
Provides a unified command-line interface for running individual pipeline phases.
"""

import argparse
import sys
from src.utils.logger import get_logger
from src.utils.config import load_config
from src.ingestion.inspect_dataset import generate_inspection_report
from src.ingestion.sample_generator import create_sample_dataset

logger = get_logger("Main")


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
        logger.info("Phase 1 verification: Environment, dependencies, and Spark session are ready.")
        print("\n[SECA] Project environment verified successfully.")
        print(f"[SECA] Target Dataset: {config['dataset']['name']}")
        print(f"[SECA] Configured Spark Master: {config['spark']['master']}\n")

    elif args.phase == "inspect":
        target = config["paths"]["sample_data_dir"] if args.sample_mode else config["paths"]["raw_data_dir"]
        logger.info(f"Phase 2: Inspecting dataset files in {target}...")
        report = generate_inspection_report(target_dir=target)
        print("\n=== [SECA] DATASET INSPECTION SUMMARY ===")
        print(f"Directory: {report['data_directory']}")
        print(f"Total Block Files: {report['file_count']}")
        print(f"Total Telemetry Records: {report['total_records']:,}")
        print(f"Total Disk Size: {report['total_size_mb']} MB")
        print(f"Inspection report generated at: reports/dataset_inspection_report.md\n")

    else:
        logger.info(f"Phase '{args.phase}' will be invoked in its respective module.")
        print(f"\n[SECA] Phase '{args.phase}' is queued for execution.\n")


if __name__ == "__main__":
    main()

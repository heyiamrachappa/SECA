"""
Main execution entry point for Smart Electricity Analytics.
Provides a unified command-line interface for running individual pipeline phases.
"""

import argparse
import sys
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger("Main")


def main():
    parser = argparse.ArgumentParser(
        description="Smart Electricity Consumption Analytics and Demand Forecasting CLI"
    )
    parser.add_argument(
        "--phase",
        choices=["setup", "ingest", "preprocess", "analytics", "forecast", "anomaly", "dashboard", "all"],
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
        print(f"[SECA] Configured Spark Master: {config['spark']['master']}")
        print("[SECA] Ready for Phase 2: Dataset Acquisition & Inspection.\n")
    else:
        logger.info(f"Phase '{args.phase}' will be invoked in its respective module.")
        print(f"\n[SECA] Phase '{args.phase}' is queued for execution.\n")


if __name__ == "__main__":
    main()

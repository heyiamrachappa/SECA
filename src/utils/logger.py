"""
Logging utility for Smart Electricity Analytics.
Provides uniform console and file logging across all modules.
"""

import logging
import os
import sys
from pathlib import Path
from src.utils.config import get_project_root


def get_logger(name: str = "SmartElectricityAnalytics", log_file: str = "logs/app.log") -> logging.Logger:
    """
    Creates or retrieves a standardized logger.

    Parameters:
        name (str): Name of the logger (typically module or class name).
        log_file (str): Relative path to the output log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

        # File Handler
        root = get_project_root()
        full_log_path = root / log_file if not os.path.isabs(log_file) else Path(log_file)
        full_log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(full_log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    return logger

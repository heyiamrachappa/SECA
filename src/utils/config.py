"""
Configuration loader utility for Smart Electricity Analytics.
Reads YAML configuration file and provides typed dictionary access.
"""

import os
from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Loads and parses the project YAML configuration file.

    Parameters:
        config_path (str): Relative or absolute path to config.yaml.

    Returns:
        dict: Parsed configuration dictionary.
    """
    root = get_project_root()
    full_path = root / config_path if not os.path.isabs(config_path) else Path(config_path)

    if not full_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

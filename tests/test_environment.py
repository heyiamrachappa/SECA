"""
Unit tests for Phase 1: Environment verification and setup.
Tests Python version, core imports, configuration loading, and Spark session initialization.
"""

import sys
import pytest
from src.utils.config import load_config, get_project_root
from src.utils.spark_session import get_spark_session


def test_python_version():
    """Verify that Python is at least 3.10."""
    assert sys.version_info >= (3, 10), "Python version must be 3.10 or higher"


def test_core_dependencies_importable():
    """Verify that all core libraries can be imported without error."""
    import pyspark
    import pandas
    import numpy
    import sklearn
    import streamlit
    import plotly
    import pyarrow
    import yaml

    assert pyspark.__version__ is not None
    assert pandas.__version__ is not None
    assert numpy.__version__ is not None
    assert sklearn.__version__ is not None
    assert streamlit.__version__ is not None
    assert plotly.__version__ is not None
    assert pyarrow.__version__ is not None
    assert yaml.__version__ is not None


def test_config_loader():
    """Verify that config.yaml loads properly and contains required sections."""
    config = load_config()
    assert "app" in config
    assert "spark" in config
    assert "paths" in config
    assert "dataset" in config
    assert config["app"]["name"] == "Smart Electricity Consumption Analytics"


def test_spark_session_initialization():
    """Verify that SparkSession creates and executes a basic action in local mode."""
    spark = get_spark_session(app_name="PyTestSparkInit", master="local[1]", driver_memory="1g")
    assert spark is not None
    df = spark.createDataFrame([(1, "test")], ["id", "val"])
    assert df.count() == 1
    spark.stop()

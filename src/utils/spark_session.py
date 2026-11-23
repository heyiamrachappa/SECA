"""
Spark session factory and manager for local/distributed processing.
Ensures correct JVM path resolution, memory limits, and optimal partition tuning.
"""

import os
from pathlib import Path
from typing import Optional
from pyspark.sql import SparkSession
from src.utils.config import load_config, get_project_root
from src.utils.logger import get_logger

logger = get_logger("SparkSessionManager")


def ensure_java_environment() -> None:
    """
    Ensures JAVA_HOME is configured.
    If not set in the environment, checks standard user directory ~/.jdk/jdk-17.
    """
    if "JAVA_HOME" not in os.environ or not os.path.exists(os.environ["JAVA_HOME"]):
        potential_jdk = Path.home() / ".jdk" / "jdk-17"
        if potential_jdk.exists():
            os.environ["JAVA_HOME"] = str(potential_jdk)
            java_bin = str(potential_jdk / "bin")
            if java_bin not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{java_bin}:{os.environ.get('PATH', '')}"
            logger.info(f"Auto-configured JAVA_HOME to: {potential_jdk}")
        else:
            logger.warning(
                "JAVA_HOME is not set and ~/.jdk/jdk-17 was not found. "
                "PySpark will attempt to locate java from system PATH."
            )


def get_spark_session(
    app_name: Optional[str] = None,
    master: Optional[str] = None,
    driver_memory: Optional[str] = None,
    log_level: Optional[str] = None
) -> SparkSession:
    """
    Builds and returns an optimized PySpark SparkSession.

    Parameters:
        app_name (str, optional): Custom application name.
        master (str, optional): Spark master URL (default 'local[*]').
        driver_memory (str, optional): JVM driver memory allocation (e.g. '4g').
        log_level (str, optional): Spark log level ('INFO', 'WARN', 'ERROR').

    Returns:
        SparkSession: The initialized SparkSession.
    """
    ensure_java_environment()
    config = load_config()
    spark_cfg = config.get("spark", {})

    app = app_name or spark_cfg.get("app_name", "SmartElectricityAnalytics")
    master_url = master or spark_cfg.get("master", "local[*]")
    mem = driver_memory or spark_cfg.get("driver_memory", "4g")
    lvl = log_level or spark_cfg.get("log_level", "WARN")
    partitions = spark_cfg.get("sql_shuffle_partitions", "16")

    logger.info(f"Initializing SparkSession: '{app}' (master={master_url}, driver_memory={mem})")

    spark = (
        SparkSession.builder
        .appName(app)
        .master(master_url)
        .config("spark.driver.memory", mem)
        .config("spark.executor.memory", mem)
        .config("spark.sql.shuffle.partitions", partitions)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel(lvl)
    logger.info(f"SparkSession successfully initialized with Spark version {spark.version}")
    return spark

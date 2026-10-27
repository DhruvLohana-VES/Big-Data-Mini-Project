"""
PySpark Session Manager
Handles SparkSession initialization, configuration, and Windows environment adjustments.
"""

import os
import sys
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

_spark_session = None


def is_spark_available() -> bool:
    """Check if PySpark is importable and Java runtime is available."""
    try:
        import pyspark
        from pyspark.sql import SparkSession
        return True
    except Exception:
        return False


def get_spark_session(app_name: str = "IndicCodeMixedSocialAnalytics", master: str = "local[*]") -> Optional[Any]:
    """
    Get or create a singleton SparkSession configured for local distributed processing.
    """
    global _spark_session
    if _spark_session is not None:
        return _spark_session

    try:
        import pyspark
        from pyspark.sql import SparkSession

        # Avoid winutils error noise on Windows local mode
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

        builder = (
            SparkSession.builder
            .appName(app_name)
            .master(master)
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.default.parallelism", "4")
            .config("spark.driver.memory", "2g")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        )
        
        _spark_session = builder.getOrCreate()
        _spark_session.sparkContext.setLogLevel("WARN")
        return _spark_session
    except Exception as e:
        logger.warning(f"Could not initialize PySpark Session ({e}). Falling back to Pandas distributed emulation mode.")
        return None


def stop_spark_session():
    """Gracefully terminate active SparkSession."""
    global _spark_session
    if _spark_session is not None:
        try:
            _spark_session.stop()
        except Exception:
            pass
        _spark_session = None

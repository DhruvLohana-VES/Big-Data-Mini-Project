"""
Big Data and PySpark Analytics Package
"""
from .spark_session import get_spark_session, is_spark_available
from .spark_preprocessing import SparkPreprocessor, preprocess_all_datasets
from .spark_analytics import SparkSocialAnalytics
from .streaming_demo import SocialMediaStreamSimulator

__all__ = [
    "get_spark_session",
    "is_spark_available",
    "SparkPreprocessor",
    "preprocess_all_datasets",
    "SparkSocialAnalytics",
    "SocialMediaStreamSimulator"
]

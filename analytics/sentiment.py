"""
Sentiment Analytics for Code-Mixed Social Posts.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


class SentimentAnalytics:
    """Computes sentiment breakdown, polarity distribution, and language-wise comparisons."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_sentiment_distribution(self) -> Dict[str, Any]:
        """Distribution of positive, negative, neutral posts."""
        total = len(self.df)
        counts = self.df["sentiment"].value_counts().to_dict()
        percentages = {k: round((v / total) * 100, 2) for k, v in counts.items()}
        return {"counts": counts, "percentages": percentages, "total": total}

    def get_sentiment_by_language(self) -> pd.DataFrame:
        """Crosstab of sentiment across Hinglish, Marathi, English, Hindi."""
        ct = pd.crosstab(self.df["language"], self.df["sentiment"], normalize="index") * 100
        return ct.round(2)

    def get_sarcasm_sentiment_impact(self) -> pd.DataFrame:
        """Examines the relationship between sarcasm and apparent vs true sentiment."""
        ct = pd.crosstab(self.df["sarcasm"], self.df["sentiment"])
        return ct

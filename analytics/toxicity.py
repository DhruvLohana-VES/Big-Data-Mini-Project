"""
Toxicity and Offensive Content Analytics.
"""

from typing import Dict, Any
import pandas as pd


class ToxicityAnalytics:
    """Computes toxicity rates, cross-tabulations, and language-specific distributions."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_toxicity_distribution(self) -> Dict[str, Any]:
        """Distribution of toxic vs non-toxic posts."""
        total = len(self.df)
        counts = self.df["toxicity"].value_counts().to_dict()
        toxic_count = counts.get("toxic", 0)
        toxic_pct = round((toxic_count / total) * 100, 2) if total > 0 else 0.0
        return {
            "counts": counts,
            "toxic_count": toxic_count,
            "toxic_percentage": toxic_pct,
            "total": total
        }

    def get_toxicity_by_language(self) -> pd.DataFrame:
        """Crosstab of toxicity across languages."""
        ct = pd.crosstab(self.df["language"], self.df["toxicity"], normalize="index") * 100
        return ct.round(2)

    def get_sentiment_vs_toxicity(self) -> pd.DataFrame:
        """Crosstab of sentiment against toxicity."""
        return pd.crosstab(self.df["sentiment"], self.df["toxicity"])

"""
Code-Mixing and Multilingualism Metrics Module.
Calculates Code-Mixing Index (CMI), Script-Switching Ratios, and Lexical Complexity.
"""

import re
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class CodeMixingAnalytics:
    """
    Computes computational sociolinguistic metrics for code-mixed text.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @staticmethod
    def calculate_cmi(text: str) -> float:
        """
        Calculate Code-Mixing Index (CMI) based on Gambäck & Das (2014):
        CMI = 100 * [1 - (max(w_i) / n - u)] where w_i is tokens in language i.
        """
        if not isinstance(text, str) or not text.strip():
            return 0.0

        words = re.findall(r"\b[a-zA-Z\u0900-\u097F]+\b", text)
        n = len(words)
        if n <= 1:
            return 0.0

        dev_words = sum(1 for w in words if re.search(r"[\u0900-\u097F]", w))
        lat_words = sum(1 for w in words if re.search(r"[a-zA-Z]", w))

        max_lang_count = max(dev_words, lat_words)
        cmi = 100.0 * (1.0 - (max_lang_count / n))
        return round(cmi, 2)

    def compute_corpus_cmi_stats(self) -> Dict[str, Any]:
        """Compute aggregate CMI statistics across the dataset."""
        cmis = self.df["text"].apply(self.calculate_cmi)
        return {
            "mean_cmi": round(float(cmis.mean()), 2),
            "median_cmi": round(float(cmis.median()), 2),
            "max_cmi": round(float(cmis.max()), 2),
            "code_mixed_post_pct": round(float((cmis > 0).mean() * 100), 2)
        }

    def get_script_mixing_breakdown(self) -> pd.DataFrame:
        """Distribution of scripts across languages."""
        ct = pd.crosstab(self.df["language"], self.df["script"], margins=True)
        return ct

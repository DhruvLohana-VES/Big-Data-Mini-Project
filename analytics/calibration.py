"""
Confidence Calibration & Uncertainty Quantification Module.
Implements Expected Calibration Error (ECE), Reliability Diagrams, and Confidence-Based Filtering.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


class CalibrationAnalytics:
    """
    Evaluates model calibration and Expected Calibration Error (ECE) for LLM annotations.
    """

    @staticmethod
    def compute_ece_and_reliability(
        y_true: pd.Series,
        y_pred: pd.Series,
        confidences: pd.Series,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Compute Expected Calibration Error (ECE) and reliability bin data.
        """
        valid_mask = ~(y_true.isna() | y_pred.isna() | confidences.isna())
        yt = y_true[valid_mask].astype(str).str.lower().values
        yp = y_pred[valid_mask].astype(str).str.lower().values
        conf = confidences[valid_mask].astype(float).values

        if len(yt) == 0:
            return {"ece": 0.0, "bins": []}

        correctness = (yt == yp).astype(float)
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        
        ece = 0.0
        bin_data = []
        n_total = len(yt)

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Select samples in this confidence bin
            if i == n_bins - 1:
                in_bin = (conf >= bin_lower) & (conf <= bin_upper)
            else:
                in_bin = (conf >= bin_lower) & (conf < bin_upper)
            
            bin_size = np.sum(in_bin)
            if bin_size > 0:
                bin_acc = float(np.mean(correctness[in_bin]))
                bin_conf = float(np.mean(conf[in_bin]))
                abs_diff = abs(bin_acc - bin_conf)
                ece += (bin_size / n_total) * abs_diff

                bin_data.append({
                    "bin_index": i,
                    "bin_lower": round(bin_lower, 2),
                    "bin_upper": round(bin_upper, 2),
                    "bin_center": round((bin_lower + bin_upper) / 2, 2),
                    "sample_count": int(bin_size),
                    "accuracy": round(bin_acc, 4),
                    "confidence": round(bin_conf, 4),
                    "calibration_gap": round(abs_diff, 4)
                })
            else:
                bin_data.append({
                    "bin_index": i,
                    "bin_lower": round(bin_lower, 2),
                    "bin_upper": round(bin_upper, 2),
                    "bin_center": round((bin_lower + bin_upper) / 2, 2),
                    "sample_count": 0,
                    "accuracy": 0.0,
                    "confidence": round((bin_lower + bin_upper) / 2, 2),
                    "calibration_gap": 0.0
                })

        return {
            "ece": round(float(ece), 4),
            "ece_percentage": round(float(ece * 100), 2),
            "bins": bin_data,
            "total_samples": n_total
        }

    @staticmethod
    def evaluate_confidence_filtering(
        df: pd.DataFrame,
        gold_col: str = "sentiment",
        pred_col: str = "llm_sentiment",
        conf_col: str = "llm_confidence",
        thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    ) -> pd.DataFrame:
        """
        Evaluate downstream accuracy and retention at varying confidence filtering thresholds.
        """
        from sklearn.metrics import accuracy_score, f1_score

        rows = []
        total_n = len(df)

        for thresh in thresholds:
            filtered = df[df[conf_col] >= thresh]
            retention_pct = round((len(filtered) / total_n) * 100, 1) if total_n > 0 else 0.0

            if len(filtered) > 0:
                yt = filtered[gold_col].astype(str).str.lower()
                yp = filtered[pred_col].astype(str).str.lower()
                acc = float(accuracy_score(yt, yp))
                f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
            else:
                acc = 0.0
                f1 = 0.0

            rows.append({
                "Confidence Threshold": thresh,
                "Retained Samples": len(filtered),
                "Data Retention (%)": f"{retention_pct}%",
                "Accuracy": round(acc, 4),
                "Macro-F1": round(f1, 4)
            })

        return pd.DataFrame(rows)

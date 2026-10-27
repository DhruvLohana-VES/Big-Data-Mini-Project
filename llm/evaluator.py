"""
LLM vs Human Gold Evaluation and Subgroup Diagnostics Module.
Computes multi-metric scores (F1, Precision, Recall, Accuracy, Kappa)
and conducts fine-grained subgroup evaluations (Sarcasm, Script mixing, Hinglish vs Marathi).
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix
)

logger = logging.getLogger(__name__)


class LLMEvaluator:
    """
    Evaluator comparing LLM prompting paradigms against Human Gold benchmarks.
    """

    def __init__(self, gold_df: pd.DataFrame):
        self.gold_df = gold_df.copy()

    def evaluate_predictions(
        self,
        pred_df: pd.DataFrame,
        task: str = "sentiment",
        gold_col: Optional[str] = None,
        pred_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute standard NLP evaluation metrics for a task.
        """
        g_col = gold_col or task
        p_col = pred_col or f"llm_{task}"

        merged = pd.merge(self.gold_df, pred_df[["id", p_col]], on="id", how="inner")
        if merged.empty:
            return {}

        y_true = merged[g_col].astype(str).str.lower()
        y_pred = merged[p_col].astype(str).str.lower()

        labels = sorted(list(set(y_true.unique()).union(set(y_pred.unique()))))

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        rec = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels))
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

        return {
            "task": task,
            "sample_size": len(merged),
            "accuracy": round(acc, 4),
            "precision_macro": round(prec, 4),
            "recall_macro": round(rec, 4),
            "macro_f1": round(f1, 4),
            "cohen_kappa": round(kappa, 4),
            "labels": labels,
            "confusion_matrix": cm
        }

    def evaluate_subgroups(
        self,
        pred_df: pd.DataFrame,
        task: str = "sentiment",
        gold_col: Optional[str] = None,
        pred_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Evaluate LLM performance partitioned across difficult linguistic subgroups:
        1. Sarcastic vs Non-Sarcastic
        2. Script-Mixed vs Pure Script
        3. Hinglish vs Marathi
        4. Romanized vs Devanagari Native Script
        """
        g_col = gold_col or task
        p_col = pred_col or f"llm_{task}"

        merged = pd.merge(self.gold_df, pred_df[["id", p_col]], on="id", how="inner")
        if merged.empty:
            return pd.DataFrame()

        subgroups = {
            "Overall": merged,
            "Sarcastic Posts": merged[merged["sarcasm"] == True],
            "Non-Sarcastic Posts": merged[merged["sarcasm"] == False],
            "Script-Mixed Posts": merged[merged["script"] == "mixed"],
            "Pure-Script Posts": merged[merged["script"] != "mixed"],
            "Hinglish Language": merged[merged["language"] == "hinglish"],
            "Marathi Language": merged[merged["language"] == "marathi"],
            "Romanized Script": merged[merged["script"] == "roman"],
            "Devanagari Script": merged[merged["script"] == "devanagari"]
        }

        subgroup_rows = []
        for sg_name, sg_df in subgroups.items():
            if len(sg_df) == 0:
                continue
            
            y_t = sg_df[g_col].astype(str).str.lower()
            y_p = sg_df[p_col].astype(str).str.lower()
            
            acc = float(accuracy_score(y_t, y_p))
            f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
            prec = float(precision_score(y_t, y_p, average="macro", zero_division=0))
            rec = float(recall_score(y_t, y_p, average="macro", zero_division=0))

            subgroup_rows.append({
                "Subgroup": sg_name,
                "Sample Count": len(sg_df),
                "Task": task.title(),
                "Accuracy": round(acc, 4),
                "Macro-F1": round(f1, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4)
            })

        return pd.DataFrame(subgroup_rows)


def evaluate_llm_vs_gold(gold_path: str = "data/gold/gold_master.csv", llm_dir: str = "data/llm_annotations") -> Dict[str, Any]:
    """
    Run full comparison across Zero-Shot, Few-Shot, and Reasoning-Aware LLM annotations against Gold.
    Saves outputs to results/tables/.
    """
    if not os.path.exists(gold_path):
        raise FileNotFoundError(f"Gold master dataset not found at {gold_path}")

    gold_df = pd.read_csv(gold_path)
    evaluator = LLMEvaluator(gold_df)

    strategies = ["zero_shot", "few_shot", "reasoning_aware"]
    all_metrics = []
    all_subgroups = []

    for strat in strategies:
        strat_file = os.path.join(llm_dir, f"{strat}.csv")
        if not os.path.exists(strat_file):
            logger.warning(f"Strategy file {strat_file} not found. Running annotation on the fly...")
            from llm.annotator import annotate_dataset
            pred_df = annotate_dataset(strategy=strat, input_path=gold_path)
        else:
            pred_df = pd.read_csv(strat_file)

        # Sentiment Evaluation
        sent_metrics = evaluator.evaluate_predictions(pred_df, task="sentiment")
        sent_metrics["strategy"] = strat
        all_metrics.append({
            "Strategy": strat.replace("_", "-").title(),
            "Task": "Sentiment",
            "Accuracy": sent_metrics.get("accuracy", 0.0),
            "Macro-F1": sent_metrics.get("macro_f1", 0.0),
            "Precision": sent_metrics.get("precision_macro", 0.0),
            "Recall": sent_metrics.get("recall_macro", 0.0),
            "Cohen Kappa (k)": sent_metrics.get("cohen_kappa", 0.0)
        })

        # Toxicity Evaluation
        tox_metrics = evaluator.evaluate_predictions(pred_df, task="toxicity")
        tox_metrics["strategy"] = strat
        all_metrics.append({
            "Strategy": strat.replace("_", "-").title(),
            "Task": "Toxicity",
            "Accuracy": tox_metrics.get("accuracy", 0.0),
            "Macro-F1": tox_metrics.get("macro_f1", 0.0),
            "Precision": tox_metrics.get("precision_macro", 0.0),
            "Recall": tox_metrics.get("recall_macro", 0.0),
            "Cohen Kappa (k)": tox_metrics.get("cohen_kappa", 0.0)
        })

        # Subgroup evaluations
        sg_sent = evaluator.evaluate_subgroups(pred_df, task="sentiment")
        sg_sent["Strategy"] = strat.replace("_", "-").title()
        all_subgroups.append(sg_sent)

    metrics_df = pd.DataFrame(all_metrics)
    subgroups_df = pd.concat(all_subgroups, ignore_index=True) if all_subgroups else pd.DataFrame()

    os.makedirs("results/tables", exist_ok=True)
    metrics_df.to_csv("results/tables/llm_vs_human_metrics.csv", index=False)
    subgroups_df.to_csv("results/tables/subgroup_performance.csv", index=False)

    return {
        "metrics_df": metrics_df,
        "subgroups_df": subgroups_df
    }


if __name__ == "__main__":
    res = evaluate_llm_vs_gold()
    print("=== LLM vs Human Gold Evaluation Metrics ===")
    print(res["metrics_df"].to_string(index=False))

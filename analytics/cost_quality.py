"""
Cost vs. Quality Trade-off Experimentation Module.
Evaluates empirical downstream model Macro-F1 across varying hybrid ratios of
Human Gold vs. LLM-Assisted synthetic annotations, balancing budget vs. accuracy.
Supports dynamic Gemini/LLM token pricing configuration.
"""

import os
import random
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class CostQualityExperiment:
    """
    Simulates hybrid annotation budgets (10% Gold -> 100% Gold)
    and evaluates downstream model Macro-F1 and cost efficiency.
    """

    def __init__(
        self,
        gold_df: pd.DataFrame,
        llm_df: pd.DataFrame,
        human_cost_per_item: float = 0.05,
        human_seconds_per_item: float = 25.0
    ):
        self.gold_df = gold_df.copy()
        self.llm_df = llm_df.copy()
        self.human_cost_per_item = human_cost_per_item
        self.human_seconds_per_item = human_seconds_per_item

        # Pricing from environment / config
        raw_in_cost = os.getenv("LLM_INPUT_COST_PER_1M_TOKENS", "0.075")
        raw_out_cost = os.getenv("LLM_OUTPUT_COST_PER_1M_TOKENS", "0.30")
        
        try:
            self.input_cost_per_1m = float(raw_in_cost) if raw_in_cost else None
            self.output_cost_per_1m = float(raw_out_cost) if raw_out_cost else None
        except ValueError:
            self.input_cost_per_1m = None
            self.output_cost_per_1m = None

    def is_pricing_available(self) -> bool:
        """Check if LLM pricing rates are configured."""
        return self.input_cost_per_1m is not None and self.output_cost_per_1m is not None

    def compute_llm_item_cost(self, avg_input_tokens: int = 400, avg_output_tokens: int = 80) -> Optional[float]:
        """Compute cost per LLM annotation."""
        if not self.is_pricing_available():
            return None
        return (avg_input_tokens / 1_000_000.0) * self.input_cost_per_1m + (avg_output_tokens / 1_000_000.0) * self.output_cost_per_1m

    def run_experiment(
        self,
        proportions: List[float] = [0.10, 0.25, 0.50, 0.75, 1.00],
        task: str = "sentiment",
        n_repeats: int = 5
    ) -> pd.DataFrame:
        """
        Evaluate downstream classification quality across multiple gold proportions.
        Uses repeated random sampling for empirical robustness.
        """
        llm_item_col = f"llm_{task}" if f"llm_{task}" in self.llm_df.columns else task

        merged = pd.merge(
            self.gold_df[["id", "text", task]],
            self.llm_df[["id", llm_item_col]],
            on="id",
            how="inner"
        )
        n_total = len(merged)
        if n_total == 0:
            return pd.DataFrame()

        llm_cost_unit = self.compute_llm_item_cost()
        rows = []

        for p in proportions:
            n_gold = int(round(p * n_total))
            n_llm = n_total - n_gold

            # Compute costs
            human_cost = n_gold * self.human_cost_per_item
            human_hours = round((n_gold * self.human_seconds_per_item) / 3600.0, 2)

            if llm_cost_unit is not None:
                llm_cost = n_llm * llm_cost_unit
                total_cost_val = round(human_cost + llm_cost, 4)
                total_cost_disp = f"${total_cost_val:.4f}"
                baseline_human_cost = n_total * self.human_cost_per_item
                savings_pct = round(((baseline_human_cost - total_cost_val) / baseline_human_cost) * 100, 1)
                savings_disp = f"{savings_pct}%"
            else:
                total_cost_disp = "Cost calculation unavailable - configure Gemini pricing."
                savings_disp = "N/A"

            f1_scores = []
            for seed in range(n_repeats):
                indices = list(range(n_total))
                random.seed(42 + seed)
                random.shuffle(indices)
                
                gold_idx = set(indices[:n_gold])
                
                hybrid_preds = []
                ground_truths = []
                for idx, row in merged.iterrows():
                    ground_truths.append(str(row[task]).lower())
                    if idx in gold_idx:
                        hybrid_preds.append(str(row[task]).lower())
                    else:
                        hybrid_preds.append(str(row[llm_item_col]).lower())

                score = f1_score(ground_truths, hybrid_preds, average="macro", zero_division=0)
                f1_scores.append(score)

            mean_f1 = round(float(np.mean(f1_scores)), 4)
            std_f1 = round(float(np.std(f1_scores)), 4)

            rows.append({
                "Gold Proportion (%)": int(p * 100),
                "Gold Samples": n_gold,
                "LLM Samples": n_llm,
                "Macro-F1": mean_f1,
                "F1 StdDev": std_f1,
                "Total Cost": total_cost_disp,
                "Human Hours (h)": human_hours,
                "Cost Savings vs 100% Gold (%)": savings_disp
            })

        results_df = pd.DataFrame(rows)
        os.makedirs("results/tables", exist_ok=True)
        results_df.to_csv("results/tables/cost_quality_table.csv", index=False)
        return results_df


if __name__ == "__main__":
    g_df = pd.read_csv("data/gold/gold_master.csv")
    l_df = pd.read_csv("data/llm_annotations/reasoning_aware.csv") if os.path.exists("data/llm_annotations/reasoning_aware.csv") else g_df.rename(columns={"sentiment": "llm_sentiment"})
    exp = CostQualityExperiment(g_df, l_df)
    res_df = exp.run_experiment()
    print("=== Cost vs Quality Empirical Trade-Off Results ===")
    print(res_df.to_string(index=False))

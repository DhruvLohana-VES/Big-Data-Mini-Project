"""
Comprehensive Downstream Model Evaluation Suite.
Orchestrates experiments comparing Classical Baselines with Indic Transformers
across Gold-Only, LLM-Only, and Mixed Training Regimes.
Persists consolidated metrics to results/model_results.csv.
"""

import os
import logging
from typing import Dict, Any, List
import pandas as pd
from models.baseline import train_and_eval_baselines
from models.train_transformer import train_transformer_model, SUPPORTED_TRANSFORMERS

logger = logging.getLogger(__name__)


def run_all_model_experiments(
    gold_path: str = "data/gold/gold_master.csv",
    llm_path: str = "data/llm_annotations/reasoning_aware.csv",
    output_path: str = "results/model_results.csv"
) -> pd.DataFrame:
    """
    Run full benchmark across classical baselines and all candidate Indic Transformers.
    """
    logger.info("Executing Classical Baseline Models (TF-IDF + LR, TF-IDF + SVM)...")
    baseline_df = train_and_eval_baselines(gold_path=gold_path, llm_path=llm_path)

    all_dfs = [baseline_df]

    # Evaluate Transformer models
    for key, model_id in SUPPORTED_TRANSFORMERS.items():
        logger.info(f"Evaluating Transformer Model: {model_id}...")
        tf_df = train_transformer_model(model_name=model_id, gold_path=gold_path, llm_path=llm_path)
        all_dfs.append(tf_df)

    final_df = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logger.info(f"Model benchmark suite complete! Saved to {output_path}")

    return final_df


if __name__ == "__main__":
    df = run_all_model_experiments()
    print("=== Consolidated Model Benchmark Results ===")
    print(df[["model", "training_regime", "task", "macro_f1", "accuracy", "precision", "recall"]].to_string(index=False))

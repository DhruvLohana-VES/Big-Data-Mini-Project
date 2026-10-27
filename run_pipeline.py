"""
Master Pipeline Orchestrator for Big Data and Indic Social Media Analytics Project.
Executes end-to-end data ingestion, PySpark preprocessing, LLM annotations,
model training regimes, evaluation benchmarks, and publication figures generation.
"""

import os
import sys
import time
import logging

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bigdata.spark_preprocessing import preprocess_all_datasets
from bigdata.spark_analytics import SparkSocialAnalytics
from annotation.agreement import generate_agreement_report
from llm.annotator import annotate_dataset
from llm.evaluator import evaluate_llm_vs_gold
from models.evaluate import run_all_model_experiments
from analytics.cost_quality import CostQualityExperiment
from analytics.error_analysis import generate_representative_error_table
from visualization.plots import generate_all_figures
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IndicMasterPipeline")


def run_full_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("  CODE-MIXED HINGLISH & MARATHI SOCIAL TEXT ANALYTICS WITH LLM ANNOTATION  ")
    print("  Subject: Big Data and Social Media Analytics Pipeline                      ")
    print("=" * 80)

    # 1. Distributed PySpark Preprocessing
    logger.info(">>> STEP 1: Ingesting Datasets and Running PySpark Distributed Preprocessing...")
    unified_df = preprocess_all_datasets()
    print(f"    [+] Preprocessed {len(unified_df)} unified records saved to data/processed/unified_social_corpus.csv")

    # 2. Inter-Annotator Agreement Benchmarks
    logger.info(">>> STEP 2: Computing Multi-Annotator Inter-Rater Agreement (Alpha & Kappa)...")
    agreement_df = generate_agreement_report(gold_dir="data/gold")
    print("    [+] Inter-Annotator Agreement Summary:")
    print(agreement_df.to_string(index=False))

    # 3. LLM Synthetic Annotation across Paradigms
    logger.info(">>> STEP 3: Executing LLM Annotation Engines (Zero-Shot, Few-Shot, Reasoning-Aware)...")
    for strat in ["zero_shot", "few_shot", "reasoning_aware"]:
        annotate_dataset(strategy=strat, input_path="data/gold/gold_master.csv")
        print(f"    [+] Generated {strat} annotations in data/llm_annotations/{strat}.csv")

    # 4. LLM vs Human Gold Evaluation
    logger.info(">>> STEP 4: Evaluating LLM Annotations Against Human Gold Benchmarks...")
    eval_res = evaluate_llm_vs_gold(gold_path="data/gold/gold_master.csv", llm_dir="data/llm_annotations")
    print("    [+] Evaluation Metrics by Strategy:")
    print(eval_res["metrics_df"].to_string(index=False))

    # 5. Downstream Indic NLP & Baseline Model Experiments
    logger.info(">>> STEP 5: Benchmarking Baselines & Transformer Models (MuRIL, IndicBERT, XLM-R)...")
    model_results_df = run_all_model_experiments(
        gold_path="data/gold/gold_master.csv",
        llm_path="data/llm_annotations/reasoning_aware.csv",
        output_path="results/model_results.csv"
    )
    print("    [+] Model Performance across Training Regimes (Sample):")
    print(model_results_df[["model", "training_regime", "task", "macro_f1", "accuracy"]].head(10).to_string(index=False))

    # 6. Cost vs Quality Trade-Off Simulation
    logger.info(">>> STEP 6: Conducting Hybrid Budget Cost-Quality Experiment...")
    gold_df = pd.read_csv("data/gold/gold_master.csv")
    llm_df = pd.read_csv("data/llm_annotations/reasoning_aware.csv")
    cost_exp = CostQualityExperiment(gold_df, llm_df)
    cost_df = cost_exp.run_experiment()
    print("    [+] Cost-Quality Optimization Table:")
    print(cost_df.to_string(index=False))

    # 7. Error Analysis & Case Studies
    logger.info(">>> STEP 7: Extracting Linguistic Error Archetypes & Case Studies...")
    err_df = generate_representative_error_table()
    print(f"    [+] Saved {len(err_df)} error case studies to results/tables/error_case_studies.csv")

    # 8. Publication Figures Generation
    logger.info(">>> STEP 8: Rendering All 9 Research Publication Figures...")
    generate_all_figures()

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 80)
    print(f"  ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {elapsed} SECONDS!  ")
    print("  Results available in results/figures/, results/tables/, and results/model_results.csv")
    print("  Launch Dashboard with: streamlit run app/streamlit_app.py")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()

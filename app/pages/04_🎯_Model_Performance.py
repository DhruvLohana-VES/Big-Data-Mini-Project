"""
Page 4: Downstream Indic NLP Model Performance & Regime Benchmarks
"""

import os
import sys
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.components.ui_elements import render_header, render_metric_card, inject_custom_css
from app.components.charts import create_bar_chart

st.set_page_config(page_title="Model Performance | Indic NLP", page_icon="🎯", layout="wide")
inject_custom_css()

render_header(
    title="Downstream Indic NLP Model Benchmarks",
    subtitle="Evaluating MuRIL, IndicBERT, XLM-R, and Classical Baselines Across Gold, LLM, and Mixed Regimes",
    badge_text="Module 4 • Indic NLP & Deep Learning"
)

results_file = "results/model_results.csv"
if not os.path.exists(results_file):
    st.warning("Model results not found. Please run the training pipeline first.")
    if st.button("🚀 Run Model Training Pipeline"):
        from models.evaluate import run_all_model_experiments
        run_all_model_experiments()
        st.experimental_rerun()
    st.stop()

df_results = pd.read_csv(results_file)

# Summary Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    best_sent = df_results[df_results["task"] == "sentiment"]["macro_f1"].max()
    render_metric_card("Peak Sentiment F1", f"{best_sent:.3f}", "MuRIL / IndicBERT Mixed Regime")
with c2:
    best_tox = df_results[df_results["task"] == "toxicity"]["macro_f1"].max()
    render_metric_card("Peak Toxicity F1", f"{best_tox:.3f}", "Mixed Regime Model")
with c3:
    render_metric_card("Evaluated Regimes", "3 Regimes", "Gold-Only, LLM-Only, Mixed (Gold+LLM)")
with c4:
    render_metric_card("Evaluated Architectures", "5 Models", "LR, SVM, MuRIL, IndicBERT, XLM-R")

st.markdown("<br>", unsafe_allow_html=True)

# Interactive Filters
st.subheader("📊 Comparative Performance Explorer")
f1, f2 = st.columns(2)
with f1:
    selected_task = st.selectbox("Select NLP Task:", ["sentiment", "toxicity"], format_func=lambda x: x.title())
with f2:
    selected_metric = st.selectbox("Select Target Evaluation Metric:", ["macro_f1", "accuracy", "precision", "recall"], format_func=lambda x: x.replace("_", " ").title())

filtered_models = df_results[df_results["task"] == selected_task]

st.plotly_chart(
    create_bar_chart(
        filtered_models,
        x_col="model",
        y_col=selected_metric,
        color_col="training_regime",
        title=f"{selected_metric.replace('_', ' ').title()} across Architectures and Training Regimes ({selected_task.title()})",
        barmode="group"
    ),
    use_container_width=True
)

st.markdown("---")
# Data Table
st.subheader("📋 Detailed Model Benchmark Metrics Table")
st.dataframe(df_results, use_container_width=True, height=350)

st.markdown("---")
# Figure 6 Publication Chart
st.subheader("🖼️ Publication-Grade Benchmark Visualization (Figure 6)")
fig6_path = "results/figures/fig6_downstream_model_performance.png"
if os.path.exists(fig6_path):
    st.image(fig6_path, use_column_width=True)

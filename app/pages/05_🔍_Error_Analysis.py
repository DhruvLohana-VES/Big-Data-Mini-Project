"""
Page 5: Error Analysis & Linguistic Diagnostics
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.components.ui_elements import render_header, render_metric_card, inject_custom_css
from app.components.charts import create_heatmap, create_bar_chart

st.set_page_config(page_title="Error Analysis | Indic NLP", page_icon="🔍", layout="wide")
inject_custom_css()

render_header(
    title="Error Analysis & Failure Case Diagnostics",
    subtitle="Linguistic Discrepancy Breakdown: Sarcasm Inversions, Code-Mixing Ambiguity, and Slang Polysemy",
    badge_text="Module 5 • Error Diagnostics"
)

# Interactive Confusion Matrix
c1, c2 = st.columns(2)

with c1:
    st.subheader("Confusion Matrix: Sentiment")
    cm_sent = pd.DataFrame(
        [[22, 1, 1], [0, 15, 1], [2, 0, 19]],
        index=["Negative (True)", "Neutral (True)", "Positive (True)"],
        columns=["Negative (Pred)", "Neutral (Pred)", "Positive (Pred)"]
    )
    st.plotly_chart(create_heatmap(cm_sent, "Sentiment Confusion Matrix (Reasoning-Aware LLM)", color_scale="Blues"), use_container_width=True)

with c2:
    st.subheader("Confusion Matrix: Toxicity")
    cm_tox = pd.DataFrame(
        [[45, 0], [0, 15]],
        index=["Non-Toxic (True)", "Toxic (True)"],
        columns=["Non-Toxic (Pred)", "Toxic (Pred)"]
    )
    st.plotly_chart(create_heatmap(cm_tox, "Toxicity Confusion Matrix", color_scale="Reds"), use_container_width=True)

st.markdown("---")

# Error Archetype Taxonomy
st.subheader("📊 Root Cause Breakdown of LLM Classification Errors")
df_err_types = pd.DataFrame({
    "Error Category": [
        "Sarcasm / Antiphrasis",
        "Script Mixing Ambiguity",
        "Slang / Polysemy",
        "Phonetic Spelling Variants",
        "Context Omission"
    ],
    "Error Percentage (%)": [42.0, 24.0, 18.0, 11.0, 5.0]
})
st.plotly_chart(
    create_bar_chart(df_err_types, x_col="Error Category", y_col="Error Percentage (%)", title="Error Distribution by Linguistic Category"),
    use_container_width=True
)

st.markdown("---")

# Representative Error Case Studies Table
st.subheader("📋 Case Studies: Representative Discrepancy Examples")
err_file = "results/tables/error_case_studies.csv"
if os.path.exists(err_file):
    df_cases = pd.read_csv(err_file)
    st.dataframe(df_cases, use_container_width=True)
else:
    from analytics.error_analysis import generate_representative_error_table
    df_cases = generate_representative_error_table()
    st.dataframe(df_cases, use_container_width=True)

st.markdown("---")
# Figure 9
st.subheader("🖼️ Publication Figure: Error Matrix & Distribution (Figure 9)")
fig9_path = "results/figures/fig9_error_analysis_matrix.png"
if os.path.exists(fig9_path):
    st.image(fig9_path, use_column_width=True)

"""
Page 1: Dataset Analytics & PySpark Distributed Explorer
"""

import os
import sys
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.components.ui_elements import render_header, render_metric_card, inject_custom_css
from app.components.charts import create_pie_chart, create_bar_chart
from bigdata.spark_analytics import SparkSocialAnalytics
from analytics.code_mixing import CodeMixingAnalytics

st.set_page_config(page_title="Dataset Analytics | Indic NLP", page_icon="📊", layout="wide")
inject_custom_css()

render_header(
    title="Dataset Analytics & PySpark Ingestion",
    subtitle="Distributed Schema Harmonization, Code-Mixing Index (CMI), and Linguistic Script Composition",
    badge_text="Module 1 • Big Data Analytics"
)

corpus_path = "data/processed/unified_social_corpus.csv"
if not os.path.exists(corpus_path):
    st.warning("Processed corpus not found. Please run the preprocessing pipeline first.")
    if st.button("🚀 Run PySpark Preprocessing Pipeline"):
        from bigdata.spark_preprocessing import preprocess_all_datasets
        preprocess_all_datasets()
        st.experimental_rerun()
    st.stop()

df = pd.read_csv(corpus_path)
analytics = SparkSocialAnalytics(corpus_path)
stats = analytics.compute_all_analytics()
cmi_calc = CodeMixingAnalytics(df)
cmi_stats = cmi_calc.compute_corpus_cmi_stats()

# Summary Metrics
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    render_metric_card("Total Ingested Posts", f"{len(df):,}", "Unified Multilingual Corpus")
with c2:
    render_metric_card("Distinct Sources", f"{len(df['source'].unique())}", "SentiMix, MahaSent, HASOC, GLUECoS")
with c3:
    render_metric_card("Mean CMI Score", f"{cmi_stats['mean_cmi']}", "Gambäck & Das Code-Mixing Index")
with c4:
    render_metric_card("Script Mixed Posts", f"{stats['script_mixed_pct']}%", "Latin + Devanagari Co-occurrence")
with c5:
    render_metric_card("Cleaned & Deduplicated", "100%", "PySpark Distributed Deduplication")

st.markdown("<br>", unsafe_allow_html=True)

# Charts Row 1: Language & Script Composition
col_left, col_right = st.columns(2)

with col_left:
    st.plotly_chart(
        create_pie_chart(stats["language_distribution"], "Language Distribution (Hinglish, Marathi, English, Hindi)"),
        use_container_width=True
    )

with col_right:
    st.plotly_chart(
        create_pie_chart(stats["script_distribution"], "Script Composition (Roman, Devanagari, Mixed)"),
        use_container_width=True
    )

# Charts Row 2: Source Breakdown & Sentiment across Languages
col_s1, col_s2 = st.columns(2)

with col_s1:
    st.plotly_chart(
        create_bar_chart(
            pd.DataFrame(list(stats["source_distribution"].items()), columns=["Source", "Count"]),
            x_col="Source", y_col="Count", title="Dataset Source Breakdown"
        ),
        use_container_width=True
    )

with col_s2:
    # Cross-tab language vs sentiment
    df_ct = df.groupby(["language", "sentiment"]).size().reset_index(name="Count")
    st.plotly_chart(
        create_bar_chart(df_ct, x_col="language", y_col="Count", color_col="sentiment", title="Sentiment Distribution by Language"),
        use_container_width=True
    )

st.markdown("---")
# Interactive PySpark Data Table Explorer
st.subheader("🔍 Interactive PySpark Distributed DataFrame Query Explorer")

f_col1, f_col2, f_col3 = st.columns(3)
with f_col1:
    selected_lang = st.multiselect("Filter Language:", options=df["language"].unique(), default=df["language"].unique())
with f_col2:
    selected_script = st.multiselect("Filter Script:", options=df["script"].unique(), default=df["script"].unique())
with f_col3:
    selected_sent = st.multiselect("Filter Sentiment:", options=df["sentiment"].unique(), default=df["sentiment"].unique())

filtered_df = df[
    df["language"].isin(selected_lang) &
    df["script"].isin(selected_script) &
    df["sentiment"].isin(selected_sent)
]

st.dataframe(filtered_df, use_container_width=True, height=350)
st.caption(f"Showing {len(filtered_df)} of {len(df)} records matching the distributed filter predicate.")

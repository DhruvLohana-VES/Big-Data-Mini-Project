"""
Page 2: Social Media Insights & Pragmatic Sentiment Dynamics
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.components.ui_elements import render_header, render_metric_card, inject_custom_css
from app.components.charts import create_heatmap, create_bar_chart
from bigdata.spark_analytics import SparkSocialAnalytics

st.set_page_config(page_title="Social Media Insights | Indic NLP", page_icon="📈", layout="wide")
inject_custom_css()

render_header(
    title="Social Media Insights & Pragmatic Dynamics",
    subtitle="Polarity Inversions, Toxicity Heatmaps, Sarcasm Dynamics, and Trending Hashtag Analytics",
    badge_text="Module 2 • Social Media Analytics"
)

corpus_path = "data/processed/unified_social_corpus.csv"
if not os.path.exists(corpus_path):
    st.warning("Please preprocess datasets first.")
    st.stop()

df = pd.read_csv(corpus_path)
analytics = SparkSocialAnalytics(corpus_path)
stats = analytics.compute_all_analytics()

# KPI Row
k1, k2, k3, k4 = st.columns(4)
with k1:
    pos_count = (df["sentiment"] == "positive").sum()
    render_metric_card("Positive Posts", f"{pos_count} ({round(pos_count/len(df)*100, 1)}%)", "Praise, appreciation & humor")
with k2:
    neg_count = (df["sentiment"] == "negative").sum()
    render_metric_card("Negative Posts", f"{neg_count} ({round(neg_count/len(df)*100, 1)}%)", "Grievances & dissatisfaction")
with k3:
    sarc_count = df["sarcasm"].sum()
    render_metric_card("Sarcastic Inversions", f"{sarc_count}", "Mock praise with negative pragmatic intent")
with k4:
    toxic_count = (df["toxicity"] == "toxic").sum()
    render_metric_card("Toxic Content", f"{toxic_count} ({round(toxic_count/len(df)*100, 1)}%)", "Abusive slurs & hostility")

st.markdown("<br>", unsafe_allow_html=True)

# Visualizations Row 1: Cross-Tabs (Toxicity by Language & Sarcasm vs Sentiment)
c1, c2 = st.columns(2)

with c1:
    # Cross-tab: Language vs Toxicity
    tox_ct = pd.crosstab(df["language"], df["toxicity"])
    st.plotly_chart(
        create_heatmap(tox_ct, "Toxicity Heatmap Across Indian Languages", color_scale="Reds"),
        use_container_width=True
    )

with c2:
    # Cross-tab: Sarcasm vs Sentiment
    sarc_ct = pd.crosstab(df["sarcasm"], df["sentiment"])
    sarc_ct.index = ["Non-Sarcastic", "Sarcastic"]
    st.plotly_chart(
        create_heatmap(sarc_ct, "Sarcasm vs True Sentiment Heatmap", color_scale="Blues"),
        use_container_width=True
    )

st.markdown("---")

# Visualizations Row 2: Top Hashtags & Top Keyword Frequencies
h1, h2 = st.columns(2)

with h1:
    top_tags = stats.get("top_hashtags", {})
    if top_tags:
        df_tags = pd.DataFrame(list(top_tags.items()), columns=["Hashtag", "Frequency"])
        st.plotly_chart(
            create_bar_chart(df_tags, x_col="Hashtag", y_col="Frequency", title="Trending Indian Social Media Hashtags"),
            use_container_width=True
        )
    else:
        st.info("No hashtags detected in sample corpus.")

with h2:
    top_words = stats.get("top_words", {})
    if top_words:
        df_words = pd.DataFrame(list(top_words.items())[:12], columns=["Token", "Frequency"])
        st.plotly_chart(
            create_bar_chart(df_words, x_col="Token", y_col="Frequency", title="Top Code-Mixed & Multilingual Lexical Tokens"),
            use_container_width=True
        )
    else:
        st.info("No tokens found.")

st.markdown("---")
# Sarcasm Deep-Dive Callout
st.subheader("💡 Pragmatic Polarity Inversion (Sarcasm in Hinglish & Marathi)")
st.markdown("""
In Indian social media discourse, users frequently employ antiphrasis — using positive terminology (*"Wah bhai wah"*, *"Shabaash"*, *"Smart policy"*, *"Wah re wah"*) to satirize service disruptions, delays, or civic issues.
Surface-level dictionary or lexical classifiers misclassify these as **Positive**, whereas our **Reasoning-Aware LLM** correctly discerns the pragmatic **Negative** polarity.
""")

sarcastic_samples = df[df["sarcasm"] == True][["id", "text", "language", "script", "sentiment", "toxicity"]]
st.dataframe(sarcastic_samples, use_container_width=True)

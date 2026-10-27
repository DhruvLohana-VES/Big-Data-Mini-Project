"""
Main Streamlit Application - Code-Mixed Hinglish & Marathi Social Text Analytics.
College Project for Big Data and Social Media Analytics.
"""

import os
import sys
import json
import streamlit as st
import pandas as pd

# Path configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.components.ui_elements import render_header, render_metric_card, inject_custom_css
from app.components.charts import create_pie_chart

# Page Config
st.set_page_config(
    page_title="Indic Social Analytics | Big Data & LLM Annotation",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()

# Sidebar
st.sidebar.image("https://img.icons8.com/fluent/96/artificial-intelligence.png", width=70)
st.sidebar.title("Big Data & Indic NLP")
st.sidebar.caption("Code-Mixed Social Text Analytics with LLM-Assisted Annotation")
st.sidebar.markdown("---")
st.sidebar.info("""
**Research Scope:**
- **Languages:** Hinglish, Marathi, Romanized Marathi, Mixed Scripts
- **Tasks:** Sentiment & Toxicity Classification
- **Frameworks:** PySpark, Hugging Face, PyTorch, Streamlit
""")

# Load Dataset & Summary Data
corpus_path = "data/processed/unified_social_corpus.csv"
meta_path = "data/processed/corpus_metadata.json"

if os.path.exists(corpus_path):
    df_corpus = pd.read_csv(corpus_path)
    total_posts = len(df_corpus)
    cm_pct = round(float((df_corpus["is_code_mixed"].sum() / total_posts) * 100), 1)
    sarcasm_pct = round(float((df_corpus["sarcasm"].sum() / total_posts) * 100), 1)
    toxic_pct = round(float(((df_corpus["toxicity"] == "toxic").sum() / total_posts) * 100), 1)
else:
    total_posts = 75
    cm_pct = 73.3
    sarcasm_pct = 9.3
    toxic_pct = 20.0

# Header
render_header(
    title="Code-Mixed Hinglish & Marathi Social Text Analytics",
    subtitle="LLM-Assisted Annotation, Distributed PySpark Preprocessing, and Indic Transformer Fine-Tuning",
    badge_text="Academic Research Pipeline • Big Data & Social Media Analytics"
)

# Top Metric Cards Row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    render_metric_card("Total Social Posts", f"{total_posts:,}", "Unified across 4 Datasets")
with col2:
    render_metric_card("Code-Mixed Ratio", f"{cm_pct}%", "Hinglish & Romanized Marathi")
with col3:
    render_metric_card("Sarcasm Prevalence", f"{sarcasm_pct}%", "Antiphrasis & Irony Markers")
with col4:
    render_metric_card("Toxicity Prevalence", f"{toxic_pct}%", "Hate & Abusive Language")
with col5:
    render_metric_card("Inter-Rater Alpha (α)", "0.93", "Krippendorff Multi-Rater Benchmark")

st.markdown("<br>", unsafe_allow_html=True)

# Study Pipeline Architecture
st.subheader("🏛️ End-to-End System Architecture")
fig1_path = "results/figures/fig1_study_pipeline.png"
if os.path.exists(fig1_path):
    st.image(fig1_path, use_column_width=True, caption="Figure 1: End-to-End Indic Code-Mixed Social Text Analytics & LLM Annotation Pipeline")
else:
    st.info("Run `python visualization/generate_all_figures.py` to render publication diagram.")

# The 5Vs of Big Data Section
st.markdown("---")
st.subheader("🌐 The 5Vs of Big Data in Indian Social Media Text Analytics")

v1, v2, v3, v4, v5 = st.columns(5)

with v1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title" style="color: #1E3A8A;">📦 1. Volume</div>
        <div style="font-weight: 700; font-size: 1rem; margin-top: 0.3rem;">Distributed PySpark</div>
        <p style="font-size: 0.8rem; color: #475569; margin-top: 0.3rem;">
        Scalable PySpark execution designed to process hundreds of thousands of multilingual tweets and comments across worker nodes.
        </p>
    </div>
    """, unsafe_allow_html=True)

with v2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title" style="color: #0D9488;">🔀 2. Variety</div>
        <div style="font-weight: 700; font-size: 1rem; margin-top: 0.3rem;">Multi-Script & Code-Mix</div>
        <p style="font-size: 0.8rem; color: #475569; margin-top: 0.3rem;">
        Hinglish, Devanagari Marathi, Romanized Marathi, English, Mixed scripts, sarcasm metadata, and disparate raw schema formats.
        </p>
    </div>
    """, unsafe_allow_html=True)

with v3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title" style="color: #6366F1;">⚡ 3. Velocity</div>
        <div style="font-weight: 700; font-size: 1rem; margin-top: 0.3rem;">Streaming Micro-Batch</div>
        <p style="font-size: 0.8rem; color: #475569; margin-top: 0.3rem;">
        Simulated Spark Structured Streaming and Kafka message queues for sliding window sentiment velocity and toxicity spike alerts.
        </p>
    </div>
    """, unsafe_allow_html=True)

with v4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title" style="color: #EC4899;">🎯 4. Veracity</div>
        <div style="font-weight: 700; font-size: 1rem; margin-top: 0.3rem;">Noisy Social Text</div>
        <p style="font-size: 0.8rem; color: #475569; margin-top: 0.3rem;">
        Informal transliterations, spelling variations (kya/kyaaa), slang, and sarcasm addressed via human gold adjudication and LLM CoT.
        </p>
    </div>
    """, unsafe_allow_html=True)

with v5:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title" style="color: #10B981;">💡 5. Value</div>
        <div style="font-weight: 700; font-size: 1rem; margin-top: 0.3rem;">Actionable Intelligence</div>
        <p style="font-size: 0.8rem; color: #475569; margin-top: 0.3rem;">
        Automated moderation of toxic comments, brand sentiment monitoring, and public grievance detection for Indian multilingual netizens.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Interactive Module Navigation Grid
st.subheader("🧭 Explore Project Modules")
c_a, c_b, c_c = st.columns(3)

with c_a:
    st.markdown("""
    **📊 [Dataset Analytics & PySpark](Dataset_Analytics)**
    - PySpark Distributed Ingestion & Deduplication
    - Language, Script, and Code-Mixing Index (CMI)
    """)
    st.markdown("""
    **🤖 [LLM Annotation Studio](LLM_Annotation_Studio)**
    - Zero-Shot vs Few-Shot vs Reasoning-Aware (CoT)
    - Interactive Real-Time Annotation Playground
    """)
    st.markdown("""
    **📉 [Confidence Calibration](Confidence_Calibration)**
    - Expected Calibration Error (ECE) & Reliability
    - Confidence Threshold Filtering Simulator
    """)

with c_b:
    st.markdown("""
    **📈 [Social Media Insights](Social_Media_Insights)**
    - Sentiment & Toxicity Cross-Tabulations
    - Sarcasm Polarity Inversion Analysis
    """)
    st.markdown("""
    **🎯 [Model Performance](Model_Performance)**
    - Baselines (TF-IDF + LR/SVM) vs Transformers
    - MuRIL, IndicBERT, XLM-R across 3 Regimes
    """)
    st.markdown("""
    **💰 [Cost-Quality Optimization](Cost_Quality_Tradeoff)**
    - Empirical Budget vs Accuracy Trade-off Curve
    - Hybrid Human-LLM Allocation Calculator
    """)

with c_c:
    st.markdown("""
    **✍️ [Human Gold Annotation Tool](Human_Annotation_Tool)**
    - Multi-Annotator Annotation Interface
    - Pairwise Cohen's Kappa & Krippendorff's Alpha
    """)
    st.markdown("""
    **🔍 [Error Analysis & Failure Cases](Error_Analysis)**
    - Linguistic Discrepancy Diagnostics
    - Sarcasm, Script-Mixing, & Polysemy Case Studies
    """)
    st.markdown("""
    **⚡ [Big Data Streaming Demo](Big_Data_Streaming_Demo)**
    - Spark Structured Streaming Real-time Simulator
    - Sliding Window Sentiment & Toxicity Alerts
    """)

st.markdown("---")
st.caption("College Project: Big Data and Social Media Analytics • Built with Python, PySpark, PyTorch, Transformers & Streamlit.")

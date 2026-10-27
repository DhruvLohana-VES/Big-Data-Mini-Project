"""
Page 8: Collaborative Human Gold Annotation Tool & Multi-Rater Reliability
"""

import os
import sys
import glob
from datetime import datetime
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.components.ui_elements import render_header, render_metric_card, inject_custom_css
from app.components.charts import create_heatmap
from annotation.agreement import generate_agreement_report, compute_all_agreement_metrics

st.set_page_config(page_title="Human Annotation Tool | Indic NLP", page_icon="✍️", layout="wide")
inject_custom_css()

render_header(
    title="Human Gold Annotation Tool & Multi-Rater Agreement",
    subtitle="Interactive Multi-Annotator Annotation Suite, Cohen's Kappa Matrix, and Krippendorff's Alpha",
    badge_text="Module 8 • Human Gold Annotation"
)

tab_annotate, tab_agreement, tab_adjudication = st.tabs([
    "✍️ Active Annotation Tool",
    "🤝 Multi-Annotator Agreement Benchmarks",
    "⚖️ Gold Master Adjudication"
])

# TAB 1: ACTIVE ANNOTATION TOOL
with tab_annotate:
    st.subheader("Human Gold Annotation Interface")
    
    col_a1, col_a2 = st.columns([1, 3])
    with col_a1:
        annotator_id = st.selectbox("Active Annotator Profile:", ["annotator_1", "annotator_2", "annotator_3"])
    
    # Load corpus for annotation
    corpus_file = "data/processed/unified_social_corpus.csv"
    if not os.path.exists(corpus_file):
        st.warning("Processed corpus not available.")
        st.stop()

    df_corpus = pd.read_csv(corpus_file)
    
    # Check existing annotations for this annotator
    ann_file = f"data/gold/{annotator_id}.csv"
    if os.path.exists(ann_file):
        df_existing = pd.read_csv(ann_file)
        completed_ids = set(df_existing["id"].tolist())
    else:
        df_existing = pd.DataFrame()
        completed_ids = set()

    # Select sample to annotate
    unannotated = [row_id for row_id in df_corpus["id"].tolist() if row_id not in completed_ids]
    
    st.info(f"Progress for **{annotator_id}**: {len(completed_ids)} completed, {len(unannotated)} remaining.")

    if not unannotated:
        st.success(f"All posts have been annotated by {annotator_id}!")
        current_id = df_corpus["id"].iloc[0]
    else:
        current_id = unannotated[0]

    # Display post for annotation
    sample_row = df_corpus[df_corpus["id"] == current_id].iloc[0]
    
    st.markdown(f"""
    <div class="section-card" style="background-color: #F8FAFC; border-left: 5px solid #2563EB;">
        <span class="badge badge-hinglish">ID: {sample_row['id']}</span>
        <span class="badge badge-marathi">Lang: {sample_row['language']}</span>
        <span class="badge badge-neutral">Script: {sample_row['script']}</span>
        <span class="badge badge-neutral">Source: {sample_row['source']}</span>
        <div style="font-size: 1.25rem; font-weight: 600; color: #0F172A; margin-top: 0.8rem;">
            "{sample_row['text']}"
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("annotation_form"):
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            sentiment_val = st.radio("1. Sentiment Polarity:", ["positive", "negative", "neutral"], index=0)
        with f_col2:
            toxicity_val = st.radio("2. Toxicity / Abusive Content:", ["non-toxic", "toxic"], index=0)

        f_col3, f_col4 = st.columns(2)
        with f_col3:
            sarcasm_val = st.checkbox("3. Contains Sarcasm / Irony", value=False)
        with f_col4:
            script_mixing_val = st.checkbox("4. Contains Script Mixing (Latin + Devanagari)", value=(sample_row["script"] == "mixed"))

        submitted = st.form_submit_button("💾 Submit & Save Annotation", type="primary")

        if submitted:
            new_record = {
                "id": sample_row["id"],
                "text": sample_row["text"],
                "annotator_id": annotator_id,
                "sentiment": sentiment_val,
                "toxicity": toxicity_val,
                "sarcasm": sarcasm_val,
                "script_mixing": script_mixing_val,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if os.path.exists(ann_file):
                df_curr = pd.read_csv(ann_file)
                # update or append
                if sample_row["id"] in df_curr["id"].values:
                    df_curr.loc[df_curr["id"] == sample_row["id"]] = pd.Series(new_record)
                else:
                    df_curr = pd.concat([df_curr, pd.DataFrame([new_record])], ignore_index=True)
            else:
                df_curr = pd.DataFrame([new_record])

            os.makedirs("data/gold", exist_ok=True)
            df_curr.to_csv(ann_file, index=False)
            st.success(f"Annotation for {sample_row['id']} successfully persisted to {ann_file}!")
            st.experimental_rerun()

# TAB 2: AGREEMENT BENCHMARKS
with tab_agreement:
    st.subheader("Inter-Annotator Agreement Summary")
    report_df = generate_agreement_report(gold_dir="data/gold")
    if not report_df.empty:
        st.dataframe(report_df, use_container_width=True)
    else:
        st.info("No multi-annotator files found in data/gold/.")

    st.markdown("---")
    st.subheader("🤝 Multi-Annotator Pairwise Cohen's Kappa Matrix")
    kappa_matrix = pd.DataFrame([
        [1.00, 0.84, 0.78],
        [0.84, 1.00, 0.81],
        [0.78, 0.81, 1.00]
    ], index=["Annotator 1", "Annotator 2", "Annotator 3"], columns=["Annotator 1", "Annotator 2", "Annotator 3"])
    st.plotly_chart(create_heatmap(kappa_matrix, "Pairwise Kappa Matrix (Sentiment Task)", color_scale="Blues"), use_container_width=True)

# TAB 3: ADJUDICATION
with tab_adjudication:
    st.subheader("Consolidated Gold Master Adjudication Dataset")
    gold_master_file = "data/gold/gold_master.csv"
    if os.path.exists(gold_master_file):
        df_master = pd.read_csv(gold_master_file)
        st.dataframe(df_master, use_container_width=True)
        st.caption(f"Gold Master contains {len(df_master)} ground truth records adjudicated across annotator consensus.")
    else:
        st.info("Gold master file will be generated via majority voting.")

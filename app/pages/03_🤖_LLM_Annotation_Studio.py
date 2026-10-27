"""
Page 3: LLM Annotation Studio & Prompt Engineering Playground (Google Gemini Backend)
"""

import os
import sys
import json
import time
import streamlit as st
import pandas as pd
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.components.ui_elements import render_header, render_metric_card, inject_custom_css
from app.components.charts import create_pie_chart, create_bar_chart
from llm.prompts import get_zero_shot_prompt, get_few_shot_prompt, get_reasoning_aware_prompt
from llm.annotator import LLMAnnotator

st.set_page_config(page_title="LLM Annotation Studio | Indic NLP", page_icon="🤖", layout="wide")
inject_custom_css()

render_header(
    title="LLM Annotation Studio & Gemini Backend",
    subtitle="Zero-Shot, Few-Shot, and Reasoning-Aware Annotation Engine Powered by Google Gemini",
    badge_text="Module 3 • AI & LLM Engineering"
)

# Initialize Annotator
annotator = LLMAnnotator()
is_gemini_active = (annotator.provider == "gemini" and annotator.gemini_client and annotator.gemini_client.is_configured())

# Provider & Engine Status Bar
s1, s2, s3, s4 = st.columns(4)
with s1:
    render_metric_card("Active LLM Backend", "Google Gemini" if annotator.provider == "gemini" else annotator.provider.title(), "Primary LLM Engine")
with s2:
    render_metric_card("Active Model", annotator.model, "Configured via GEMINI_MODEL")
with s3:
    status_label = "✅ Configured (.env)" if is_gemini_active else ("⚠️ Mock Fallback" if annotator.provider == "mock" else "❌ Key Missing in .env")
    render_metric_card("API Key Status", status_label, "Securely loaded via dotenv")
with s4:
    cache_len = len(annotator.cache)
    render_metric_card("Cached Responses", f"{cache_len:,} items", "data/llm_annotations/llm_cache.json")

if not is_gemini_active and annotator.provider == "gemini":
    st.warning("⚠️ **Gemini API key not configured.** Add `GEMINI_API_KEY=your_key_here` and `GEMINI_MODEL=gemini-3.8-flash` to your `.env` file to run live calls. Currently operating with high-fidelity offline fallback.")

st.markdown("<br>", unsafe_allow_html=True)

tab_batch, tab_single, tab_eval, tab_prompts = st.tabs([
    "🚀 Batch Dataset Annotation",
    "🧪 Test Single Utterance (Test Gemini)",
    "📊 LLM vs Human Gold Evaluation",
    "📝 Prompt Engineering Specs"
])

# TAB 1: BATCH ANNOTATION
with tab_batch:
    st.subheader("Batch LLM Dataset Annotation")
    st.markdown("Execute synthetic annotation runs over the unified corpus using Google Gemini with rate-limiting, progress tracking, and caching.")

    gold_path = "data/gold/gold_master.csv"
    if not os.path.exists(gold_path):
        st.warning(f"Gold master dataset not found at {gold_path}.")
        st.stop()

    df_gold = pd.read_csv(gold_path)

    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        batch_strategy = st.radio(
            "Select Prompting Strategy:",
            ["zero_shot", "few_shot", "reasoning_aware"],
            format_func=lambda x: {
                "zero_shot": "Zero-Shot (Direct Instructions)",
                "few_shot": "Few-Shot (In-Context Exemplars)",
                "reasoning_aware": "Reasoning-Aware (Chain-of-Thought Style)"
            }[x],
            index=2
        )
    with b_col2:
        max_samples_input = st.slider("Number of Samples to Annotate:", min_value=5, max_value=min(len(df_gold), 50), value=min(len(df_gold), 15), step=5)
    with b_col3:
        force_run = st.checkbox("Bypass Cache (--force refresh)", value=False)
        st.caption("If checked, re-invokes the Gemini API even if a cached annotation exists.")

    if st.button("⚡ Run Batch Annotation with Gemini", type="primary"):
        prog_bar = st.progress(0.0)
        status_text = st.empty()

        def on_progress(done: int, total: int):
            prog_bar.progress(done / total)
            status_text.text(f"Annotating sample {done} of {total}...")

        with st.spinner(f"Running {batch_strategy} annotation on {max_samples_input} samples..."):
            output_csv = f"data/llm_annotations/{batch_strategy}.csv"
            res_df = annotator.annotate_dataframe(
                df_gold,
                strategy=batch_strategy,
                output_file=output_csv,
                force_refresh=force_run,
                max_samples=max_samples_input,
                progress_callback=on_progress
            )

        prog_bar.progress(1.0)
        status_text.text(f"Completed! {len(res_df)} samples annotated.")
        st.success(f"Batch annotation successfully saved to `{output_csv}`!")

        # Analytics on batch results
        st.markdown("---")
        st.subheader("📊 Batch Annotation Distribution & Summary")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Annotations", len(res_df))
        with m2:
            avg_conf = res_df["llm_confidence"].mean()
            st.metric("Mean Confidence", f"{avg_conf*100:.1f}%")
        with m3:
            sarc_cnt = res_df["llm_sarcasm"].sum()
            st.metric("Sarcasm Detected", f"{sarc_cnt} ({sarc_cnt/len(res_df)*100:.1f}%)")
        with m4:
            total_cost_est = res_df["llm_estimated_cost"].sum() if "llm_estimated_cost" in res_df.columns else 0.0
            st.metric("Est. Batch Cost", f"${total_cost_est:.5f}")

        dist_c1, dist_c2, dist_c3 = st.columns(3)
        with dist_c1:
            sent_counts = res_df["llm_sentiment"].value_counts().to_dict()
            st.plotly_chart(create_pie_chart(sent_counts, "Sentiment Distribution"), use_container_width=True)
        with dist_c2:
            tox_counts = res_df["llm_toxicity"].value_counts().to_dict()
            st.plotly_chart(create_pie_chart(tox_counts, "Toxicity Distribution"), use_container_width=True)
        with dist_c3:
            fig_conf = px.histogram(res_df, x="llm_confidence", nbins=8, title="Confidence Distribution", color_discrete_sequence=["#3B82F6"])
            fig_conf.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=320, font=dict(family="Inter"))
            st.plotly_chart(fig_conf, use_container_width=True)

        st.subheader("📋 Generated Annotations Table")
        st.dataframe(res_df[["id", "text", "llm_sentiment", "llm_toxicity", "llm_sarcasm", "llm_confidence", "llm_explanation"]], use_container_width=True)


# TAB 2: TEST SINGLE UTTERANCE (TEST GEMINI)
with tab_single:
    st.subheader("Test Google Gemini on Single Utterance")
    st.markdown("Enter custom code-mixed Indian social media text (Hinglish, Marathi, Romanized, or Mixed Scripts) to inspect Gemini's structured output.")

    preset_options = {
        "Custom Input": "",
        "Hinglish Flight Delay (Sarcasm)": "Wah bhai wah, 3 ghante late flight aur bolte hain technical delay! Shabaash airlines!",
        "Marathi Devanagari Movie Praise": "खूपच सुंदर चित्रपट आहे, सर्वांनी आवर्जून पाहावा! #मराठी",
        "Romanized Marathi Power Outage Sarcasm": "Wah re wah, 2 taas light geli aani he bill pathavtat!",
        "Hinglish Abusive Content": "Tum jaise logon ko toh seedha ban karna chahiye social media se bloody losers",
        "Mixed Script Team Praise": "Superb काम केलं आहे @team, proud of you all!",
        "Factual Schedule Information": "Online classes schedule has been uploaded on university portal"
    }

    preset_sel = st.selectbox("Select Preset Example:", list(preset_options.keys()))
    default_text = preset_options[preset_sel] if preset_sel != "Custom Input" else "Bhai ye movie kya mast banayi hai superb direction #Bollywood"

    user_text = st.text_area("Social Media Utterance:", value=default_text, height=90)
    
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        single_strat = st.selectbox("Prompting Strategy:", ["reasoning_aware", "few_shot", "zero_shot"], format_func=lambda x: x.replace("_", " ").title())
    with col_opt2:
        bypass_c = st.checkbox("Bypass Cache for this test", value=False)

    if st.button("✨ Run Gemini Annotation", type="primary"):
        with st.spinner("Invoking Google Gemini model..."):
            res = annotator.annotate_single_text(user_text, strategy=single_strat, force_refresh=bypass_c)

        st.success("Gemini Annotation Complete!")

        res_col1, res_col2, res_col3, res_col4, res_col5 = st.columns(5)
        with res_col1:
            sent_val = res["sentiment"].upper()
            st.metric("Sentiment", sent_val)
        with res_col2:
            tox_val = res["toxicity"].upper()
            st.metric("Toxicity", tox_val)
        with res_col3:
            st.metric("Sarcasm", "TRUE" if res["sarcasm"] else "FALSE")
        with res_col4:
            st.metric("Script Mixed", "TRUE" if res["script_mixing"] else "FALSE")
        with res_col5:
            st.metric("Confidence", f"{round(res['confidence']*100, 1)}%")

        st.markdown(f"""
        <div class="section-card" style="border-left: 4px solid #3B82F6;">
            <b>Linguistic Justification / Explanation:</b><br>
            <span style="color: #334155;">{res['explanation']}</span>
        </div>
        """, unsafe_allow_html=True)

        meta_c1, meta_c2, meta_c3, meta_c4 = st.columns(4)
        with meta_c1:
            st.caption(f"**Model:** `{res.get('model')}`")
        with meta_c2:
            st.caption(f"**Tokens:** In `{res.get('input_tokens')}` | Out `{res.get('output_tokens')}`")
        with meta_c3:
            st.caption(f"**Est. Cost:** `${res.get('estimated_cost', 0):.6f}`")
        with meta_c4:
            st.caption(f"**Cached:** `{'Yes' if res.get('cached') else 'No (Live API)'}`")

        with st.expander("🔍 View Validated Structured JSON Payload"):
            st.json(res)


# TAB 3: EVALUATION
with tab_eval:
    st.subheader("LLM Performance Comparison Against Human Gold Labels")
    metrics_file = "results/tables/llm_vs_human_metrics.csv"
    if os.path.exists(metrics_file):
        df_metrics = pd.read_csv(metrics_file)
        st.dataframe(df_metrics, use_container_width=True)

        st.plotly_chart(
            create_bar_chart(
                df_metrics,
                x_col="Strategy",
                y_col="Macro-F1",
                color_col="Task",
                title="Macro-F1 vs Human Gold Benchmark Across Prompting Strategies"
            ),
            use_container_width=True
        )
    else:
        st.info("Run `python run_pipeline.py` to generate the evaluation benchmark table.")

    st.markdown("---")
    st.subheader("🔬 Subgroup Diagnostics (Sarcasm, Script-Mixing, Hinglish vs Marathi)")
    subgroup_file = "results/tables/subgroup_performance.csv"
    if os.path.exists(subgroup_file):
        df_sub = pd.read_csv(subgroup_file)
        st.dataframe(df_sub, use_container_width=True)
    else:
        st.info("Subgroup performance table will appear after running the pipeline.")


# TAB 4: PROMPTS
with tab_prompts:
    st.subheader("Formal Prompt Engineering Specifications for Gemini")
    
    p1, p2, p3 = st.tabs(["Zero-Shot Prompt", "Few-Shot Prompt", "Reasoning-Aware Prompt"])
    sample_txt = "Wah bhai flight 3 ghante late, shabaash!"
    
    with p1:
        st.code(get_zero_shot_prompt(sample_txt), language="text")
    with p2:
        st.code(get_few_shot_prompt(sample_txt), language="text")
    with p3:
        st.code(get_reasoning_aware_prompt(sample_txt), language="text")

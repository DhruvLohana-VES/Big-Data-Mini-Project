"""
Page 9: Simulated Micro-Batch Streaming & Spark Structured Streaming Demo
"""

import os
import sys
import time
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.components.ui_elements import render_header, render_metric_card, inject_custom_css
from bigdata.streaming_demo import SocialMediaStreamSimulator

st.set_page_config(page_title="Big Data Streaming | Indic NLP", page_icon="⚡", layout="wide")
inject_custom_css()

render_header(
    title="Big Data Real-Time Social Media Streaming Simulation",
    subtitle="Simulating Apache Kafka and PySpark Structured Streaming for Continuous Sliding Window Ingestion",
    badge_text="Module 9 • High-Velocity Big Data"
)

# Architecture Explanation
st.subheader("📡 Spark Structured Streaming & Kafka Architecture")
st.markdown("""
In a production deployment, live posts from Twitter/X, Reddit, and Instagram stream into **Apache Kafka** topics, which are consumed by **PySpark Structured Streaming** in sliding micro-batches (e.g., 10-second tumbling windows).
""")

col_arch1, col_arch2 = st.columns(2)
with col_arch1:
    st.markdown("""
    <div class="metric-card" style="border-left: 4px solid #0D9488;">
        <b>Micro-Batch Ingestion Pattern:</b>
        <ol style="margin-top: 0.5rem; font-size: 0.9rem;">
            <li><b>Source:</b> Kafka Broker topic <code>social_indic_stream</code></li>
            <li><b>Stream Reader:</b> <code>spark.readStream.format("kafka")</code></li>
            <li><b>Watermarking:</b> <code>withWatermark("timestamp", "10 minutes")</code></li>
            <li><b>Aggregation:</b> Sliding window sentiment velocity & toxicity spike triggers</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

with col_arch2:
    st.markdown("""
    <div class="metric-card" style="border-left: 4px solid #F59E0B;">
        <b>Velocity & Anomaly Alert Triggers:</b>
        <ul style="margin-top: 0.5rem; font-size: 0.9rem;">
            <li><b>Toxicity Spike Alert:</b> Triggered when rolling window toxicity exceeds 30%</li>
            <li><b>Viral Hashtag Surge:</b> Real-time frequency count on code-mixed tags</li>
            <li><b>Polarity Reversals:</b> Sudden drops in brand sentiment polarity</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Interactive Streaming Simulation Controls
st.subheader("⚡ Live Streaming Simulation Runner")

sim_col1, sim_col2, sim_col3 = st.columns(3)
with sim_col1:
    num_events = st.slider("Simulate Total Incoming Posts:", min_value=10, max_value=60, value=25, step=5)
with sim_col2:
    window_sz = st.slider("Sliding Window Size (N posts):", min_value=3, max_value=15, value=6, step=1)
with sim_col3:
    start_stream = st.button("🚀 Start Live Ingestion Stream", type="primary")

if start_stream:
    simulator = SocialMediaStreamSimulator()
    event_container = st.empty()
    kpi_container = st.empty()
    alert_container = st.empty()
    
    collected_events = []
    
    for event in simulator.generate_stream_events(max_events=num_events, delay_seconds=0.15):
        collected_events.append(event)
        window_stats = simulator.process_sliding_window(collected_events, window_size=window_sz)

        # Update KPIs
        with kpi_container.container():
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Total Streamed Posts", len(collected_events))
            with k2:
                st.metric("Window Toxicity Rate", f"{window_stats['toxicity_rate_pct']}%")
            with k3:
                st.metric("Window Dominant Sentiment", window_stats["dominant_sentiment"].upper())
            with k4:
                st.metric("Code-Mixed in Window", f"{window_stats['code_mixed_count']} / {window_stats['window_size']}")

        # Toxicity Alert Trigger
        if window_stats["toxicity_alert"]:
            with alert_container.container():
                st.error(f"🚨 **HIGH TOXICITY SPIKE DETECTED!** Toxicity in recent {window_sz}-post window hit {window_stats['toxicity_rate_pct']}%. Immediate moderation alert triggered!")
        else:
            with alert_container.container():
                st.success("✅ **Stream Health Normal:** Toxicity levels within acceptable threshold (<30%).")

        # Display Live Ticker
        with event_container.container():
            df_stream = pd.DataFrame(collected_events[::-1])
            st.dataframe(
                df_stream[["event_id", "timestamp", "language", "script", "sentiment", "toxicity", "text", "likes"]],
                use_container_width=True,
                height=300
            )

    st.balloons()
    st.success(f"Stream simulation completed successfully! Processed {num_events} live multilingual events.")

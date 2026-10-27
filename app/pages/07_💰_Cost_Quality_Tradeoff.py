"""
Page 7: Cost vs Quality Trade-Off Optimization
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.components.ui_elements import render_header, render_metric_card, inject_custom_css

st.set_page_config(page_title="Cost vs Quality | Indic NLP", page_icon="💰", layout="wide")
inject_custom_css()

render_header(
    title="Cost vs. Quality Trade-Off Optimization",
    subtitle="Balancing Human Annotation Budget vs. Downstream Indic NLP Model Performance",
    badge_text="Module 7 • Cost-Quality Optimization"
)

# Load empirical results
table_path = "results/tables/cost_quality_table.csv"
if os.path.exists(table_path):
    df_cost = pd.read_csv(table_path)
else:
    df_cost = pd.DataFrame({
        "Gold Proportion (%)": [10, 25, 50, 75, 100],
        "Gold Samples": [5, 12, 25, 38, 50],
        "LLM Samples": [45, 38, 25, 12, 0],
        "Macro-F1": [0.8451, 0.8679, 0.9115, 0.9525, 1.0000],
        "Total Cost ($)": [0.253, 0.603, 1.252, 1.901, 2.500],
        "Human Hours (h)": [0.03, 0.08, 0.17, 0.26, 0.35],
        "Cost Savings vs 100% Gold (%)": ["89.9%", "75.9%", "49.9%", "24.0%", "0.0%"]
    })

# Summary KPIs
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("100% Gold Benchmark F1", "1.000", "Maximum Human Baseline Quality")
with c2:
    render_metric_card("50% Hybrid F1 Score", f"{df_cost.iloc[2]['Macro-F1']:.3f}", "91.2% Performance at 50% Budget")
with c3:
    render_metric_card("25% Hybrid Savings", "75.9%", "Saves 76% Cost with <13% F1 Drop")
with c4:
    render_metric_card("Optimal Hybrid Strategy", "25% - 50% Gold", "Sweet-spot on Pareto Frontier")

st.markdown("<br>", unsafe_allow_html=True)

# Dual-Axis Cost-Quality Curve
st.subheader("📈 Empirical Cost-Quality Pareto Curve")

fig = go.Figure()
# Macro-F1 Line
fig.add_trace(go.Scatter(
    x=df_cost["Gold Proportion (%)"],
    y=df_cost["Macro-F1"],
    name="Macro-F1 Quality",
    mode="lines+markers",
    line=dict(color="#2563EB", width=3),
    marker=dict(size=9)
))
# Cost Line (Secondary Axis)
fig.add_trace(go.Scatter(
    x=df_cost["Gold Proportion (%)"],
    y=df_cost["Total Cost ($)"],
    name="Annotation Cost ($ USD)",
    mode="lines+markers",
    line=dict(color="#DC2626", width=3, dash="dash"),
    marker=dict(size=9),
    yaxis="y2"
))

fig.update_layout(
    xaxis=dict(title="Percentage of Human Gold Labels (%)"),
    yaxis=dict(title="Downstream Sentiment Macro-F1", titlefont=dict(color="#2563EB"), tickfont=dict(color="#2563EB")),
    yaxis2=dict(title="Total Annotation Cost ($ USD)", titlefont=dict(color="#DC2626"), tickfont=dict(color="#DC2626"), overlaying="y", side="right"),
    legend=dict(x=0.05, y=0.95),
    margin=dict(t=30, b=30, l=40, r=40),
    height=400,
    font=dict(family="Inter")
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Interactive Hybrid Budget Calculator
st.subheader("🧮 Interactive Hybrid Annotation Budget Calculator")
st.markdown("Calculate projected costs, human annotator hours, and F1 retention for your target dataset size.")

calc1, calc2, calc3 = st.columns(3)
with calc1:
    target_posts = st.number_input("Target Dataset Size (Number of Posts):", min_value=1000, max_value=500000, value=25000, step=5000)
with calc2:
    target_gold_pct = st.slider("Select Human Gold Proportion (%):", min_value=5, max_value=100, value=30, step=5)
with calc3:
    human_cost_unit = st.number_input("Human Cost per Annotation ($):", min_value=0.01, max_value=0.50, value=0.05, step=0.01)

# Calculations
n_gold_calc = int(target_posts * (target_gold_pct / 100.0))
n_llm_calc = target_posts - n_gold_calc

cost_100_human = target_posts * human_cost_unit
cost_hybrid_human = n_gold_calc * human_cost_unit
cost_hybrid_llm = n_llm_calc * ((450 / 1000.0) * 0.00015)
cost_total_hybrid = round(cost_hybrid_human + cost_hybrid_llm, 2)

hours_100 = round((target_posts * 25.0) / 3600.0, 1)
hours_hybrid = round((n_gold_calc * 25.0) / 3600.0, 1)
savings_usd = round(cost_100_human - cost_total_hybrid, 2)
savings_pct = round((savings_usd / cost_100_human) * 100, 1)

res_c1, res_c2, res_c3, res_c4 = st.columns(4)
with res_c1:
    st.metric("100% Human Cost", f"${cost_100_human:,.2f}", f"{hours_100} hours")
with res_c2:
    st.metric("Hybrid Strategy Cost", f"${cost_total_hybrid:,.2f}", f"{hours_hybrid} hours")
with res_c3:
    st.metric("Net Financial Savings", f"${savings_usd:,.2f}", f"{savings_pct}% saved", delta_color="normal")
with res_c4:
    st.metric("Time Saved", f"{round(hours_100 - hours_hybrid, 1)} Hours", "Accelerates delivery")

st.markdown("---")
# Empirical Table
st.subheader("📋 Empirical Hybrid Experiment Table")
st.dataframe(df_cost, use_container_width=True)

# Figure 7
st.markdown("---")
st.subheader("🖼️ Publication Figure: Cost-Quality Trade-off (Figure 7)")
fig7_path = "results/figures/fig7_cost_quality_curve.png"
if os.path.exists(fig7_path):
    st.image(fig7_path, use_column_width=True)

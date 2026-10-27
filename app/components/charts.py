"""
Plotly Charting Utilities for Interactive Dashboard Analytics.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Aesthetic Color Palette
COLORS = {
    "primary": "#3B82F6",
    "secondary": "#8B5CF6",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "info": "#06B6D4",
    "dark": "#1E293B",
    "light": "#F8FAFC",
    "palette": ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#06B6D4"]
}


def create_pie_chart(data_dict: Dict[str, int], title: str, hole: float = 0.45) -> go.Figure:
    """Create an interactive donut / pie chart."""
    labels = list(data_dict.keys())
    values = list(data_dict.values())

    fig = go.Figure(data=[go.Pie(
        labels=[str(l).title() for l in labels],
        values=values,
        hole=hole,
        marker=dict(colors=COLORS["palette"][:len(labels)]),
        textinfo="label+percent",
        hoverinfo="label+value+percent",
        textfont=dict(family="Inter", size=13),
    )])
    fig.update_layout(
        title=dict(text=title, font=dict(family="Inter", size=15, color="#1E293B")),
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=320
    )
    return fig


def create_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color_col: Optional[str] = None,
    barmode: str = "group"
) -> go.Figure:
    """Create an interactive bar chart."""
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        barmode=barmode,
        color_discrete_sequence=COLORS["palette"]
    )
    fig.update_layout(
        font=dict(family="Inter"),
        title=dict(font=dict(size=15, color="#1E293B")),
        xaxis=dict(title=x_col.replace("_", " ").title()),
        yaxis=dict(title=y_col.replace("_", " ").title()),
        margin=dict(t=40, b=20, l=20, r=20),
        height=340
    )
    return fig


def create_heatmap(
    matrix_df: pd.DataFrame,
    title: str,
    color_scale: str = "Blues",
    annot: bool = True
) -> go.Figure:
    """Create an interactive confusion matrix or crosstab heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=matrix_df.values,
        x=[str(c).title() for c in matrix_df.columns],
        y=[str(i).title() for i in matrix_df.index],
        colorscale=color_scale,
        text=matrix_df.values if annot else None,
        texttemplate="%{text}" if annot else None,
        textfont=dict(family="Inter", size=14, color="black"),
        hoverongaps=False
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(family="Inter", size=15, color="#1E293B")),
        font=dict(family="Inter"),
        margin=dict(t=40, b=30, l=40, r=20),
        height=340
    )
    return fig


def create_radar_chart(categories: List[str], values_dict: Dict[str, List[float]], title: str) -> go.Figure:
    """Create an interactive multi-dimensional radar chart."""
    fig = go.Figure()
    
    for i, (name, vals) in enumerate(values_dict.items()):
        # Close radar loop
        r_vals = vals + [vals[0]]
        theta_cats = categories + [categories[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=theta_cats,
            fill='toself',
            name=name,
            line=dict(color=COLORS["palette"][i % len(COLORS["palette"])], width=2)
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.0]),
        ),
        showlegend=True,
        title=dict(text=title, font=dict(family="Inter", size=15)),
        margin=dict(t=40, b=30, l=30, r=30),
        height=360
    )
    return fig

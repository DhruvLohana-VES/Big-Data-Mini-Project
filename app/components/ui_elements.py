"""
UI Styling and Custom Component Utilities for Streamlit Application.
"""

import streamlit as st


def inject_custom_css():
    """Inject custom modern CSS styles."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .main-header {
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 50%, #6366F1 100%);
            padding: 2.2rem 2.5rem;
            border-radius: 16px;
            color: #ffffff;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.25);
        }

        .main-header h1 {
            font-size: 2.1rem;
            font-weight: 800;
            margin: 0;
            color: #ffffff !important;
            letter-spacing: -0.02em;
        }

        .main-header p {
            font-size: 1.05rem;
            color: #E0E7FF;
            margin-top: 0.5rem;
            margin-bottom: 0;
            font-weight: 400;
        }

        .metric-card {
            background: #ffffff;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.4rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            height: 100%;
        }

        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
            border-color: #CBD5E1;
        }

        .metric-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }

        .metric-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.2;
        }

        .metric-subtext {
            font-size: 0.82rem;
            color: #64748B;
            margin-top: 0.35rem;
        }

        .badge {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 0.35rem;
        }

        .badge-positive { background-color: #DCFCE7; color: #166534; }
        .badge-negative { background-color: #FEE2E2; color: #991B1B; }
        .badge-neutral { background-color: #F1F5F9; color: #475569; }
        .badge-toxic { background-color: #FEE2E2; color: #B91C1C; }
        .badge-nontoxic { background-color: #E0E7FF; color: #3730A3; }
        .badge-hinglish { background-color: #FEF3C7; color: #92400E; }
        .badge-marathi { background-color: #F3E8FF; color: #6B21A8; }
        
        .section-card {
            background: #ffffff;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
        }
    </style>
    """, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, badge_text: str = "Big Data & Social Media Analytics"):
    """Render a styled header block."""
    st.markdown(f"""
    <div class="main-header">
        <span class="badge" style="background-color: rgba(255, 255, 255, 0.2); color: #ffffff; margin-bottom: 0.75rem;">{badge_text}</span>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtext: str = "", delta: str = "", delta_color: str = "green"):
    """Render an aesthetic statistical metric card."""
    delta_html = f'<span style="color: {delta_color}; font-weight: 600; font-size: 0.85rem;">{delta}</span>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtext">{delta_html} {subtext}</div>
    </div>
    """, unsafe_allow_html=True)


def render_badge(label: str, badge_type: str = "neutral") -> str:
    """Return HTML string for a colored badge."""
    return f'<span class="badge badge-{badge_type}">{label}</span>'

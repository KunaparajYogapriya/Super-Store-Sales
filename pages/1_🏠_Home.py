"""
Home Page - Superstore Sales Intelligence.
"""

import os
import sys
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_loader import load_data, get_dataset_summary

st.set_page_config(page_title="Home | Superstore Intelligence", page_icon="🏠", layout="wide")

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Data
try:
    df = load_data()
    summary = get_dataset_summary(df)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Hero Header
st.markdown("""
    <div style="margin-bottom: 2rem; padding: 1.5rem 0;">
        <span class="badge badge-success" style="font-size: 0.85rem; padding: 0.3rem 0.8rem;">PROD READY ML SYSTEM</span>
        <h1 style="font-size: 2.8rem; font-weight: 800; color: #0f172a; margin-top: 0.5rem; margin-bottom: 0.5rem; letter-spacing: -0.03em;">
            Superstore Sales Intelligence
        </h1>
        <p style="font-size: 1.2rem; color: #64748b; margin-bottom: 0;">
            Machine Learning powered sales analytics, order revenue prediction, and time-series forecasting.
        </p>
    </div>
""", unsafe_allow_html=True)

# Quick Stat Badges
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Lifetime Sales</div>
            <div class="kpi-value">${summary['total_sales']:,.2f}</div>
            <div class="kpi-subtitle">Across {summary['num_rows']:,} Transactions</div>
        </div>
    """, unsafe_allow_html=True)

with col_s2:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Lifetime Profit</div>
            <div class="kpi-value">${summary['total_profit']:,.2f}</div>
            <div class="kpi-subtitle">Net Profitability</div>
        </div>
    """, unsafe_allow_html=True)

with col_s3:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Orders</div>
            <div class="kpi-value">{summary['total_orders']:,}</div>
            <div class="kpi-subtitle">Unique Customer Orders</div>
        </div>
    """, unsafe_allow_html=True)

with col_s4:
    profit_margin = (summary['total_profit'] / summary['total_sales']) * 100
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Overall Profit Margin</div>
            <div class="kpi-value">{profit_margin:.2f}%</div>
            <div class="kpi-subtitle">Overall Business Efficiency</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Three Concise Capability Cards
col_c1, col_c2, col_c3 = st.columns(3)

with col_c1:
    st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.2rem;">📊</div>
            <h3>Business Analytics</h3>
            <p>Interactive executive dashboards with dynamic multi-dimensional filters for Sales, Profitability, Product Sub-Categories, and Regional trends.</p>
        </div>
    """, unsafe_allow_html=True)

with col_c2:
    st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.2rem;">🤖</div>
            <h3>Machine Learning</h3>
            <p>Supervised ML regression models trained with Random Forest and XGBoost to predict transaction Sales and net Profit while strictly preventing data leakage.</p>
        </div>
    """, unsafe_allow_html=True)

with col_c3:
    st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.2rem;">📅</div>
            <h3>Sales Forecasting</h3>
            <p>Time-series SARIMAX forecasting engine providing 3, 6, and 12-month future monthly sales projections complete with 80% confidence intervals.</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### System Capabilities Summary")
st.info("""
**Superstore Sales Intelligence** provides end-to-end analytical visibility and automated predictions for retail management.
Navigate through the sidebar pages to inspect historical business dashboards, run predictions on customized customer orders, or project future revenue demand.
""")

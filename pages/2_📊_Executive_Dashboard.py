"""
Executive Dashboard Page - Superstore Sales Intelligence.
"""

import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_loader import load_data
from visualization import plot_sales_profit_trend, plot_category_breakdown, plot_subcategory_performance, plot_regional_segment_matrix

st.set_page_config(page_title="Executive Dashboard | Superstore Intelligence", page_icon="📊", layout="wide")

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

st.markdown("""
    <div class="section-header">
        <h2>Executive Business Dashboard</h2>
        <p>Interactive high-level performance indicators and key business visual analytics.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Interactive Filters
st.sidebar.markdown("### 🔍 Global Filters")

# Date Filter
min_date = df['Order Date'].min().date()
max_date = df['Order Date'].max().date()
selected_date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Categorical Filters
all_regions = ["All"] + sorted(df['Region'].unique().tolist())
selected_region = st.sidebar.selectbox("Region", all_regions)

all_categories = ["All"] + sorted(df['Category'].unique().tolist())
selected_category = st.sidebar.selectbox("Category", all_categories)

all_segments = ["All"] + sorted(df['Segment'].unique().tolist())
selected_segment = st.sidebar.selectbox("Segment", all_segments)

all_ship_modes = ["All"] + sorted(df['Ship Mode'].unique().tolist())
selected_ship_mode = st.sidebar.selectbox("Ship Mode", all_ship_modes)

# Filter Application
filtered_df = df.copy()

if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
    start_d, end_d = selected_date_range
    filtered_df = filtered_df[
        (filtered_df['Order Date'].dt.date >= start_d) & 
        (filtered_df['Order Date'].dt.date <= end_d)
    ]

if selected_region != "All":
    filtered_df = filtered_df[filtered_df['Region'] == selected_region]

if selected_category != "All":
    filtered_df = filtered_df[filtered_df['Category'] == selected_category]

if selected_segment != "All":
    filtered_df = filtered_df[filtered_df['Segment'] == selected_segment]

if selected_ship_mode != "All":
    filtered_df = filtered_df[filtered_df['Ship Mode'] == selected_ship_mode]

if filtered_df.empty:
    st.warning("⚠️ No records match the selected filter criteria. Please adjust your filter selections.")
    st.stop()

# 5 Executive KPI Cards
t_sales = filtered_df['Sales'].sum()
t_profit = filtered_df['Profit'].sum()
t_orders = filtered_df['Order ID'].nunique()
t_qty = filtered_df['Quantity'].sum()
margin = (t_profit / t_sales * 100) if t_sales > 0 else 0.0

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Sales</div>
            <div class="kpi-value">${t_sales:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Profit</div>
            <div class="kpi-value" style="color: {'#10b981' if t_profit >= 0 else '#ef4444'};">${t_profit:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Orders</div>
            <div class="kpi-value">{t_orders:,}</div>
        </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Units</div>
            <div class="kpi-value">{t_qty:,}</div>
        </div>
    """, unsafe_allow_html=True)

with k5:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Profit Margin</div>
            <div class="kpi-value" style="color: {'#10b981' if margin >= 0 else '#ef4444'};">{margin:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Trend Chart
st.plotly_chart(plot_sales_profit_trend(filtered_df), use_container_width=True)

# Two Column Breakdown
col_b1, col_b2 = st.columns(2)

with col_b1:
    st.plotly_chart(plot_category_breakdown(filtered_df, metric="Sales"), use_container_width=True)

with col_b2:
    st.plotly_chart(plot_regional_segment_matrix(filtered_df), use_container_width=True)

st.plotly_chart(plot_subcategory_performance(filtered_df), use_container_width=True)

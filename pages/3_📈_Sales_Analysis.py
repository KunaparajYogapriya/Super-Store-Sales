"""
Sales Analysis Page - Superstore Sales Intelligence.
"""

import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_loader import load_data
from visualization import plot_state_choropleth, apply_custom_layout, PALETTE

st.set_page_config(page_title="Sales Analysis | Superstore Intelligence", page_icon="📈", layout="wide")

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

df = load_data()

st.markdown("""
    <div class="section-header">
        <h2>In-Depth Sales Analytics</h2>
        <p>Comprehensive evaluation of revenue streams across time, geography, product categories, and customer segments.</p>
    </div>
""", unsafe_allow_html=True)

# Yearly Sales Trend
st.subheader("1. Annual & Monthly Sales Performance")
c1, c2 = st.columns(2)

with c1:
    yearly_df = df.groupby('Year')['Sales'].sum().reset_index()
    fig_yr = px.bar(
        yearly_df,
        x='Year',
        y='Sales',
        text_auto='.3s',
        color_discrete_sequence=[PALETTE['accent']]
    )
    fig_yr.update_traces(textposition='outside')
    fig_yr = apply_custom_layout(fig_yr, title="Total Sales by Year", x_title="Year", y_title="Sales ($)")
    st.plotly_chart(fig_yr, use_container_width=True)

with c2:
    month_df = df.groupby('Month Name')['Sales'].sum().reindex([
        'January', 'February', 'March', 'April', 'May', 'June', 
        'July', 'August', 'September', 'October', 'November', 'December'
    ]).reset_index()
    fig_mo = px.line(
        month_df,
        x='Month Name',
        y='Sales',
        markers=True,
        color_discrete_sequence=[PALETTE['light_blue']]
    )
    fig_mo = apply_custom_layout(fig_mo, title="Aggregated Seasonality by Month", x_title="Month", y_title="Sales ($)")
    st.plotly_chart(fig_mo, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Category & Sub-Category
st.subheader("2. Product Category & Sub-Category Sales")
c3, c4 = st.columns(2)

with c3:
    cat_df = df.groupby('Category')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
    fig_cat = px.pie(
        cat_df,
        names='Category',
        values='Sales',
        hole=0.4,
        color_discrete_sequence=[PALETTE['accent'], PALETTE['emerald'], PALETTE['amber']]
    )
    fig_cat.update_layout(title="Sales Share by Category")
    st.plotly_chart(fig_cat, use_container_width=True)

with c4:
    sub_df = df.groupby('Sub-Category')['Sales'].sum().reset_index().sort_values('Sales', ascending=True)
    fig_sub = px.bar(
        sub_df,
        y='Sub-Category',
        x='Sales',
        orientation='h',
        text_auto='.2s',
        color_discrete_sequence=[PALETTE['accent']]
    )
    fig_sub = apply_custom_layout(fig_sub, title="Sales by Sub-Category", x_title="Sales ($)", y_title="Sub-Category")
    st.plotly_chart(fig_sub, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Geographic & Ship Mode Breakdown
st.subheader("3. Geographic & Logistics Distribution")
c5, c6 = st.columns(2)

with c5:
    top10_states = df.groupby('State')['Sales'].sum().nlargest(10).reset_index().sort_values('Sales', ascending=True)
    fig_st = px.bar(
        top10_states,
        y='State',
        x='Sales',
        orientation='h',
        text_auto='.2s',
        color_discrete_sequence=[PALETTE['primary']]
    )
    fig_st = apply_custom_layout(fig_st, title="Top 10 States by Sales Volume", x_title="Sales ($)", y_title="State")
    st.plotly_chart(fig_st, use_container_width=True)

with c6:
    ship_df = df.groupby('Ship Mode')['Sales'].sum().reset_index()
    fig_ship = px.bar(
        ship_df,
        x='Ship Mode',
        y='Sales',
        text_auto='.3s',
        color_discrete_sequence=[PALETTE['emerald']]
    )
    fig_ship = apply_custom_layout(fig_ship, title="Sales by Ship Mode", x_title="Ship Mode", y_title="Sales ($)")
    st.plotly_chart(fig_ship, use_container_width=True)

st.plotly_chart(plot_state_choropleth(df, metric="Sales"), use_container_width=True)

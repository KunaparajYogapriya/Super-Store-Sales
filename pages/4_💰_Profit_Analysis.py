"""
Profit Analysis Page - Superstore Sales Intelligence.
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_loader import load_data
from visualization import plot_discount_vs_profit, apply_custom_layout, PALETTE

st.set_page_config(page_title="Profit Analysis | Superstore Intelligence", page_icon="💰", layout="wide")

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

df = load_data()

st.markdown("""
    <div class="section-header">
        <h2>Profitability & Discount Impact Analysis</h2>
        <p>Analyze profit drivers, margin leaks, discount sensitivity, and loss-making product categories.</p>
    </div>
""", unsafe_allow_html=True)

# Profitability Summary Cards
t_sales = df['Sales'].sum()
t_profit = df['Profit'].sum()
overall_margin = (t_profit / t_sales) * 100

loss_orders = df[df['Profit'] < 0]
loss_total = loss_orders['Profit'].sum()
loss_count = len(loss_orders)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Net Profit</div>
            <div class="kpi-value" style="color:#10b981;">${t_profit:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Overall Profit Margin</div>
            <div class="kpi-value">{overall_margin:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Unprofitable Orders</div>
            <div class="kpi-value" style="color:#ef4444;">{loss_count:,}</div>
            <div class="kpi-subtitle negative">{loss_count/len(df)*100:.1f}% of total transactions</div>
        </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Accumulated Losses</div>
            <div class="kpi-value" style="color:#ef4444;">${loss_total:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Profit by Sub-Category (Identifying Losses)
st.subheader("1. Profitability by Product Sub-Category")
sub_profit = df.groupby(['Category', 'Sub-Category'])[['Sales', 'Profit']].sum().reset_index()
sub_profit['Profit Margin (%)'] = (sub_profit['Profit'] / sub_profit['Sales']) * 100
sub_profit = sub_profit.sort_values('Profit', ascending=False)

fig_sub_p = go.Figure()
fig_sub_p.add_trace(go.Bar(
    x=sub_profit['Sub-Category'],
    y=sub_profit['Profit'],
    marker_color=np.where(sub_profit['Profit'] >= 0, PALETTE['emerald'], PALETTE['crimson']),
    text=sub_profit['Profit'].apply(lambda x: f"${x:,.0f}"),
    textposition='outside'
))
fig_sub_p = apply_custom_layout(
    fig_sub_p, 
    title="Sub-Category Total Profit (Highlighting Loss-Making Sub-Categories in Red)",
    x_title="Sub-Category", 
    y_title="Total Profit ($)"
)
st.plotly_chart(fig_sub_p, use_container_width=True)

# Discount Impact Analysis
st.subheader("2. Discount vs. Profitability Analysis")
st.plotly_chart(plot_discount_vs_profit(df), use_container_width=True)

disc_summary = df.groupby('Discount')[['Profit', 'Sales', 'Quantity']].agg({
    'Profit': ['count', 'sum', 'mean'],
    'Sales': 'sum'
}).reset_index()
disc_summary.columns = ['Discount Level', 'Order Count', 'Total Profit', 'Avg Profit/Order', 'Total Sales']
disc_summary['Profit Margin %'] = (disc_summary['Total Profit'] / disc_summary['Total Sales']) * 100

col_d1, col_d2 = st.columns([3, 2])
with col_d1:
    fig_disc_m = px.bar(
        disc_summary,
        x='Discount Level',
        y='Avg Profit/Order',
        text_auto='.1f',
        color=np.where(disc_summary['Avg Profit/Order'] >= 0, 'Positive', 'Negative'),
        color_discrete_map={'Positive': PALETTE['emerald'], 'Negative': PALETTE['crimson']}
    )
    fig_disc_m = apply_custom_layout(fig_disc_m, title="Average Profit per Order by Discount Rate", x_title="Discount Rate", y_title="Avg Profit ($)")
    st.plotly_chart(fig_disc_m, use_container_width=True)

with col_d2:
    st.markdown("#### Discount Key Insights")
    st.warning("""
    - **Discounts > 20% severely erode profit margins**, resulting in consistent average net losses per transaction.
    - Product categories like **Tables** and **Bookcases** suffer heavy negative profit margins when discounts exceed 30%.
    - Recommended Strategy: Cap maximum promotional discount rates at **15%** for Office Supplies and Furniture.
    """)

st.markdown("<br>", unsafe_allow_html=True)

# Top & Bottom Products
st.subheader("3. Product Performance Breakdown")
tp_col1, tp_col2 = st.columns(2)

with tp_col1:
    st.markdown("#### 🟢 Top 5 Most Profitable Products")
    top_p = df.groupby('Product Name')[['Sales', 'Profit']].sum().nlargest(5, 'Profit').reset_index()
    top_p['Margin %'] = (top_p['Profit'] / top_p['Sales']) * 100
    st.dataframe(
        top_p.style.format({'Sales': '${:,.2f}', 'Profit': '${:,.2f}', 'Margin %': '{:.1f}%'}),
        use_container_width=True,
        hide_index=True
    )

with tp_col2:
    st.markdown("#### 🔴 Top 5 Biggest Loss-Making Products")
    bot_p = df.groupby('Product Name')[['Sales', 'Profit']].sum().nsmallest(5, 'Profit').reset_index()
    bot_p['Margin %'] = (bot_p['Profit'] / bot_p['Sales']) * 100
    st.dataframe(
        bot_p.style.format({'Sales': '${:,.2f}', 'Profit': '${:,.2f}', 'Margin %': '{:.1f}%'}),
        use_container_width=True,
        hide_index=True
    )

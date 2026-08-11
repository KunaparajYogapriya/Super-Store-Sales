"""
Sales Forecasting Page - Superstore Sales Intelligence.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from visualization import plot_forecasting_chart

st.set_page_config(page_title="Sales Forecasting | Superstore Intelligence", page_icon="📅", layout="wide")

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "forecasting_model.pkl")

@st.cache_resource
def load_forecasting_artifact():
    if os.path.exists(MODEL_PATH):
        try:
            artifact = joblib.load(MODEL_PATH)
            return artifact
        except Exception as e:
            st.error(f"Error loading forecasting model artifact: {e}")
            return None
    return None

artifact = load_forecasting_artifact()

st.markdown("""
    <div class="section-header">
        <h2>Monthly Sales Time-Series Forecasting</h2>
        <p>Project future monthly demand and revenue using SARIMAX time-series state-space modeling.</p>
    </div>
""", unsafe_allow_html=True)

if artifact is None:
    st.error("⚠️ Forecasting model is not trained yet. Please run `python src/train_forecasting_model.py` first.")
    st.stop()

model_fit = artifact['model_fit']
historical_ts = artifact['historical_ts']
eval_metrics = artifact['eval_metrics']

# Interactive Horizon Selector
st.markdown("### 🎯 Select Forecast Horizon")
horizon = st.radio(
    "Choose number of future months to project:",
    options=[3, 6, 12],
    index=2,
    horizontal=True
)

# Generate Forecast & Confidence Intervals
try:
    forecast_res = model_fit.get_forecast(steps=horizon)
    forecast_mean = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.2)  # 80% CI
    
    # Clip negative values if any
    forecast_mean = forecast_mean.clip(lower=0)
    
    forecast_df = pd.DataFrame({
        "forecast_sales": forecast_mean,
        "lower_bound": conf_int.iloc[:, 0].clip(lower=0),
        "upper_bound": conf_int.iloc[:, 1]
    })
    
    # Summary Metrics
    total_forecast = forecast_df['forecast_sales'].sum()
    avg_monthly_forecast = forecast_df['forecast_sales'].mean()
    
    # Historical comparison window
    last_window_sales = historical_ts.tail(horizon).sum()
    growth_pct = ((total_forecast - last_window_sales) / last_window_sales) * 100 if last_window_sales > 0 else 0.0
    
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Forecast Period</div>
                <div class="kpi-value">{horizon} Months</div>
                <div class="kpi-subtitle">Next {horizon} Monthly Periods</div>
            </div>
        """, unsafe_allow_html=True)
        
    with f2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Expected Total Sales</div>
                <div class="kpi-value">${total_forecast:,.2f}</div>
                <div class="kpi-subtitle">Projected Cumulative</div>
            </div>
        """, unsafe_allow_html=True)
        
    with f3:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Avg Monthly Sales</div>
                <div class="kpi-value">${avg_monthly_forecast:,.2f}</div>
                <div class="kpi-subtitle">Per Month Run-Rate</div>
            </div>
        """, unsafe_allow_html=True)
        
    with f4:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Projected Growth</div>
                <div class="kpi-value" style="color: {'#10b981' if growth_pct >= 0 else '#ef4444'};">{growth_pct:+.1f}%</div>
                <div class="kpi-subtitle">vs prior {horizon}-mo window</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Render Interactive Plotly Chart
    fig_fc = plot_forecasting_chart(historical_ts, forecast_df)
    st.plotly_chart(fig_fc, use_container_width=True)
    
    # Detailed Monthly Projection Table
    st.markdown("### 📋 Monthly Forecast Projections Table")
    table_df = forecast_df.reset_index()
    table_df.columns = ['Month Date', 'Projected Sales ($)', 'Lower Bound (80% CI)', 'Upper Bound (80% CI)']
    table_df['Month Date'] = table_df['Month Date'].dt.strftime('%B %Y')
    st.dataframe(
        table_df.style.format({
            'Projected Sales ($)': '${:,.2f}',
            'Lower Bound (80% CI)': '${:,.2f}',
            'Upper Bound (80% CI)': '${:,.2f}'
        }),
        use_container_width=True,
        hide_index=True
    )
    
except Exception as e:
    st.error(f"Error generating forecast: {e}")

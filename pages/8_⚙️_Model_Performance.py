"""
Model Performance & Diagnostics Page - Superstore Sales Intelligence.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_loader import load_data
from feature_engineering import get_sales_data, get_profit_data
from visualization import plot_actual_vs_predicted, plot_residuals, apply_custom_layout, PALETTE

st.set_page_config(page_title="Model Performance | Superstore Intelligence", page_icon="⚙️", layout="wide")

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "metrics_summary.json")
SALES_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "sales_model.pkl")
PROFIT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "profit_model.pkl")

st.markdown("""
    <div class="section-header">
        <h2>Machine Learning Model Performance & Evaluation</h2>
        <p>Comparative model benchmarking, cross-validation metrics, diagnostic residual analysis, and feature importances.</p>
    </div>
""", unsafe_allow_html=True)

# Load metrics summary
if os.path.exists(SUMMARY_PATH):
    with open(SUMMARY_PATH, "r") as f:
        summary_data = json.load(f)
else:
    summary_data = {}

# 1. Sales Models Comparison
st.subheader("1. Sales Prediction Models Benchmarking")
if "sales_model" in summary_data:
    sales_info = summary_data["sales_model"]
    st.markdown(f"**Selected Champion Model:** `<span class='badge badge-success'>{sales_info['best_model']}</span>`", unsafe_allow_html=True)
    sales_comp_df = pd.DataFrame(sales_info["comparison"])
    sales_comp_df.columns = ['Model Name', 'MAE ($)', 'MSE', 'RMSE ($)', 'R² Score']
    st.dataframe(
        sales_comp_df.style.format({
            'MAE ($)': '${:,.2f}',
            'MSE': '{:,.2f}',
            'RMSE ($)': '${:,.2f}',
            'R² Score': '{:.4f}'
        }).highlight_max(subset=['R² Score'], color='#d1fae5'),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Sales model metrics not available.")

st.markdown("<br>", unsafe_allow_html=True)

# 2. Profit Models Comparison
st.subheader("2. Profit Prediction Models Benchmarking")
if "profit_model" in summary_data:
    profit_info = summary_data["profit_model"]
    st.markdown(f"**Selected Champion Model:** `<span class='badge badge-success'>{profit_info['best_model']}</span>`", unsafe_allow_html=True)
    profit_comp_df = pd.DataFrame(profit_info["comparison"])
    profit_comp_df.columns = ['Model Name', 'MAE ($)', 'MSE', 'RMSE ($)', 'R² Score']
    st.dataframe(
        profit_comp_df.style.format({
            'MAE ($)': '${:,.2f}',
            'MSE': '{:,.2f}',
            'RMSE ($)': '${:,.2f}',
            'R² Score': '{:.4f}'
        }).highlight_max(subset=['R² Score'], color='#d1fae5'),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Profit model metrics not available.")

st.markdown("<br>", unsafe_allow_html=True)

# 3. Time Series Forecasting Metrics
st.subheader("3. Monthly Sales Forecasting Metrics")
if "forecasting_model" in summary_data:
    fc_info = summary_data["forecasting_model"]
    fc_metrics = fc_info["metrics"]
    fm1, fm2, fm3 = st.columns(3)
    with fm1:
        st.metric("Out-of-Sample MAE", f"${fc_metrics['MAE']:,.2f}")
    with fm2:
        st.metric("Out-of-Sample RMSE", f"${fc_metrics['RMSE']:,.2f}")
    with fm3:
        st.metric("Out-of-Sample MAPE", f"{fc_metrics['MAPE']:.2f}%")

st.markdown("<br>", unsafe_allow_html=True)

# 4. Diagnostic Charts
st.subheader("4. Model Diagnostic Plots & Feature Importances")
tab1, tab2, tab3 = st.tabs(["Sales Model Diagnostics", "Profit Model Diagnostics", "Feature Importances"])

with tab1:
    if os.path.exists(SALES_MODEL_PATH):
        sales_art = joblib.load(SALES_MODEL_PATH)
        df_raw = load_data()
        X_s, y_s, _, _ = get_sales_data(df_raw)
        _, X_test_s, _, y_test_s = train_test_split(X_s, y_s, test_size=0.2, random_state=42)
        y_pred_s = sales_art['pipeline'].predict(X_test_s)
        
        c_diag1, c_diag2 = st.columns(2)
        with c_diag1:
            st.plotly_chart(plot_actual_vs_predicted(y_test_s.values, y_pred_s, title="Sales: Actual vs Predicted"), use_container_width=True)
        with c_diag2:
            st.plotly_chart(plot_residuals(y_test_s.values, y_pred_s), use_container_width=True)

with tab2:
    if os.path.exists(PROFIT_MODEL_PATH):
        profit_art = joblib.load(PROFIT_MODEL_PATH)
        df_raw = load_data()
        X_p, y_p, _, _ = get_profit_data(df_raw)
        _, X_test_p, _, y_test_p = train_test_split(X_p, y_p, test_size=0.2, random_state=42)
        y_pred_p = profit_art['pipeline'].predict(X_test_p)
        
        c_pdiag1, c_pdiag2 = st.columns(2)
        with c_pdiag1:
            st.plotly_chart(plot_actual_vs_predicted(y_test_p.values, y_pred_p, title="Profit: Actual vs Predicted"), use_container_width=True)
        with c_pdiag2:
            st.plotly_chart(plot_residuals(y_test_p.values, y_pred_p), use_container_width=True)

with tab3:
    st.markdown("#### Feature Importance Analysis")
    if os.path.exists(PROFIT_MODEL_PATH):
        p_art = joblib.load(PROFIT_MODEL_PATH)
        pipe = p_art['pipeline']
        reg = pipe.named_steps['regressor']
        prep = pipe.named_steps['preprocessor']
        
        if hasattr(reg, 'feature_importances_'):
            try:
                # Extract OneHotEncoder feature names
                ohe = prep.named_transformers_['cat'].named_steps['encoder']
                cat_names = list(ohe.get_feature_names_out(p_art['cat_cols']))
                all_feature_names = p_art['num_cols'] + cat_names
                
                imp = reg.feature_importances_
                imp_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': imp}).sort_values('Importance', ascending=False).head(15)
                
                fig_imp = px.bar(
                    imp_df,
                    y='Feature',
                    x='Importance',
                    orientation='h',
                    color_discrete_sequence=[PALETTE['accent']]
                )
                fig_imp = apply_custom_layout(fig_imp, title="Top 15 Most Important Features (Profit XGBoost Model)", x_title="Importance Weight", y_title="Feature")
                st.plotly_chart(fig_imp, use_container_width=True)
            except Exception as e:
                st.info(f"Could not extract feature importances: {e}")
        else:
            st.info("Winning model does not expose feature importances.")

"""
Profit Prediction Page - Superstore Sales Intelligence.
"""

import os
import sys
import datetime
import pandas as pd
import streamlit as st
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_loader import load_data

st.set_page_config(page_title="Profit Prediction | Superstore Intelligence", page_icon="💵", layout="wide")

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Saved Profit Model Artifact
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "profit_model.pkl")

@st.cache_resource
def load_profit_pipeline():
    if os.path.exists(MODEL_PATH):
        try:
            artifact = joblib.load(MODEL_PATH)
            return artifact
        except Exception as e:
            st.error(f"Error loading profit model artifact: {e}")
            return None
    return None

artifact = load_profit_pipeline()
df = load_data()

st.markdown("""
    <div class="section-header">
        <h2>Order Profitability Predictor</h2>
        <p>Predict net profit and margin outcome for transaction configurations using ML regression.</p>
    </div>
""", unsafe_allow_html=True)

if artifact is None:
    st.error("⚠️ Profit model is not trained yet. Please run `python src/train_profit_model.py` first.")
    st.stop()

pipeline = artifact['pipeline']
best_model_name = artifact['best_model_name']
metrics = artifact['metrics']

st.caption(f"**Active Prediction Model:** {best_model_name} (R² = {metrics['R2']:.4f}, RMSE = ${metrics['RMSE']:.2f})")

with st.form("profit_prediction_form"):
    st.markdown("### Transaction Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sales_input = st.number_input("Order Total Sales ($)", min_value=0.01, max_value=25000.0, value=250.0, step=10.0)
        category = st.selectbox("Category", sorted(df['Category'].unique()))
        sub_cats = sorted(df[df['Category'] == category]['Sub-Category'].unique())
        sub_category = st.selectbox("Sub-Category", sub_cats)
        
    with col2:
        region = st.selectbox("Region", sorted(df['Region'].unique()))
        segment = st.selectbox("Segment", sorted(df['Segment'].unique()))
        ship_mode = st.selectbox("Ship Mode", sorted(df['Ship Mode'].unique()))
        
    with col3:
        state = st.selectbox("State", sorted(df['State'].unique()))
        quantity = st.number_input("Item Quantity", min_value=1, max_value=100, value=3, step=1)
        discount = st.slider("Applied Discount Rate", min_value=0.0, max_value=0.8, value=0.15, step=0.05, format="%.2f")
        shipping_days = st.number_input("Shipping Days", min_value=0, max_value=14, value=3, step=1)
        
    st.markdown("### Date Specification")
    d_col1, d_col2, d_col3 = st.columns(3)
    order_date = d_col1.date_input("Order Date", value=datetime.date.today())
    year = order_date.year
    month = order_date.month
    quarter = (month - 1) // 3 + 1
    
    d_col2.text_input("Year / Month", value=f"{year} / Month {month}", disabled=True)
    d_col3.text_input("Quarter", value=f"Q{quarter}", disabled=True)
    
    submit_btn = st.form_submit_button("💵 Predict Net Profit", use_container_width=True)

if submit_btn:
    input_data = pd.DataFrame([{
        'Sales': sales_input,
        'Quantity': quantity,
        'Discount': discount,
        'Shipping Days': shipping_days,
        'Year': year,
        'Month': month,
        'Quarter': quarter,
        'Category': category,
        'Sub-Category': sub_category,
        'Region': region,
        'Segment': segment,
        'Ship Mode': ship_mode,
        'State': state
    }])
    
    try:
        pred_profit = float(pipeline.predict(input_data)[0])
        pred_margin = (pred_profit / sales_input) * 100 if sales_input > 0 else 0.0
        
        status_class = "profit-positive" if pred_profit >= 0 else "profit-negative"
        status_text = "PROFITABLE TRANSACTION" if pred_profit >= 0 else "LOSS-MAKING TRANSACTION"
        
        st.markdown(f"""
            <div class="prediction-box {status_class}">
                <div class="prediction-label">{status_text}</div>
                <div class="prediction-amount">${pred_profit:,.2f}</div>
                <div class="prediction-subtext">
                    Estimated Profit Margin: <b>{pred_margin:.2f}%</b> on order sales of ${sales_input:,.2f}.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error executing profit prediction: {e}")

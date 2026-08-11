"""
Sales Prediction Page - Superstore Sales Intelligence.
"""

import os
import sys
import datetime
import pandas as pd
import streamlit as st
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_loader import load_data

st.set_page_config(page_title="Sales Prediction | Superstore Intelligence", page_icon="🔮", layout="wide")

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Saved Sales Model Artifact
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "sales_model.pkl")

@st.cache_resource
def load_sales_pipeline():
    if os.path.exists(MODEL_PATH):
        try:
            artifact = joblib.load(MODEL_PATH)
            return artifact
        except Exception as e:
            st.error(f"Error loading sales model artifact: {e}")
            return None
    return None

artifact = load_sales_pipeline()
df = load_data()

st.markdown("""
    <div class="section-header">
        <h2>Order Sales Value Predictor</h2>
        <p>Predict expected transaction sales revenue based on order specification parameters.</p>
    </div>
""", unsafe_allow_html=True)

if artifact is None:
    st.error("⚠️ Sales model is not trained yet. Please run `python src/train_sales_model.py` first.")
    st.stop()

pipeline = artifact['pipeline']
best_model_name = artifact['best_model_name']
metrics = artifact['metrics']

st.caption(f"**Active Prediction Model:** {best_model_name} (R² = {metrics['R2']:.4f}, RMSE = ${metrics['RMSE']:.2f})")

# Input Form
with st.form("sales_prediction_form"):
    st.markdown("### Order Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        category = st.selectbox("Category", sorted(df['Category'].unique()))
        # Filter sub-categories by chosen category
        sub_cats = sorted(df[df['Category'] == category]['Sub-Category'].unique())
        sub_category = st.selectbox("Sub-Category", sub_cats)
        region = st.selectbox("Region", sorted(df['Region'].unique()))
        
    with col2:
        segment = st.selectbox("Segment", sorted(df['Segment'].unique()))
        ship_mode = st.selectbox("Ship Mode", sorted(df['Ship Mode'].unique()))
        state = st.selectbox("State", sorted(df['State'].unique()))
        
    with col3:
        quantity = st.number_input("Item Quantity", min_value=1, max_value=100, value=3, step=1)
        discount = st.slider("Discount Rate", min_value=0.0, max_value=0.8, value=0.0, step=0.05, format="%.2f")
        shipping_days = st.number_input("Estimated Shipping Days", min_value=0, max_value=14, value=3, step=1)
        
    st.markdown("### Transaction Date Parameters")
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1:
        order_date = st.date_input("Order Date", value=datetime.date.today())
    with d_col2:
        year = order_date.year
        st.text_input("Year", value=str(year), disabled=True)
    with d_col3:
        month = order_date.month
        st.text_input("Month", value=str(month), disabled=True)
    with d_col4:
        quarter = (month - 1) // 3 + 1
        st.text_input("Quarter", value=f"Q{quarter}", disabled=True)
        
    submit_btn = st.form_submit_button("🔮 Predict Sales Revenue", use_container_width=True)

if submit_btn:
    # Build single-row DataFrame for prediction matching feature engineering names
    input_data = pd.DataFrame([{
        'Quantity': quantity,
        'Discount': discount,
        'Shipping Days': shipping_days,
        'Year': year,
        'Month': month,
        'Quarter': quarter,
        'Day': order_date.day,
        'DayOfWeek': order_date.weekday(),
        'Category': category,
        'Sub-Category': sub_category,
        'Region': region,
        'Segment': segment,
        'Ship Mode': ship_mode,
        'State': state
    }])
    
    try:
        pred_sales = pipeline.predict(input_data)[0]
        pred_sales_clean = max(0.0, float(pred_sales))
        
        st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-label">Estimated Transaction Sales Revenue</div>
                <div class="prediction-amount">${pred_sales_clean:,.2f}</div>
                <div class="prediction-subtext">
                    Based on selected order specifications, the predicted order sales value is <b>${pred_sales_clean:,.2f}</b>.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error executing sales prediction: {e}")

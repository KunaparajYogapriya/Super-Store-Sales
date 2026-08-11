"""
About Project Page - Superstore Sales Intelligence.
"""

import os
import sys
import streamlit as st

st.set_page_config(page_title="About Project | Superstore Intelligence", page_icon="ℹ️", layout="wide")

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
    <div class="section-header">
        <h2>About Superstore Sales Intelligence</h2>
        <p>End-to-End Enterprise Data Science, Machine Learning & Analytics System Architecture.</p>
    </div>
""", unsafe_allow_html=True)

col_a1, col_a2 = st.columns(2)

with col_a1:
    st.markdown("""
    ### 🎯 Project Objectives & Scoping
    The primary goal of **Superstore Sales Intelligence** is to transform historical retail order data into actionable operational insights and predictive intelligence.
    
    The system is built as a production-grade analytics suite featuring:
    1. **Data Validation & Preprocessing**: Automated schema enforcement, date parsing, and missing value checks.
    2. **Exploratory Business Analytics**: Dynamic executive dashboards with real-time multi-dimensional filters.
    3. **Leakage-Free Supervised Machine Learning**: Regression models predicting order sales and profit.
    4. **Time-Series Forecasting**: Monthly demand projection using SARIMAX state-space models.
    5. **Model Evaluation & Explainability**: Comprehensive benchmarking, residual analysis, and feature importances.
    """)

with col_a2:
    st.markdown("""
    ### 🛡️ Data Leakage Safeguards & Business Rationale
    To ensure true statistical validity and avoid target leakage:
    
    * **Sales Prediction Model**: Target is `Sales`. **`Profit` is strictly EXCLUDED** because in real-world retail transactions, net profit is a downstream calculated outcome of Sales. Using Profit to predict Sales causes target leakage.
    * **Profit Prediction Model**: Target is `Profit`. **`Sales` is INCLUDED** under the realistic operational assumption that when an order is placed at checkout, total sale price and item quantity are known before fulfillment.
    * **Time-Series Forecasting**: Aggregates Sales chronologically into monthly periods (`MS`) and uses explicit chronological train/test splits without random shuffling.
    """)

st.markdown("---")

st.markdown("### 💻 Technology Stack & Libraries")
t1, t2, t3, t4 = st.columns(4)
with t1:
    st.markdown("""
    **Core Engine & Pipeline**
    - Python 3.10+
    - Pandas & NumPy
    - Scikit-Learn
    """)
with t2:
    st.markdown("""
    **Machine Learning**
    - XGBoost Regressor
    - Random Forest
    - Gradient Boosting
    """)
with t3:
    st.markdown("""
    **Forecasting & Stats**
    - Statsmodels SARIMAX
    - Joblib Serialization
    - Pytest Suite
    """)
with t4:
    st.markdown("""
    **User Interface & Design**
    - Streamlit (Multipage)
    - Plotly Express & Graph Objects
    - Custom SaaS CSS
    """)

st.markdown("---")
st.markdown("### 📁 Project Architecture")
st.code("""
superstore_sales_ml/
│
├── app.py                            # Main Streamlit Router & Global Config
├── pages/                            # Multipage Modules (1_Home to 9_About)
├── data/                             # Dataset Storage (Superstore_Cleaned.csv)
├── models/                           # Pre-trained ML Artifacts (.pkl & JSON metrics)
├── src/                              # Core Processing Modules
│   ├── data_loader.py                # Validation & Loading Pipeline
│   ├── feature_engineering.py        # Date Derivation & Sklearn Preprocessors
│   ├── train_sales_model.py          # Sales Models Training Script
│   ├── train_profit_model.py          # Profit Models Training Script
│   ├── train_forecasting_model.py    # Time-Series SARIMAX Training Script
│   ├── evaluation.py                 # Evaluation Metrics Engine
│   └── visualization.py              # Custom Plotly SaaS Visualizer
├── tests/                            # Automated Pytest Suite
├── assets/                           # Stylesheets (styles.css)
├── requirements.txt                  # Python Dependencies
└── README.md                         # Technical Project Documentation
""", language="text")

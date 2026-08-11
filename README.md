# Superstore Sales Intelligence

An end-to-end, production-grade **Sales Analytics, Machine Learning Prediction, and Time-Series Sales Forecasting Application** built with Python, Scikit-Learn, XGBoost, Statsmodels, Plotly, and Streamlit.

---

## 📌 Project Overview & Objectives

**Superstore Sales Intelligence** provides enterprise retail executives and supply chain managers with actionable historical business analytics and predictive machine learning models. 

### Key Objectives:
1. **Data Validation & Preprocessing Pipeline**: Load, validate schema, handle temporal dimensions, and detect anomalies.
2. **Interactive Business Analytics**: Executive dashboards featuring real-time multi-dimensional filters (Date, Region, Category, Segment, Ship Mode).
3. **Leakage-Free Machine Learning Predictions**:
   - **Order Sales Predictor**: Predict expected transaction sales revenue.
   - **Order Profitability Predictor**: Predict net transaction profit and margin percentage.
4. **Time-Series Monthly Sales Forecasting**: SARIMAX state-space model projecting 3, 6, and 12-month future monthly sales demand with 80% confidence intervals.
5. **Technical Model Evaluation Dashboard**: Benchmarking comparison tables, residual error distribution plots, actual vs. predicted charts, and feature importances.

---

## 🛡️ Strict Data Leakage Rules & Business Rationale

Preventing target leakage is essential for valid production machine learning:

1. **Sales Prediction Model**:
   - **Target**: `Sales`
   - **Features**: `Quantity`, `Discount`, `Shipping Days`, `Category`, `Sub-Category`, `Region`, `Segment`, `Ship Mode`, `State`, `Year`, `Month`, `Quarter`, `Day`, `DayOfWeek`.
   - **Leakage Prevention**: **`Profit` is strictly EXCLUDED** because in real-world retail transactions, net profit is a downstream calculated output of Sales. Using Profit to predict Sales causes target leakage.

2. **Profit Prediction Model**:
   - **Target**: `Profit`
   - **Features**: `Sales`, `Quantity`, `Discount`, `Shipping Days`, `Category`, `Sub-Category`, `Region`, `Segment`, `Ship Mode`, `State`, `Year`, `Month`, `Quarter`.
   - **Business Rationale**: **`Sales` is INCLUDED** under the realistic operational scenario that when an order is placed at checkout, total sale price and item quantity are known before fulfillment.

---

## 📊 Model Evaluation Summary

All models were evaluated using an 80/20 train/test split (`random_state=42`) or out-of-sample chronological time-series splits.

### 1. Sales Prediction Models
| Model Name | MAE ($) | RMSE ($) | R² Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest Regressor** | **$209.48** | **$693.73** | **0.1853** | 🏆 Champion |
| Linear Regression | $228.48 | $694.10 | 0.1844 | Candidate |
| Gradient Boosting Regressor | $204.01 | $713.04 | 0.1393 | Candidate |
| XGBoost Regressor | $209.41 | $725.79 | 0.1082 | Candidate |

### 2. Profit Prediction Models
| Model Name | MAE ($) | RMSE ($) | R² Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost Regressor** | **$21.47** | **$85.19** | **0.8503** | 🏆 Champion |
| Gradient Boosting Regressor | $33.98 | $208.88 | 0.1001 | Candidate |
| Random Forest Regressor | $27.43 | $223.31 | -0.0285 | Candidate |
| Linear Regression | $71.35 | $284.53 | -0.6697 | Candidate |

### 3. Monthly Sales Forecasting Model (SARIMAX)
- **Model Specification**: `SARIMAX(1, 1, 1)(1, 1, 0, 12)`
- **12-Month Out-of-Sample MAE**: **$13,411.31**
- **12-Month Out-of-Sample RMSE**: **$15,989.43**
- **12-Month Out-of-Sample MAPE**: **26.75%**

---

## 🏗️ Project Architecture

```
superstore_sales_ml/
│
├── app.py                            # Streamlit Main Router & Entry Point
│
├── pages/                            # Multipage Modules
│   ├── 1_🏠_Home.py                  # Landing Page & Capability Showcase
│   ├── 2_📊_Executive_Dashboard.py   # Executive KPIs & Filtered Dashboard
│   ├── 3_📈_Sales_Analysis.py         # Sales Analytics & Regional Breakdown
│   ├── 4_💰_Profit_Analysis.py        # Profitability & Discount Sensitivity
│   ├── 5_🔮_Sales_Prediction.py       # Order Sales Predictor Form
│   ├── 6_💵_Profit_Prediction.py      # Order Profitability Predictor Form
│   ├── 7_📅_Sales_Forecasting.py      # SARIMAX Monthly Sales Horizon
│   ├── 8_⚙️_Model_Performance.py      # Evaluation Tables & Diagnostics
│   └── 9_ℹ️_About_Project.py          # System Architecture & Documentation
│
├── data/
│   └── Superstore_Cleaned.csv        # Cleaned Dataset
│
├── models/                           # Pre-Trained Serialized Models
│   ├── sales_model.pkl               # Random Forest Sales Pipeline
│   ├── profit_model.pkl              # XGBoost Profit Pipeline
│   ├── forecasting_model.pkl         # SARIMAX Forecasting Model
│   └── metrics_summary.json          # Exported Evaluation Benchmarks
│
├── src/                              # Python Logic Modules
│   ├── __init__.py
│   ├── data_loader.py                # Validation & Schema Checker
│   ├── feature_engineering.py        # Feature Transformers & Pipelines
│   ├── train_sales_model.py          # Sales Training Script
│   ├── train_profit_model.py          # Profit Training Script
│   ├── train_forecasting_model.py    # SARIMAX Forecasting Script
│   ├── train_all_models.py           # Master Script to Retrain All Models
│   ├── evaluation.py                 # Regression & Forecasting Metrics
│   └── visualization.py              # SaaS Plotly Visualization Library
│
├── tests/                            # Automated Pytest Suite
│   ├── test_data_loader.py           # Unit tests for schema validation
│   ├── test_feature_engineering.py   # Unit tests for feature matrix creation
│   └── test_models.py                # Unit tests for ML inference pipelines
│
├── assets/
│   └── styles.css                    # Custom SaaS Light Theme CSS
│
├── requirements.txt                  # Python Dependencies
└── README.md                         # Project Documentation
```

---

## ⚙️ Installation & Running the Application

### 1. Clone & Set Up Virtual Environment
```bash
git clone <repository-url>
cd superstore-sales-ml
python -m venv venv
# Activate environment:
# On Windows: venv\Scripts\activate
# On Linux/macOS: source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain ML Models
To retrain all regression and forecasting models from scratch:
```bash
python src/train_all_models.py
```

### 4. Run Automated Unit Tests
```bash
python -m pytest
```

### 5. Launch Streamlit Application
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your web browser.

---

## 🔮 Future Improvements

- **Neural Forecasting**: Benchmark Prophet and Deep Learning architectures (LSTM / N-BEATS) for long-horizon time series forecasting.
- **API Endpoint**: Wrap prediction pipelines with FastAPI REST endpoints for ERP integration.
- **Model Monitoring**: Implement Data Drift monitoring (Evidently AI) for incoming production order streams.

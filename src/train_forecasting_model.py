"""
Training & Forecasting Script for Monthly Sales Time-Series (Superstore Sales Intelligence).
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data_loader import load_data
from evaluation import evaluate_forecasting

warnings.filterwarnings('ignore')
MODELS_DIR = "models"


def prepare_monthly_series(df: pd.DataFrame) -> pd.Series:
    """Aggregate Sales by Month to form a clean regular time-series."""
    df_copy = df.copy()
    df_copy['Order Date'] = pd.to_datetime(df_copy['Order Date'])
    monthly_sales = df_copy.set_index('Order Date').resample('MS')['Sales'].sum()
    monthly_sales.index.freq = 'MS'
    return monthly_sales


def train_forecasting_model():
    """Train time-series SARIMA model on monthly sales and evaluate on chronological split."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Loading dataset for Sales Forecasting Model training...")
    df = load_data()
    
    ts = prepare_monthly_series(df)
    print(f"Monthly Time-Series created: {len(ts)} periods from {ts.index.min().strftime('%Y-%m')} to {ts.index.max().strftime('%Y-%m')}")
    
    # Chronological Split (Train: first 36 months, Test: last 12 months)
    test_size = 12
    train_ts = ts.iloc[:-test_size]
    test_ts = ts.iloc[-test_size:]
    
    print("\n--- Training SARIMAX Forecasting Model ---")
    # Fit SARIMAX model (p,d,q) x (P,D,Q,s)
    try:
        model = SARIMAX(train_ts, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
    except Exception as e:
        print(f"Fallback to simple ARIMA due to: {e}")
        model = SARIMAX(train_ts, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        
    test_preds = model_fit.forecast(steps=test_size)
    metrics = evaluate_forecasting(test_ts.values, test_preds.values)
    
    print(f"Forecasting Evaluation (12-Month Out-of-Sample):")
    print(f"MAE:  ${metrics['MAE']:,.2f}")
    print(f"RMSE: ${metrics['RMSE']:,.2f}")
    print(f"MAPE:  {metrics['MAPE']:.2f}%")
    
    # Fit final model on full historical time-series
    try:
        final_model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False)
        final_fit = final_model.fit(disp=False)
    except Exception:
        final_model = SARIMAX(ts, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
        final_fit = final_model.fit(disp=False)
        
    # Generate 12-month future forecast
    forecast_res = final_fit.get_forecast(steps=12)
    forecast_mean = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.2)  # 80% confidence interval
    
    forecast_df = pd.DataFrame({
        "forecast_sales": forecast_mean,
        "lower_bound": conf_int.iloc[:, 0],
        "upper_bound": conf_int.iloc[:, 1]
    })
    
    artifact = {
        "model_fit": final_fit,
        "historical_ts": ts,
        "eval_metrics": metrics,
        "last_date": ts.index.max().strftime('%Y-%m-%d')
    }
    
    model_path = os.path.join(MODELS_DIR, "forecasting_model.pkl")
    joblib.dump(artifact, model_path)
    print(f"Saved Forecasting model artifact to {model_path}")
    
    # Update metrics_summary.json
    summary_path = os.path.join(MODELS_DIR, "metrics_summary.json")
    summary_data = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary_data = json.load(f)
            
    summary_data["forecasting_model"] = {
        "model_name": "SARIMAX(1,1,1)(1,1,0,12)",
        "metrics": metrics,
        "total_historical_months": len(ts)
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4)
        
    return artifact


if __name__ == "__main__":
    train_forecasting_model()

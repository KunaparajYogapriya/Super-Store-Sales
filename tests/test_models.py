"""
Unit tests for Machine Learning prediction & time-series forecasting inference pipelines.
"""

import os
import sys
import pytest
import joblib
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))


def test_sales_model_inference():
    """Test loading pre-trained sales model and running single-sample inference."""
    sales_path = os.path.join(MODELS_DIR, "sales_model.pkl")
    assert os.path.exists(sales_path), "Sales model file missing!"
    
    artifact = joblib.load(sales_path)
    pipeline = artifact['pipeline']
    
    sample_input = pd.DataFrame([{
        'Quantity': 3,
        'Discount': 0.1,
        'Shipping Days': 3,
        'Year': 2024,
        'Month': 5,
        'Quarter': 2,
        'Day': 15,
        'DayOfWeek': 2,
        'Category': 'Furniture',
        'Sub-Category': 'Chairs',
        'Region': 'West',
        'Segment': 'Consumer',
        'Ship Mode': 'Standard Class',
        'State': 'California'
    }])
    
    prediction = pipeline.predict(sample_input)
    assert len(prediction) == 1
    assert prediction[0] >= 0


def test_profit_model_inference():
    """Test loading pre-trained profit model and running single-sample inference."""
    profit_path = os.path.join(MODELS_DIR, "profit_model.pkl")
    assert os.path.exists(profit_path), "Profit model file missing!"
    
    artifact = joblib.load(profit_path)
    pipeline = artifact['pipeline']
    
    sample_input = pd.DataFrame([{
        'Sales': 250.0,
        'Quantity': 3,
        'Discount': 0.15,
        'Shipping Days': 3,
        'Year': 2024,
        'Month': 5,
        'Quarter': 2,
        'Category': 'Technology',
        'Sub-Category': 'Phones',
        'Region': 'East',
        'Segment': 'Corporate',
        'Ship Mode': 'Second Class',
        'State': 'New York'
    }])
    
    prediction = pipeline.predict(sample_input)
    assert len(prediction) == 1
    assert isinstance(float(prediction[0]), float)


def test_forecasting_model_inference():
    """Test loading pre-trained SARIMAX forecasting model and generating steps."""
    fc_path = os.path.join(MODELS_DIR, "forecasting_model.pkl")
    assert os.path.exists(fc_path), "Forecasting model file missing!"
    
    artifact = joblib.load(fc_path)
    model_fit = artifact['model_fit']
    
    forecast_res = model_fit.get_forecast(steps=6)
    forecast_mean = forecast_res.predicted_mean
    assert len(forecast_mean) == 6

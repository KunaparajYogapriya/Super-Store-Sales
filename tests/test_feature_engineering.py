"""
Unit tests for Feature Engineering & Sklearn Preprocessing Pipelines.
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_loader import load_data
from feature_engineering import get_sales_data, get_profit_data, create_preprocessor


def test_get_sales_data_prevents_leakage():
    """Verify that Profit is strictly excluded from Sales prediction feature matrix."""
    df = load_data()
    X, y, num_cols, cat_cols = get_sales_data(df)
    
    assert 'Sales' not in X.columns
    assert 'Profit' not in X.columns, "DATA LEAKAGE WARNING: Profit found in Sales features!"
    assert len(X) == len(y)
    assert len(num_cols) > 0
    assert len(cat_cols) > 0


def test_get_profit_data():
    """Verify Profit feature matrix structure."""
    df = load_data()
    X, y, num_cols, cat_cols = get_profit_data(df)
    
    assert 'Profit' not in X.columns
    assert 'Sales' in X.columns
    assert len(X) == len(y)


def test_preprocessor_pipeline_transformation():
    """Test sklearn preprocessor transformation shape."""
    df = load_data()
    X, y, num_cols, cat_cols = get_sales_data(df)
    
    preprocessor = create_preprocessor(num_cols, cat_cols)
    X_transformed = preprocessor.fit_transform(X)
    
    assert X_transformed.shape[0] == len(df)
    assert X_transformed.shape[1] > len(num_cols) + len(cat_cols)  # due to OneHotEncoding

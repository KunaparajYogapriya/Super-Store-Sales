"""
Unit tests for Data Loader and Schema Validation module.
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_loader import load_data, get_dataset_summary, REQUIRED_COLUMNS


def test_load_data_success():
    """Test loading the Superstore cleaned dataset."""
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing required column: {col}"


def test_data_types_and_conversion():
    """Test that dates are properly parsed into datetime objects."""
    df = load_data()
    assert pd.api.types.is_datetime64_any_dtype(df['Order Date'])
    assert pd.api.types.is_datetime64_any_dtype(df['Ship Date'])
    assert 'Shipping Days' in df.columns
    assert 'Year' in df.columns
    assert 'Month' in df.columns


def test_get_dataset_summary():
    """Test summary generation statistics."""
    df = load_data()
    summary = get_dataset_summary(df)
    assert summary['num_rows'] == len(df)
    assert summary['total_sales'] > 0
    assert summary['total_orders'] > 0

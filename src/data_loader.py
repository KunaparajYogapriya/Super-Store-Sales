"""
Data Loader & Schema Validation Module for Superstore Sales Intelligence.
"""

import os
from typing import Dict, List, Tuple, Optional
import pandas as pd

REQUIRED_COLUMNS = [
    'Order ID', 'Order Date', 'Ship Date', 'Ship Mode', 'Customer ID',
    'Segment', 'City', 'State', 'Region', 'Category', 'Sub-Category',
    'Sales', 'Quantity', 'Discount', 'Profit'
]

DEFAULT_DATA_PATHS = [
    os.path.join("data", "Superstore_Cleaned.csv"),
    "Superstore_Cleaned.csv",
    os.path.join("data", "Sample - Superstore.csv"),
    "Sample - Superstore.csv"
]


def find_dataset_path(custom_path: Optional[str] = None) -> str:
    """Find a valid dataset path from custom argument or default paths."""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    
    for path in DEFAULT_DATA_PATHS:
        if os.path.exists(path):
            return path
            
    raise FileNotFoundError(
        "Superstore dataset file not found. Please ensure 'Superstore_Cleaned.csv' "
        "is present in the root or 'data/' directory."
    )


def load_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load Superstore dataset, validate schema, and convert data types.
    
    Args:
        filepath: Optional path to the CSV file.
        
    Returns:
        pd.DataFrame: Cleaned and validated DataFrame.
        
    Raises:
        ValueError: If dataset schema is invalid.
        FileNotFoundError: If dataset file is missing.
    """
    actual_path = find_dataset_path(filepath)
    df = pd.read_csv(actual_path)
    
    # Schema validation
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset schema invalid. Missing required columns: {missing_cols}"
        )
    
    # Date parsing
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    if 'Ship Date' in df.columns:
        df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')
        
    # Derived shipping days if missing or invalid
    if 'Shipping Days' not in df.columns and 'Order Date' in df.columns and 'Ship Date' in df.columns:
        df['Shipping Days'] = (df['Ship Date'] - df['Order Date']).dt.days.clip(lower=0)
        
    # Derived temporal features if missing
    if 'Year' not in df.columns and 'Order Date' in df.columns:
        df['Year'] = df['Order Date'].dt.year
    if 'Month' not in df.columns and 'Order Date' in df.columns:
        df['Month'] = df['Order Date'].dt.month
    if 'Month Name' not in df.columns and 'Order Date' in df.columns:
        df['Month Name'] = df['Order Date'].dt.month_name()
    if 'Quarter' not in df.columns and 'Order Date' in df.columns:
        df['Quarter'] = df['Order Date'].dt.quarter
        
    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict:
    """Return key metrics and data validation summary of the loaded dataset."""
    date_min = df['Order Date'].min() if 'Order Date' in df.columns else None
    date_max = df['Order Date'].max() if 'Order Date' in df.columns else None
    
    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "total_sales": df['Sales'].sum() if 'Sales' in df.columns else 0.0,
        "total_profit": df['Profit'].sum() if 'Profit' in df.columns else 0.0,
        "total_orders": df['Order ID'].nunique() if 'Order ID' in df.columns else len(df),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "start_date": str(date_min.strftime('%Y-%m-%d')) if pd.notnull(date_min) else "N/A",
        "end_date": str(date_max.strftime('%Y-%m-%d')) if pd.notnull(date_max) else "N/A",
    }

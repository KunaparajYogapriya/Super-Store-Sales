"""
Feature Engineering & Sklearn Preprocessing Pipelines for Superstore Sales Intelligence.
"""

from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

SALES_NUMERICAL_FEATURES = ['Quantity', 'Discount', 'Shipping Days', 'Year', 'Month', 'Quarter', 'Day', 'DayOfWeek']
SALES_CATEGORICAL_FEATURES = ['Category', 'Sub-Category', 'Region', 'Segment', 'Ship Mode', 'State']

PROFIT_NUMERICAL_FEATURES = ['Sales', 'Quantity', 'Discount', 'Shipping Days', 'Year', 'Month', 'Quarter']
PROFIT_CATEGORICAL_FEATURES = ['Category', 'Sub-Category', 'Region', 'Segment', 'Ship Mode', 'State']


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from Order Date and Ship Date."""
    df_out = df.copy()
    if 'Order Date' in df_out.columns:
        df_out['Order Date'] = pd.to_datetime(df_out['Order Date'])
        df_out['Year'] = df_out['Order Date'].dt.year
        df_out['Month'] = df_out['Order Date'].dt.month
        df_out['Quarter'] = df_out['Order Date'].dt.quarter
        df_out['Day'] = df_out['Order Date'].dt.day
        df_out['DayOfWeek'] = df_out['Order Date'].dt.dayofweek
        
    if 'Shipping Days' not in df_out.columns and 'Ship Date' in df_out.columns and 'Order Date' in df_out.columns:
        df_out['Ship Date'] = pd.to_datetime(df_out['Ship Date'])
        df_out['Shipping Days'] = (df_out['Ship Date'] - df_out['Order Date']).dt.days.clip(lower=0)
    elif 'Shipping Days' not in df_out.columns:
        df_out['Shipping Days'] = 3  # default median shipping days
        
    return df_out


def get_sales_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Prepare feature matrix X and target y for Sales Prediction.
    
    Data Leakage Rule: Profit is strictly EXCLUDED when predicting Sales.
    """
    df_proc = extract_temporal_features(df)
    
    num_cols = [c for c in SALES_NUMERICAL_FEATURES if c in df_proc.columns]
    cat_cols = [c for c in SALES_CATEGORICAL_FEATURES if c in df_proc.columns]
    
    X = df_proc[num_cols + cat_cols]
    y = df_proc['Sales']
    
    return X, y, num_cols, cat_cols


def get_profit_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Prepare feature matrix X and target y for Profit Prediction.
    
    Sales is included as a feature under the operational assumption that order value is known prior to fulfillment.
    """
    df_proc = extract_temporal_features(df)
    
    num_cols = [c for c in PROFIT_NUMERICAL_FEATURES if c in df_proc.columns]
    cat_cols = [c for c in PROFIT_CATEGORICAL_FEATURES if c in df_proc.columns]
    
    X = df_proc[num_cols + cat_cols]
    y = df_proc['Profit']
    
    return X, y, num_cols, cat_cols


def create_preprocessor(numerical_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Create a scikit-learn ColumnTransformer for numerical scaling and categorical encoding."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ]
    )
    
    return preprocessor

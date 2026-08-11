"""
Training Script for Profit Prediction Model (Superstore Sales Intelligence).
"""

import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from data_loader import load_data
from feature_engineering import get_profit_data, create_preprocessor
from evaluation import evaluate_regression

MODELS_DIR = "models"


def train_profit_models():
    """Train multiple regression models to predict Profit and select the best model."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Loading dataset for Profit Model training...")
    df = load_data()
    
    X, y, num_cols, cat_cols = get_profit_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = create_preprocessor(num_cols, cat_cols)
    
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    }
    
    results = []
    trained_pipelines = {}
    
    print("\n--- Profit Model Comparison ---")
    for name, model in candidate_models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        metrics = evaluate_regression(y_test, y_pred, model_name=name)
        results.append(metrics)
        trained_pipelines[name] = pipeline
        print(f"{name:30s} | MAE: {metrics['MAE']:8.2f} | RMSE: {metrics['RMSE']:8.2f} | R2: {metrics['R2']:6.4f}")
        
    # Select best model based on highest R2
    best_result = max(results, key=lambda x: x['R2'])
    best_model_name = best_result['model_name']
    best_pipeline = trained_pipelines[best_model_name]
    
    print(f"\nWinner: {best_model_name} (R2 = {best_result['R2']:.4f}, RMSE = {best_result['RMSE']:.2f})")
    
    # Save best model pipeline
    model_path = os.path.join(MODELS_DIR, "profit_model.pkl")
    artifact = {
        "pipeline": best_pipeline,
        "best_model_name": best_model_name,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "metrics": best_result,
        "all_metrics": results
    }
    joblib.dump(artifact, model_path)
    print(f"Saved best Profit model pipeline to {model_path}")
    
    # Update metrics_summary.json
    summary_path = os.path.join(MODELS_DIR, "metrics_summary.json")
    summary_data = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary_data = json.load(f)
            
    summary_data["profit_model"] = {
        "best_model": best_model_name,
        "metrics": best_result,
        "comparison": results
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4)
        
    return artifact


if __name__ == "__main__":
    train_profit_models()

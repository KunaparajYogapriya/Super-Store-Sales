"""
Master script to train all ML models (Sales, Profit, Forecasting).
"""

from train_sales_model import train_sales_models
from train_profit_model import train_profit_models
from train_forecasting_model import train_forecasting_model


def main():
    print("==================================================")
    print("    SUPERSTORE SALES INTELLIGENCE - MODEL TRAINING   ")
    print("==================================================")
    
    print("\n[1/3] Training Sales Prediction Model...")
    train_sales_models()
    
    print("\n[2/3] Training Profit Prediction Model...")
    train_profit_models()
    
    print("\n[3/3] Training Monthly Sales Forecasting Model...")
    train_forecasting_model()
    
    print("\n==================================================")
    print("    ALL MODELS TRAINED & SERIALIZED SUCCESSFULLY    ")
    print("==================================================")


if __name__ == "__main__":
    main()

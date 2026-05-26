"""
Amazon Smart Pricing Pipeline : This message framed using AI
-----------------------------
A multi-modal machine learning pipeline that extracts text features via SentenceTransformers
and image features via Vision Transformers (ViT) to predict product listing prices.
Supports robust ensembling with K-Fold cross-validation using Linear Regression and LightGBM.
"""

import numpy as np
import pandas as pd

# Central Configuration, Helprs, Fusion and Models Modules
import config
from utils import initialize_directories
from metrics import calculate_smape
from feature_fusion import prepare_features
from models import train_linear_regression, train_lightgbm, lgb



# 5. Main Execution Pipeline Entrpoint


def main():
    # Setup folders
    initialize_directories()
    
    print("[INFO] Loading datasets...")
    try:
        train_df = pd.read_csv(config.TRAIN_CSV)
        test_df = pd.read_csv(config.TEST_CSV)
    except FileNotFoundError as e:
        print(f"[ERROR] Could not load datasets. Please check file paths: {e}")
        return


    # Validate Schema Columns
    required_train_cols = ['sample_id', 'catalog_content', 'image_link', 'price']
    required_test_cols = ['sample_id', 'catalog_content', 'image_link']
    
    assert all(col in train_df.columns for col in required_train_cols), "Missing required columns in train.csv"
    assert all(col in test_df.columns for col in required_test_cols), "Missing required columns in test.csv"

    y_train = train_df['price'].values.astype(np.float32)


    # Multi-modal Feature Engineering (Defaults to Text modality )
    X_train, X_test = prepare_features(train_df, test_df, use_text=True, use_image=False)


    # 1. Train and Evaluate Scaled Linar Regression 
    print("\n--- Training Linear Regression Baseline ---")
    lr_model, lr_oof, lr_test, scaler = train_linear_regression(X_train, y_train, X_test)
    lr_smape = calculate_smape(y_train, lr_oof)
    print(f"Linear Regression Validation SMAPE: {lr_smape:.4f}")

    # 2. Train and Evaluate LightGBM Boosting Model
    if lgb is not None:
        print("\n--- Training LightGBM Model ---")
        lgb_model, lgb_oof, lgb_test = train_lightgbm(X_train, y_train, X_test)
        lgb_smape = calculate_smape(y_train, lgb_oof)
        print(f"LightGBM Validation SMAPE: {lgb_smape:.4f}")
    else:
        print("\n[INFO] LightGBM is not available. Skipping GBDT model training.")
        lgb_test = np.zeros_like(lr_test)


    # 3. Simple Predictions Ensemble (Blending)
    print("\n--- Blending Model Predictions ---")
    if lgb is not None:
        ensemble_predictions = (0.5 * lr_test) + (0.5 * lgb_test)
    else:
        ensemble_predictions = lr_test

    # 4. Generate Final Submission File
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': ensemble_predictions
    })
    submission_df.to_csv(config.SAMPLE_OUT_CSV, index=False)
    print(f"[INFO] Submission saved successfully to: {config.SAMPLE_OUT_CSV}")


if __name__ == '__main__':
    main()

"""
Main training and validation script for Freight Rate Prediction Challenge.
"""
from __future__ import annotations
import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_cleaner import FreightDataCleaner
from src.feature_engineering import FreightFeatureEngineer
from src.models import FreightRateModel, FreightEnsembleModel
from src.validation import ExpandingTimeSeriesCV, run_cross_validation
from src.pipeline import FreightRatePipeline
from src.utils import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Freight Rate Prediction ML Pipeline")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument("--output-dir", default="artifacts", help="Path to save artifacts")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FREIGHT RATE PREDICTION CHALLENGE - ML TRAINING & VALIDATION PIPELINE")
    print("=" * 70)

    # 1. Load Datasets
    train_path = data_dir / "train_test.csv"
    val_path = data_dir / "validation.csv"

    print(f"\n[1/4] Loading development data from {train_path}...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path) if val_path.exists() else None
    
    print(f"Loaded {len(train_df):,} training records with {train_df.shape[1]} raw columns.")
    if val_df is not None:
        print(f"Loaded {len(val_df):,} validation records for spatial lookup.")

    # 2. Preprocessing & Feature Engineering
    print("\n[2/4] Executing Data Cleaning & Domain Feature Engineering...")
    cleaner = FreightDataCleaner()
    cleaner.fit(train_df)
    if val_df is not None:
        cleaner.update_coords_from_validation(val_df)
        
    clean_train = cleaner.transform(train_df)
    
    fe = FreightFeatureEngineer()
    fe.fit(clean_train)
    feat_train = fe.transform(clean_train)
    feature_cols = fe.get_feature_names()
    
    print(f"Successfully generated {len(feature_cols)} engineered features:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:02d}. {col}")

    # 3. Temporal Cross-Validation
    print("\n[3/4] Evaluating Models via Expanding-Window Time Series Cross-Validation...")
    
    model_types = {
        'LightGBM (RPM)': (FreightRateModel, {'model_type': 'lightgbm', 'target_mode': 'rpm'}),
        'CatBoost (RPM)': (FreightRateModel, {'model_type': 'catboost', 'target_mode': 'rpm'}),
        'XGBoost (RPM)': (FreightRateModel, {'model_type': 'xgboost', 'target_mode': 'rpm'}),
        'Weighted Ensemble': (FreightEnsembleModel, {'weights': {'lightgbm': 0.40, 'catboost': 0.40, 'xgboost': 0.20}})
    }

    cv_results = {}
    for name, (model_cls, kwargs) in model_types.items():
        print(f"\n>>> Running 5-Fold Temporal CV for: {name}...")
        summary, fold_df = run_cross_validation(feat_train, feature_cols, model_cls, kwargs)
        cv_results[name] = {
            'summary': summary,
            'folds': fold_df.to_dict(orient='records')
        }
        print(f"  Mean MAE: ${summary['Mean_MAE']:.2f} (+/- ${summary['Std_MAE']:.2f})")
        print(f"  Mean RMSE: ${summary['Mean_RMSE']:.2f} | R²: {summary['Mean_R2']:.4f} | MAPE: {summary['Mean_MAPE']*100:.2f}%")

    # Save CV Results
    cv_path = output_dir / "cross_validation_results.json"
    with open(cv_path, 'w') as f:
        json.dump(cv_results, f, indent=2)
    print(f"\nSaved cross-validation results to {cv_path}")

    # 4. Final Full-Dataset Pipeline Training
    print("\n[4/4] Training Final Production Ensemble on all 48,000 Labeled Records...")
    pipeline = FreightRatePipeline(ensemble_weights={'lightgbm': 0.40, 'catboost': 0.40, 'xgboost': 0.20})
    pipeline.fit(train_df, val_df_for_coords=val_df)
    print("Production Ensemble trained successfully!")

    # Feature Importance Analysis from LightGBM model in ensemble
    lgb_fi = pipeline.ensemble.models['lightgbm'].get_feature_importance()
    fi_path = output_dir / "feature_importance.csv"
    lgb_fi.to_csv(fi_path, index=False)
    print(f"Saved feature importance ranking to {fi_path}")
    print("\nTop 15 Most Influential Features:")
    print(lgb_fi.head(15).to_string(index=False))

    print("\n" + "=" * 70)
    print("TRAINING & VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

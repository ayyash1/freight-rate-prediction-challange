"""
End-to-end training, inference, and submission pipeline for Freight Rate ML.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from src.data_cleaner import FreightDataCleaner
from src.feature_engineering import FreightFeatureEngineer
from src.models import FreightEnsembleModel, FreightRateModel
from src.validation import ExpandingTimeSeriesCV, run_cross_validation
from src.utils import compute_metrics


class FreightRatePipeline:
    """
    Production end-to-end pipeline:
    1. Robust data cleaning & anomaly correction
    2. Domain feature engineering (spatial, cargo, market signals, temporal)
    3. Multi-model ensemble training (LightGBM + CatBoost + XGBoost)
    4. Validation scoring & December benchmark generation
    """
    def __init__(self, ensemble_weights: Optional[Dict[str, float]] = None):
        self.cleaner = FreightDataCleaner()
        self.feature_engineer = FreightFeatureEngineer()
        self.ensemble = FreightEnsembleModel(weights=ensemble_weights)
        self.feature_names = []
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame, val_df_for_coords: Optional[pd.DataFrame] = None) -> "FreightRatePipeline":
        """Fit preprocessing, feature pipelines, and train the model ensemble on full labeled data."""
        # 1. Fit cleaner
        self.cleaner.fit(train_df)
        if val_df_for_coords is not None:
            self.cleaner.update_coords_from_validation(val_df_for_coords)
            
        clean_train = self.cleaner.transform(train_df)
        
        # 2. Fit feature engineer
        self.feature_engineer.fit(clean_train)
        feat_train = self.feature_engineer.transform(clean_train)
        self.feature_names = self.feature_engineer.get_feature_names()
        
        # 3. Train ensemble on full labeled dataset
        X = feat_train[self.feature_names]
        y = feat_train['posted_rate']
        dist = feat_train['distance']
        
        self.ensemble.fit(X, y, dist)
        self.is_fitted = True
        return self

    def predict_validation(self, val_df: pd.DataFrame, template_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate final predictions for validation loads (TE-000001 to TE-012000).
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before generating predictions.")
            
        clean_val = self.cleaner.transform(val_df)
        feat_val = self.feature_engineer.transform(clean_val)
        
        X_val = feat_val[self.feature_names]
        dist_val = feat_val['distance']
        
        preds = self.ensemble.predict(X_val, dist_val)
        
        pred_df = pd.DataFrame({
            'load_id': val_df['load_id'],
            'predicted_rate': np.round(preds, 2)
        })
        
        if template_df is not None:
            # Ensure order strictly matches template
            pred_df = template_df[['load_id']].merge(pred_df, on='load_id', how='left')
            
        return pred_df

    def predict_december_chart(self, dec_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate predictions for the 31 fixed December benchmark loads (Lexington to Fort Wayne).
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before generating predictions.")
            
        dec_data = dec_df.copy()
        dec_data['date'] = pd.to_datetime(dec_data['date'])
        
        # If val_df provided, extract December daily market_index and quote_signal medians
        if val_df is not None:
            val_copy = val_df.copy()
            val_copy['date'] = pd.to_datetime(val_copy['date'])
            val_dec = val_copy[val_copy['date'].dt.month == 12]
            
            daily_mi = val_dec.groupby('date')['market_index'].median().to_dict()
            daily_dryvan_qs = val_dec[val_dec['equipment'] == 'Dry Van'].groupby('date')['quote_signal'].median().to_dict()
            overall_qs = val_dec['quote_signal'].median()
            
            dec_data['market_index'] = dec_data['date'].map(daily_mi)
            dec_data['quote_signal'] = dec_data['date'].map(daily_dryvan_qs).fillna(overall_qs)
        else:
            dec_data['market_index'] = 1.0
            dec_data['quote_signal'] = 2.05
            
        clean_dec = self.cleaner.transform(dec_data)
        feat_dec = self.feature_engineer.transform(clean_dec)
        
        X_dec = feat_dec[self.feature_names]
        dist_dec = feat_dec['distance']
        
        dec_preds = self.ensemble.predict(X_dec, dist_dec)
        
        result_df = dec_df.copy()
        result_df['predicted_rate'] = np.round(dec_preds, 2)
        return result_df

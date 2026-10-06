"""
Cross-validation and temporal splitting strategies for Freight Rate ML.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Generator
from src.utils import compute_metrics


class ExpandingTimeSeriesCV:
    """
    Expanding-window time series cross-validator for temporal freight data.
    Ensures zero data leakage from future dates into historical models.
    """
    def __init__(self, min_train_months: int = 5):
        self.min_train_months = min_train_months

    def split(self, df: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray, int], None, None]:
        """
        Yields (train_indices, val_indices, test_month) for each expanding fold.
        """
        dates = pd.to_datetime(df['date'])
        months = dates.dt.month.values
        unique_months = sorted(np.unique(months))
        
        for val_month in unique_months:
            if val_month <= self.min_train_months:
                continue
            train_idx = np.where(months < val_month)[0]
            val_idx = np.where(months == val_month)[0]
            yield train_idx, val_idx, val_month


def run_cross_validation(df_feat: pd.DataFrame, feature_cols: List[str], 
                         model_class, model_kwargs: Dict[str, Any] = None) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Executes expanding window temporal cross validation and aggregates fold metrics.
    """
    model_kwargs = model_kwargs or {}
    cv = ExpandingTimeSeriesCV(min_train_months=5)
    fold_metrics = []
    oof_predictions = np.zeros(len(df_feat))
    
    y = df_feat['posted_rate']
    dist = df_feat['distance']
    
    for fold_idx, (tr_idx, val_idx, test_month) in enumerate(cv.split(df_feat), 1):
        X_tr = df_feat.iloc[tr_idx][feature_cols]
        y_tr = y.iloc[tr_idx]
        dist_tr = dist.iloc[tr_idx]
        
        X_val = df_feat.iloc[val_idx][feature_cols]
        y_val = y.iloc[val_idx]
        dist_val = dist.iloc[val_idx]
        
        model = model_class(**model_kwargs)
        model.fit(X_tr, y_tr, dist_tr, X_val, y_val, dist_val)
        
        preds_val = model.predict(X_val, dist_val)
        oof_predictions[val_idx] = preds_val
        
        metrics = compute_metrics(y_val, preds_val)
        metrics['fold'] = fold_idx
        metrics['test_month'] = int(test_month)
        metrics['train_samples'] = len(tr_idx)
        metrics['val_samples'] = len(val_idx)
        fold_metrics.append(metrics)
        
    metrics_df = pd.DataFrame(fold_metrics)
    overall_summary = {
        'Mean_MAE': float(metrics_df['MAE'].mean()),
        'Std_MAE': float(metrics_df['MAE'].std()),
        'Mean_RMSE': float(metrics_df['RMSE'].mean()),
        'Mean_R2': float(metrics_df['R2'].mean()),
        'Mean_MAPE': float(metrics_df['MAPE'].mean()),
        'Mean_Median_AE': float(metrics_df['Median_AE'].mean())
    }
    
    return overall_summary, metrics_df

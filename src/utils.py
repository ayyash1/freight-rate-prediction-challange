"""
Utility functions for spatial metrics, temporal math, and evaluation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


def haversine_distance(lat1: np.ndarray | pd.Series, 
                       lon1: np.ndarray | pd.Series, 
                       lat2: np.ndarray | pd.Series, 
                       lon2: np.ndarray | pd.Series) -> np.ndarray:
    """Calculate great-circle distance in miles between coordinate pairs using Haversine formula."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    miles = 3958.8 * c  # Earth radius in miles
    return miles


def initial_bearing(lat1: np.ndarray | pd.Series, 
                    lon1: np.ndarray | pd.Series, 
                    lat2: np.ndarray | pd.Series, 
                    lon2: np.ndarray | pd.Series) -> np.ndarray:
    """Calculate forward azimuth / bearing in degrees (0 - 360) from origin to destination."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(y, x))
    return (bearing + 360.0) % 360.0


def compute_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    """Compute comprehensive regression metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    med_ae = float(np.median(np.abs(y_true - y_pred)))
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'Median_AE': med_ae
    }

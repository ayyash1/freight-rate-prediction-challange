"""
Data cleaning and validation module for Freight Rate ML pipeline.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


class FreightDataCleaner:
    """
    Cleans raw freight transaction records:
    - Inverts sign on corrupt negative weights
    - Imputes missing weights based on equipment category medians
    - Imputes missing market indices from daily time-series medians
    - Maintains master city coordinates catalog
    """
    def __init__(self):
        self.equip_median_weights: Dict[str, float] = {}
        self.global_median_weight: float = 31500.0
        self.daily_market_indices: Dict[pd.Timestamp, float] = {}
        self.global_median_market_index: float = 1.0
        self.city_pickup_coords: Dict[str, Dict[str, float]] = {}
        self.city_delivery_coords: Dict[str, Dict[str, float]] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "FreightDataCleaner":
        """Learn statistical properties and coordinate mappings from development data."""
        data = df.copy()
        data['date'] = pd.to_datetime(data['date'])
        
        # 1. Weight statistics (after sign correction)
        clean_weights = data['weight'].abs()
        self.global_median_weight = float(clean_weights.median())
        self.equip_median_weights = clean_weights.groupby(data['equipment']).median().to_dict()
        
        # 2. Daily market index statistics
        if 'market_index' in data.columns:
            self.global_median_market_index = float(data['market_index'].median())
            self.daily_market_indices = data.groupby('date')['market_index'].median().to_dict()
        
        # 3. Master city coordinates lookup
        if 'pickup_lat' in data.columns and 'pickup_lon' in data.columns:
            for city, row in data.groupby('pickup')[['pickup_lat', 'pickup_lon']].first().iterrows():
                self.city_pickup_coords[str(city)] = {
                    'pickup_lat': float(row['pickup_lat']),
                    'pickup_lon': float(row['pickup_lon'])
                }
                
        if 'delivery_lat' in data.columns and 'delivery_lon' in data.columns:
            for city, row in data.groupby('delivery')[['delivery_lat', 'delivery_lon']].first().iterrows():
                self.city_delivery_coords[str(city)] = {
                    'delivery_lat': float(row['delivery_lat']),
                    'delivery_lon': float(row['delivery_lon'])
                }
                
        self.is_fitted = True
        return self

    def update_coords_from_validation(self, val_df: pd.DataFrame) -> None:
        """Incorporate validation set coordinates if new cities are present."""
        data = val_df.copy()
        if 'pickup_lat' in data.columns and 'pickup_lon' in data.columns:
            for city, row in data.groupby('pickup')[['pickup_lat', 'pickup_lon']].first().iterrows():
                if str(city) not in self.city_pickup_coords:
                    self.city_pickup_coords[str(city)] = {
                        'pickup_lat': float(row['pickup_lat']),
                        'pickup_lon': float(row['pickup_lon'])
                    }
                    
        if 'delivery_lat' in data.columns and 'delivery_lon' in data.columns:
            for city, row in data.groupby('delivery')[['delivery_lat', 'delivery_lon']].first().iterrows():
                if str(city) not in self.city_delivery_coords:
                    self.city_delivery_coords[str(city)] = {
                        'delivery_lat': float(row['delivery_lat']),
                        'delivery_lon': float(row['delivery_lon'])
                    }
                    
        if 'market_index' in data.columns and 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            val_daily = data.groupby('date')['market_index'].median().to_dict()
            for dt, mi in val_daily.items():
                if pd.notna(mi) and dt not in self.daily_market_indices:
                    self.daily_market_indices[dt] = float(mi)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply robust cleaning transformations."""
        if not self.is_fitted:
            raise RuntimeError("FreightDataCleaner must be fitted before transforming data.")
            
        data = df.copy()
        data['date'] = pd.to_datetime(data['date'])
        
        # 1. Weight cleaning
        data['weight_was_neg'] = (data['weight'] < 0).astype(int)
        data['weight_was_null'] = data['weight'].isna().astype(int)
        data['weight_clean'] = data['weight'].abs()
        
        for eq, med in self.equip_median_weights.items():
            mask = (data['equipment'] == eq) & (data['weight_clean'].isna())
            data.loc[mask, 'weight_clean'] = med
        data['weight_clean'] = data['weight_clean'].fillna(self.global_median_weight)
        
        # 2. Coordinates lookup / filling if missing (e.g., december chart inputs)
        if 'pickup_lat' not in data.columns or data['pickup_lat'].isna().any():
            if 'pickup_lat' not in data.columns:
                data['pickup_lat'] = np.nan
                data['pickup_lon'] = np.nan
            for idx, row in data.iterrows():
                p = str(row['pickup'])
                if pd.isna(row['pickup_lat']) and p in self.city_pickup_coords:
                    data.at[idx, 'pickup_lat'] = self.city_pickup_coords[p]['pickup_lat']
                    data.at[idx, 'pickup_lon'] = self.city_pickup_coords[p]['pickup_lon']
                    
        if 'delivery_lat' not in data.columns or data['delivery_lat'].isna().any():
            if 'delivery_lat' not in data.columns:
                data['delivery_lat'] = np.nan
                data['delivery_lon'] = np.nan
            for idx, row in data.iterrows():
                d = str(row['delivery'])
                if pd.isna(row['delivery_lat']) and d in self.city_delivery_coords:
                    data.at[idx, 'delivery_lat'] = self.city_delivery_coords[d]['delivery_lat']
                    data.at[idx, 'delivery_lon'] = self.city_delivery_coords[d]['delivery_lon']
        
        # 3. Market index cleaning
        if 'market_index' not in data.columns:
            data['market_index'] = np.nan
            data['market_index_was_null'] = 1
        else:
            data['market_index_was_null'] = data['market_index'].isna().astype(int)
            
        data['daily_median_mi'] = data['date'].map(self.daily_market_indices)
        data['market_index_clean'] = data['market_index'].fillna(data['daily_median_mi']).fillna(self.global_median_market_index)
        
        # 4. Quote signal handling if missing
        if 'quote_signal' not in data.columns:
            data['quote_signal'] = np.nan
            
        return data

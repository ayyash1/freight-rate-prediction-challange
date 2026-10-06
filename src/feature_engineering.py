"""
Feature engineering module for Freight Rate ML modeling.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from src.utils import haversine_distance, initial_bearing


class FreightFeatureEngineer:
    """
    Constructs domain-specific spatial, temporal, market signal, 
    equipment, and interaction features for freight pricing models.
    """
    def __init__(self):
        self.feature_columns: List[str] = []
        self.equipment_map: Dict[str, int] = {'Dry Van': 0, 'Flatbed': 1, 'Reefer': 2}
        self.lane_priors: Dict[str, Any] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "FreightFeatureEngineer":
        """Fit feature transformations, baseline signals, and target prior encodings."""
        data = df.copy()
        data['date'] = pd.to_datetime(data['date'])
        
        # Calculate base historical priors if posted_rate exists
        if 'posted_rate' in data.columns and 'distance' in data.columns:
            rpm = data['posted_rate'] / data['distance']
            global_rpm = float(rpm.median())
            pickup_rpm = rpm.groupby(data['pickup']).median().to_dict()
            delivery_rpm = rpm.groupby(data['delivery']).median().to_dict()
            lane_rpm = rpm.groupby(data['pickup'] + '->' + data['delivery']).median().to_dict()
            
            self.lane_priors = {
                'global_rpm': global_rpm,
                'pickup_rpm': pickup_rpm,
                'delivery_rpm': delivery_rpm,
                'lane_rpm': lane_rpm
            }
        
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer full feature matrix from cleaned transaction records."""
        data = df.copy()
        data['date'] = pd.to_datetime(data['date'])
        
        # 1. Spatial & Geodesic Features
        data['haversine_dist'] = haversine_distance(
            data['pickup_lat'], data['pickup_lon'], 
            data['delivery_lat'], data['delivery_lon']
        )
        data['tortuosity'] = data['distance'] / (data['haversine_dist'] + 1e-4)
        data['bearing'] = initial_bearing(
            data['pickup_lat'], data['pickup_lon'], 
            data['delivery_lat'], data['delivery_lon']
        )
        data['bearing_sin'] = np.sin(np.radians(data['bearing']))
        data['bearing_cos'] = np.cos(np.radians(data['bearing']))
        data['delta_lat'] = data['delivery_lat'] - data['pickup_lat']
        data['delta_lon'] = data['delivery_lon'] - data['pickup_lon']
        data['abs_delta_lat'] = data['delta_lat'].abs()
        data['abs_delta_lon'] = data['delta_lon'].abs()
        data['manhattan_coord_dist'] = data['abs_delta_lat'] + data['abs_delta_lon']
        data['center_lat'] = (data['delivery_lat'] + data['pickup_lat']) / 2.0
        data['center_lon'] = (data['delivery_lon'] + data['pickup_lon']) / 2.0
        
        # 2. Distance Features
        data['log_distance'] = np.log1p(data['distance'])
        data['is_short_haul'] = (data['distance'] < 300).astype(int)
        data['is_medium_haul'] = ((data['distance'] >= 300) & (data['distance'] < 800)).astype(int)
        data['is_long_haul'] = (data['distance'] >= 800).astype(int)
        
        # 3. Cargo & Weight Interactions
        weight_col = 'weight_clean' if 'weight_clean' in data.columns else 'weight'
        data['log_weight'] = np.log1p(data[weight_col])
        data['ton_miles'] = (data[weight_col] / 2000.0) * data['distance']
        data['weight_per_mile'] = data[weight_col] / (data['distance'] + 1e-4)
        data['is_heavy'] = (data[weight_col] > 40000).astype(int)
        data['is_light'] = (data[weight_col] < 20000).astype(int)
        
        # 4. Market & Quote Signal Features
        mi_col = 'market_index_clean' if 'market_index_clean' in data.columns else 'market_index'
        data['base_rate_est'] = data['distance'] * data['quote_signal']
        data['market_adjusted_base'] = data['base_rate_est'] * data[mi_col]
        data['quote_x_market'] = data['quote_signal'] * data[mi_col]
        data['quote_div_market'] = data['quote_signal'] / (data[mi_col] + 1e-4)
        
        # 5. Equipment Categorical & Equipment Specific Interactions
        data['equipment_cat'] = data['equipment'].map(self.equipment_map).fillna(0).astype(int)
        data['is_dry_van'] = (data['equipment'] == 'Dry Van').astype(float)
        data['is_flatbed'] = (data['equipment'] == 'Flatbed').astype(float)
        data['is_reefer'] = (data['equipment'] == 'Reefer').astype(float)
        
        data['reefer_x_base'] = data['is_reefer'] * data['base_rate_est']
        data['flatbed_x_base'] = data['is_flatbed'] * data['base_rate_est']
        data['dryvan_x_base'] = data['is_dry_van'] * data['base_rate_est']
        
        data['reefer_x_quote'] = data['is_reefer'] * data['quote_signal']
        data['flatbed_x_quote'] = data['is_flatbed'] * data['quote_signal']
        data['dryvan_x_quote'] = data['is_dry_van'] * data['quote_signal']
        
        data['reefer_x_weight'] = data['is_reefer'] * data[weight_col]
        data['flatbed_x_weight'] = data['is_flatbed'] * data[weight_col]
        data['dryvan_x_weight'] = data['is_dry_van'] * data[weight_col]
        
        # 6. Temporal & Calendar Dynamics
        data['dayofweek'] = data['date'].dt.dayofweek
        data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)
        data['day'] = data['date'].dt.day
        data['month'] = data['date'].dt.month
        data['dayofyear'] = data['date'].dt.dayofyear
        data['weekofyear'] = data['date'].dt.isocalendar().week.astype(int)
        data['sin_dow'] = np.sin(2 * np.pi * data['dayofweek'] / 7.0)
        data['cos_dow'] = np.cos(2 * np.pi * data['dayofweek'] / 7.0)
        data['sin_doy'] = np.sin(2 * np.pi * data['dayofyear'] / 365.25)
        data['cos_doy'] = np.cos(2 * np.pi * data['dayofyear'] / 365.25)
        data['is_month_end'] = (data['day'] >= 25).astype(int)
        data['is_month_start'] = (data['day'] <= 5).astype(int)
        data['is_quarter_end'] = ((data['month'].isin([3, 6, 9, 12])) & (data['day'] >= 24)).astype(int)
        
        # Holiday surge indicator (Memorial Day end May, July 4, Labor Day early Sep, Thanksgiving late Nov, Christmas late Dec)
        is_thanksgiving_rush = ((data['month'] == 11) & (data['day'] >= 20) & (data['day'] <= 28)).astype(int)
        is_xmas_rush = ((data['month'] == 12) & (data['day'] >= 18) & (data['day'] <= 26)).astype(int)
        data['holiday_rush'] = is_thanksgiving_rush | is_xmas_rush
        
        # 7. Lane Prior Rates
        if self.lane_priors:
            global_val = self.lane_priors.get('global_rpm', 2.21)
            pickup_map = self.lane_priors.get('pickup_rpm', {})
            delivery_map = self.lane_priors.get('delivery_rpm', {})
            lane_map = self.lane_priors.get('lane_rpm', {})
            
            data['pickup_prior_rpm'] = data['pickup'].map(pickup_map).fillna(global_val)
            data['delivery_prior_rpm'] = data['delivery'].map(delivery_map).fillna(global_val)
            data['lane_prior_rpm'] = (data['pickup'] + '->' + data['delivery']).map(lane_map).fillna(data['pickup_prior_rpm'])
        
        self.feature_columns = [
            'distance', 'log_distance', 'haversine_dist', 'tortuosity', 'bearing', 'bearing_sin', 'bearing_cos',
            'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon', 'delta_lat', 'delta_lon',
            'abs_delta_lat', 'abs_delta_lon', 'manhattan_coord_dist', 'center_lat', 'center_lon',
            weight_col, 'log_weight', 'weight_was_neg', 'weight_was_null', 'ton_miles', 'weight_per_mile',
            'is_heavy', 'is_light', 'is_short_haul', 'is_medium_haul', 'is_long_haul',
            mi_col, 'market_index_was_null', 'quote_signal',
            'base_rate_est', 'market_adjusted_base', 'quote_x_market', 'quote_div_market',
            'equipment_cat', 'is_dry_van', 'is_flatbed', 'is_reefer',
            'reefer_x_base', 'flatbed_x_base', 'dryvan_x_base',
            'reefer_x_quote', 'flatbed_x_quote', 'dryvan_x_quote',
            'reefer_x_weight', 'flatbed_x_weight', 'dryvan_x_weight',
            'dayofweek', 'is_weekend', 'day', 'month', 'dayofyear', 'weekofyear',
            'sin_dow', 'cos_dow', 'sin_doy', 'cos_doy', 'is_month_end', 'is_month_start', 'is_quarter_end', 'holiday_rush',
            'pickup_prior_rpm', 'delivery_prior_rpm', 'lane_prior_rpm'
        ]
        
        return data

    def get_feature_names(self) -> List[str]:
        """Return list of active model feature names."""
        return self.feature_columns

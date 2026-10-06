"""
Model implementations and Ensemble architectures for Freight Rate Prediction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from src.utils import compute_metrics


class FreightRateModel:
    """
    Standard interface for Freight Rate regression models.
    Supports modeling Rate Per Mile (RPM) or direct total rate.
    """
    def __init__(self, model_type: str = "lightgbm", target_mode: str = "rpm", params: Optional[Dict[str, Any]] = None):
        self.model_type = model_type.lower()
        self.target_mode = target_mode.lower()
        self.params = params or {}
        self.model = None
        self.feature_names: List[str] = []
        self._init_model()

    def _init_model(self):
        if self.model_type == "lightgbm":
            default_params = {
                'objective': 'regression_l1',
                'metric': 'mae',
                'n_estimators': 1500,
                'learning_rate': 0.03,
                'num_leaves': 63,
                'subsample': 0.85,
                'colsample_bytree': 0.80,
                'min_child_samples': 20,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            default_params.update(self.params)
            self.model = lgb.LGBMRegressor(**default_params)

        elif self.model_type == "xgboost":
            default_params = {
                'objective': 'reg:absoluteerror',
                'eval_metric': 'mae',
                'n_estimators': 1500,
                'learning_rate': 0.03,
                'max_depth': 6,
                'subsample': 0.85,
                'colsample_bytree': 0.80,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(self.params)
            self.model = xgb.XGBRegressor(**default_params)

        elif self.model_type == "catboost":
            default_params = {
                'loss_function': 'MAE',
                'eval_metric': 'MAE',
                'iterations': 1500,
                'learning_rate': 0.04,
                'depth': 6,
                'l2_leaf_reg': 3.0,
                'random_seed': 42,
                'verbose': 0
            }
            default_params.update(self.params)
            self.model = CatBoostRegressor(**default_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, dist_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, 
            dist_val: Optional[pd.Series] = None, early_stopping_rounds: int = 50) -> "FreightRateModel":
        
        self.feature_names = list(X_train.columns)
        
        # Transform target based on target_mode
        if self.target_mode == "rpm":
            y_tr_target = y_train / dist_train
            y_val_target = (y_val / dist_val) if y_val is not None else None
        else:
            y_tr_target = y_train
            y_val_target = y_val

        if self.model_type == "lightgbm":
            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)] if X_val is not None else []
            eval_set = [(X_val, y_val_target)] if X_val is not None else None
            self.model.fit(X_train, y_tr_target, eval_set=eval_set, callbacks=callbacks)

        elif self.model_type == "xgboost":
            eval_set = [(X_val, y_val_target)] if X_val is not None else None
            if eval_set:
                self.model.set_params(early_stopping_rounds=early_stopping_rounds)
                self.model.fit(X_train, y_tr_target, eval_set=eval_set, verbose=False)
            else:
                self.model.fit(X_train, y_tr_target, verbose=False)

        elif self.model_type == "catboost":
            eval_set = (X_val, y_val_target) if X_val is not None else None
            if eval_set:
                self.model.fit(X_train, y_tr_target, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds, verbose=0)
            else:
                self.model.fit(X_train, y_tr_target, verbose=0)

        return self

    def predict(self, X: pd.DataFrame, dist: pd.Series) -> np.ndarray:
        raw_pred = self.model.predict(X)
        if self.target_mode == "rpm":
            pred_rate = raw_pred * np.asarray(dist)
        else:
            pred_rate = raw_pred
        # Ensure rate is strictly positive
        return np.clip(pred_rate, 10.0, None)

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model_type == "lightgbm":
            importances = self.model.feature_importances_
        elif self.model_type == "xgboost":
            importances = self.model.feature_importances_
        elif self.model_type == "catboost":
            importances = self.model.get_feature_importance()
        else:
            importances = np.zeros(len(self.feature_names))

        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).reset_index(drop=True)


class FreightEnsembleModel:
    """
    Weighted blend of diverse gradient boosted decision tree architectures (LightGBM, XGBoost, CatBoost).
    """
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {'lightgbm': 0.40, 'catboost': 0.40, 'xgboost': 0.20}
        self.models: Dict[str, FreightRateModel] = {
            'lightgbm': FreightRateModel(model_type='lightgbm', target_mode='rpm'),
            'catboost': FreightRateModel(model_type='catboost', target_mode='rpm'),
            'xgboost': FreightRateModel(model_type='xgboost', target_mode='rpm')
        }

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, dist_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
            dist_val: Optional[pd.Series] = None) -> "FreightEnsembleModel":
        
        for name, model in self.models.items():
            model.fit(X_train, y_train, dist_train, X_val, y_val, dist_val)
        return self

    def predict(self, X: pd.DataFrame, dist: pd.Series) -> np.ndarray:
        preds = np.zeros(len(X))
        for name, model in self.models.items():
            weight = self.weights.get(name, 0.0)
            preds += weight * model.predict(X, dist)
        return np.clip(preds, 10.0, None)

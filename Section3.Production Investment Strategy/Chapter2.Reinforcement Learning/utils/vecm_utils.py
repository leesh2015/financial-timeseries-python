"""
VECM (Vector Error Correction Model) utilities
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VECM
from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank
from typing import Tuple, Optional


def fit_vecm_model(data: pd.DataFrame, 
                   max_lags: int = 15,
                   deterministic: str = "colo") -> Tuple[VECM, object, int, int]:
    """
    Fit VECM model with optimal lag order and cointegration rank.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    max_lags : int
        Maximum lag order to test
    deterministic : str
        Deterministic term specification ("colo" = constant in cointegration)
        
    Returns:
    --------
    Tuple[VECM, object, int, int]
        (model, fitted_model, k_ar_diff_opt, coint_rank_opt)
    """
    # Find optimal lag order
    lag_order = select_order(data, maxlags=max_lags, deterministic=deterministic)
    time_lag = lag_order.aic
    
    # Find cointegration rank using Johansen test
    coint_rank_test = select_coint_rank(data, det_order=1, k_ar_diff=time_lag, method='trace')
    coint_rank_opt = coint_rank_test.rank
    k_ar_diff_opt = time_lag
    
    # Fit VECM model
    model = VECM(data, k_ar_diff=k_ar_diff_opt, coint_rank=coint_rank_opt, deterministic=deterministic)
    model_fitted = model.fit()
    
    return model, model_fitted, k_ar_diff_opt, coint_rank_opt


def get_alpha_value(model_fitted, target_index: str, history: pd.DataFrame, 
                   method: str = 'weighted_mean') -> float:
    """
    Extract error correction term (ECT) alpha value from VECM model.
    
    Alpha is the error correction speed:
    - Alpha < 0: Mean-reverting (desirable)
    - Alpha > 0: Divergent (problematic)
    
    Parameters:
    -----------
    model_fitted : VECMResults
        Fitted VECM model
    target_index : str
        Target variable name
    history : pd.DataFrame
        Historical data
    method : str
        'weighted_mean', 'min_negative', 'sum_negative'
        
    Returns:
    --------
    float
        Comprehensive alpha value for convergence judgment
    """
    try:
        alpha = model_fitted.alpha
        target_idx = history.columns.get_loc(target_index)
        
        if alpha.ndim == 2:
            if target_idx < alpha.shape[0] and alpha.shape[1] > 0:
                target_alphas = alpha[target_idx, :]
                negative_alphas = target_alphas[target_alphas < 0]
                
                if len(negative_alphas) > 0:
                    if method == 'weighted_mean':
                        abs_negative = np.abs(negative_alphas)
                        weights = abs_negative / np.sum(abs_negative)
                        weighted_alpha = np.sum(negative_alphas * weights)
                        return float(weighted_alpha)
                    elif method == 'sum_negative':
                        return float(np.sum(negative_alphas))
                    elif method == 'min_negative':
                        return float(np.min(negative_alphas))
                    else:
                        abs_negative = np.abs(negative_alphas)
                        weights = abs_negative / np.sum(abs_negative)
                        return float(np.sum(negative_alphas * weights))
                else:
                    return float(np.mean(target_alphas))
            else:
                return 0.0
        elif alpha.ndim == 1:
            if target_idx < len(alpha):
                return float(alpha[target_idx])
            else:
                return 0.0
        else:
            return 0.0
    except Exception as e:
        print(f"Warning: Error extracting alpha value: {e}")
        return 0.0


def get_vecm_confidence(model_fitted, target_index: str, history: pd.DataFrame, 
                       lower_bound: float, upper_bound: float, predicted_mean: float) -> float:
    """
    Calculate confidence of VECM model.
    
    Combines:
    1. Prediction interval width
    2. Residual standard deviation
    3. Alpha absolute value
    """
    try:
        target_idx = history.columns.get_loc(target_index)
        
        if predicted_mean > 0:
            interval_width = (upper_bound - lower_bound) / predicted_mean
        else:
            interval_width = 1.0
        
        residuals = model_fitted.resid
        if residuals.ndim == 2:
            target_residuals = residuals[:, target_idx]
        else:
            target_residuals = residuals
        
        residual_std = np.std(target_residuals)
        if predicted_mean > 0:
            normalized_residual_std = residual_std / predicted_mean
        else:
            normalized_residual_std = 1.0
        
        alpha = model_fitted.alpha
        if alpha.ndim == 2:
            target_alpha = alpha[target_idx, :]
            negative_alphas = target_alpha[target_alpha < 0]
            if len(negative_alphas) > 0:
                alpha_abs_mean = np.mean(np.abs(negative_alphas))
            else:
                alpha_abs_mean = 0.0
        else:
            if target_idx < len(alpha):
                alpha_val = alpha[target_idx]
                alpha_abs_mean = abs(alpha_val) if alpha_val < 0 else 0.0
            else:
                alpha_abs_mean = 0.0
        
        confidence_interval = max(0, 1 - min(interval_width / 1.0, 1.0))
        confidence_residual = max(0, 1 - min(normalized_residual_std / 0.5, 1.0))
        confidence_alpha = min(alpha_abs_mean / 0.05, 1.0)
        
        confidence = (0.5 * confidence_interval + 
                     0.3 * confidence_residual + 
                     0.2 * confidence_alpha)
        
        confidence_expanded = 0.2 + (confidence * 0.7)
        
        return float(np.clip(confidence_expanded, 0.0, 1.0))
        
    except Exception as e:
        return 0.5

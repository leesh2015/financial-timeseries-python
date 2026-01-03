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
    
    VECM Model:
    ΔY_t = αβ'Y_{t-1} + Γ₁ΔY_{t-1} + ... + Γ_{p-1}ΔY_{t-p+1} + ε_t
    
    where:
    - ΔY_t: first difference of Y_t
    - α: adjustment coefficients (speed of adjustment to equilibrium)
    - β: cointegration vectors (long-run relationships)
    - Γᵢ: short-run dynamics coefficients
    - ε_t: error terms (residuals)
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data (columns are variables)
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
    
    Alpha is the error correction speed in VECM:
    - Alpha < 0: Mean-reverting (desirable)
    - Alpha > 0: Divergent (problematic)
    - Alpha = 0: No equilibrium adjustment
    
    In VECM equation: ΔY_t = αβ'Y_{t-1} + ...
    - α: adjustment speed coefficients
    - When multiple cointegration relations exist, alpha is a matrix
    
    Parameters:
    -----------
    model_fitted : VECMResults
        Fitted VECM model
    target_index : str
        Target variable name
    history : pd.DataFrame
        Historical data
    method : str
        'weighted_mean': Weighted average of negative alphas (most accurate)
        'min_negative': Most negative alpha (strongest convergence)
        'sum_negative': Sum of all negative alphas
        
    Returns:
    --------
    float
        Comprehensive alpha value for convergence judgment
    """
    try:
        alpha = model_fitted.alpha
        target_idx = history.columns.get_loc(target_index)
        
        if alpha.ndim == 2:
            # 2D array: (n_vars, n_coint_relations)
            if target_idx < alpha.shape[0] and alpha.shape[1] > 0:
                target_alphas = alpha[target_idx, :]
                
                # Filter negative alphas (converging ones)
                negative_alphas = target_alphas[target_alphas < 0]
                
                if len(negative_alphas) > 0:
                    if method == 'weighted_mean':
                        # Weighted average by absolute value
                        abs_negative = np.abs(negative_alphas)
                        weights = abs_negative / np.sum(abs_negative)
                        weighted_alpha = np.sum(negative_alphas * weights)
                        return float(weighted_alpha)
                    elif method == 'sum_negative':
                        return float(np.sum(negative_alphas))
                    elif method == 'min_negative':
                        return float(np.min(negative_alphas))
                    else:
                        # Default: weighted_mean
                        abs_negative = np.abs(negative_alphas)
                        weights = abs_negative / np.sum(abs_negative)
                        return float(np.sum(negative_alphas * weights))
                else:
                    # No negative alpha - return mean (but this is problematic)
                    return float(np.mean(target_alphas))
            else:
                return 0.0
        elif alpha.ndim == 1:
            # 1D array: (n_coint_relations,)
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
    
    [Important] Prediction interval consistency: lower_bound and upper_bound must come from the same forecast period (forecast_steps)
    
    Instead of R², combines multiple indicators to measure confidence:
    1. Prediction interval width: (upper - lower) / predicted_mean (smaller = higher confidence)
    2. Residual standard deviation: smaller = higher confidence (past fit indicator)
    3. Alpha absolute value: larger = faster convergence speed (cointegration relation strength)
    
    Parameters:
    -----------
    model_fitted : VECMResults
        Fitted VECM model
    target_index : str
        Target variable name
    history : pd.DataFrame
        Historical data
    lower_bound : float
        Prediction lower bound
    upper_bound : float
        Prediction upper bound
    predicted_mean : float
        Predicted mean value
    
    Returns:
    --------
    float
        Confidence score (0-1 range, higher = higher confidence)
    """
    try:
        target_idx = history.columns.get_loc(target_index)
        
        # 1. Prediction interval width (smaller = higher confidence)
        if predicted_mean > 0:
            interval_width = (upper_bound - lower_bound) / predicted_mean
        else:
            interval_width = 1.0  # If predicted value <= 0, consider worst case
        
        # 2. Residual standard deviation (smaller = higher confidence)
        residuals = model_fitted.resid
        if residuals.ndim == 2:
            target_residuals = residuals[:, target_idx]
        else:
            target_residuals = residuals
        
        residual_std = np.std(target_residuals)
        # Normalize residual std relative to predicted value
        if predicted_mean > 0:
            normalized_residual_std = residual_std / predicted_mean
        else:
            normalized_residual_std = 1.0
        
        # 3. Alpha absolute value (larger = faster convergence speed, higher confidence)
        alpha = model_fitted.alpha
        if alpha.ndim == 2:
            target_alpha = alpha[target_idx, :]
            # Mean of absolute values of negative alphas (only converging ones)
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
        
        # Normalize each indicator to 0-1 range and combine
        # Prediction interval width: Map to wider range to increase sensitivity
        # Map 0-1.0 range to 1-0 (if >=1.0 then 0)
        confidence_interval = max(0, 1 - min(interval_width / 1.0, 1.0))
        
        # Residual std: Map to wider range
        # Map 0-0.5 range to 1-0
        confidence_residual = max(0, 1 - min(normalized_residual_std / 0.5, 1.0))
        
        # Alpha: Map to wider range
        # Map 0-0.05 range to 0-1 (if >=0.05 then 1)
        confidence_alpha = min(alpha_abs_mean / 0.05, 1.0)
        
        # Combine with weighted average (prediction interval most important)
        confidence = (0.5 * confidence_interval + 
                     0.3 * confidence_residual + 
                     0.2 * confidence_alpha)
        
        # Expand confidence to wider range (map 0-1 to 0.2-0.9 for larger differences)
        # This makes position sizes vary more significantly even when confidence is 0.5-0.8
        confidence_expanded = 0.2 + (confidence * 0.7)  # Expand 0-1 to 0.2-0.9
        
        return float(np.clip(confidence_expanded, 0.0, 1.0))
        
    except Exception as e:
        if not hasattr(get_vecm_confidence, '_error_count'):
            get_vecm_confidence._error_count = 0
        if get_vecm_confidence._error_count < 3:
            print(f"Warning: Error calculating VECM confidence: {e}")
            get_vecm_confidence._error_count += 1
        return 0.5  # Default value



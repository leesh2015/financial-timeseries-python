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


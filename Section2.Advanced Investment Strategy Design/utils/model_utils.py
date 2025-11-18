"""
Model utilities for VECM and GARCH models
"""

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.api import VECM
from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank
from typing import Tuple, Optional


def find_garch(
    residuals: np.ndarray,
    vol_type: str = 'EGARCH',
    max_p: int = 2,
    max_o: int = 2,
    max_q: int = 2
) -> Tuple[float, Tuple[int, int, int], Optional[object]]:
    """
    Find optimal GARCH model order using AIC criterion.
    
    Parameters:
    -----------
    residuals : np.ndarray
        Residuals from VECM or other model
    vol_type : str
        Volatility model type ('EGARCH', 'GARCH', 'APARCH', etc.)
    max_p : int
        Maximum ARCH order
    max_o : int
        Maximum leverage order (for EGARCH, APARCH)
    max_q : int
        Maximum GARCH order
        
    Returns:
    --------
    tuple: (best_aic, best_order, best_model)
        Best AIC value, optimal order (p, o, q), and fitted model
    """
    best_aic = np.inf
    best_order = None
    best_model = None

    for p in range(max_p):
        for o in range(max_o):
            for q in range(max_q):
                try:
                    model = arch_model(residuals, vol=vol_type, 
                                       p=p, o=o, q=q, rescale=True)
                    model_fit = model.fit(disp='off')
                    aic = model_fit.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, o, q)
                        best_model = model_fit
                except:
                    continue

    return best_aic, best_order, best_model


def initialize_vecm_models(
    train_data: pd.DataFrame,
    maxlags: int = 15,
    deterministic: str = "colo"
) -> Tuple[int, int]:
    """
    Initialize VECM models with optimal parameters.
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    maxlags : int
        Maximum lags to test
    deterministic : str
        Deterministic term specification
        
    Returns:
    --------
    tuple: (k_ar_diff_opt, coint_rank_opt)
        Optimal lag order and cointegration rank
    """
    # Find optimal lag order
    lag_order = select_order(train_data, maxlags=maxlags, deterministic=deterministic)
    k_ar_diff_opt = lag_order.aic
    
    # Find optimal cointegration rank
    coint_rank_test = select_coint_rank(train_data, det_order=1, 
                                       k_ar_diff=k_ar_diff_opt, method='trace')
    coint_rank_opt = coint_rank_test.rank
    
    return k_ar_diff_opt, coint_rank_opt


"""
GARCH model utilities
"""

import numpy as np
from arch import arch_model
from typing import Tuple, Optional


def find_garch(residuals: np.ndarray) -> Tuple[float, Tuple[int, int, int], Optional[object]]:
    """
    Find optimal GARCH order by minimizing AIC.
    
    Tests EGARCH(p,o,q) models where:
    - p: ARCH order
    - o: asymmetry order (leverage effect)
    - q: GARCH order
    
    EGARCH Model:
    log(σ²_t) = ω + Σᵢ₌₁ᵖ (αᵢ|z_{t-i}| + γᵢz_{t-i}) + Σⱼ₌₁ᵠ βⱼlog(σ²_{t-j})
    
    where:
    - z_t = ε_t / σ_t (standardized residuals)
    - γᵢ: leverage parameters (captures asymmetric volatility effects)
    - log(σ²_t): ensures positive variance
    
    Parameters:
    -----------
    residuals : np.ndarray
        Residuals from VECM model
        
    Returns:
    --------
    Tuple[float, Tuple[int, int, int], Optional[object]]
        (best_aic, best_order, best_model)
        best_order is (p, o, q) for EGARCH
    """
    best_aic = np.inf
    best_order = None
    best_model = None

    for p in range(0, 2):
        for o in range(0, 2):
            for q in range(0, 2):
                try:
                    model = arch_model(residuals, vol='EGARCH', 
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


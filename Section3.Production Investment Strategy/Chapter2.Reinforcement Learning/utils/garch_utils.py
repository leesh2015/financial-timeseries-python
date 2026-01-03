"""
GARCH (Generalized Autoregressive Conditional Heteroskedasticity) utilities
"""

import numpy as np
from arch import arch_model
from typing import Tuple, Optional

def find_garch(residuals: np.ndarray) -> Tuple[float, Tuple[int, int, int], object]:
    """
    Find the optimal EGARCH order (p, o, q) by minimizing AIC.
    
    Parameters:
    -----------
    residuals : np.ndarray
        Residuals from the VECM model
        
    Returns:
    --------
    Tuple[float, Tuple[int, int, int], object]
        (best_aic, best_order, best_model_fit)
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

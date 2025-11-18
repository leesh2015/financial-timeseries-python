"""
GARCH(p,q) Model Optimization

This script demonstrates how to find the optimal GARCH order by testing
various GARCH(p,q) models and selecting based on AIC/BIC criteria.
"""

import numpy as np
from pmdarima import auto_arima
from arch import arch_model
import sys
import os
import warnings

warnings.filterwarnings("ignore")

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
try:
    from utils.data_loader import load_financial_data, preprocess_data
    from utils.config import config
    USE_UTILS = True
except ImportError:
    USE_UTILS = False
    import yfinance as yf

# Configuration
TICKER = 'SPY'
START_DATE = '2020-01-01'
END_DATE = '2024-12-01'
MAX_P = 10  # Maximum GARCH p order
MAX_Q = 10  # Maximum GARCH q order


def main():
    """Main function"""
    # Load data
    if USE_UTILS:
        data = load_financial_data(TICKER, START_DATE, END_DATE)
        data = preprocess_data(data, frequency='D', use_adjusted=False)
        log_returns = np.log(data['Close']).diff().dropna()
    else:
        data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
        log_returns = np.log(data['Close']).diff().dropna()

    # Use AutoARIMA to find the optimal ARIMA model and fit it
    model = auto_arima(
        log_returns, 
        seasonal=False, 
        trace=True, 
        error_action='ignore', 
        suppress_warnings=True, 
        stepwise=True
    )
    residuals = model.resid()
    
    # Fit and optimize GARCH models of various orders
    aic_values = []
    bic_values = []
    
    print("Testing GARCH models...")
    for p in range(1, MAX_P + 1):
        for q in range(1, MAX_Q + 1):
            try:
                garch_model_fitted = arch_model(residuals, vol='GARCH', p=p, q=q)
                garch_results = garch_model_fitted.fit(disp="off")
                aic_values.append((p, q, garch_results.aic))
                bic_values.append((p, q, garch_results.bic))
                print(f"GARCH({p},{q}) - AIC: {garch_results.aic:.4f}, BIC: {garch_results.bic:.4f}")
            except Exception as e:
                print(f"GARCH({p},{q}) - Failed: {e}")
                aic_values.append((p, q, np.inf))
                bic_values.append((p, q, np.inf))
    
    # Select the optimal order based on AIC
    optimal_p, optimal_q, min_aic = min(aic_values, key=lambda x: x[2])
    print(f"\nOptimal GARCH order (AIC criterion): p={optimal_p}, q={optimal_q}")
    
    # Select the optimal order based on BIC
    optimal_p_bic, optimal_q_bic, min_bic = min(bic_values, key=lambda x: x[2])
    print(f"Optimal GARCH order (BIC criterion): p={optimal_p_bic}, q={optimal_q_bic}")


if __name__ == "__main__":
    main()

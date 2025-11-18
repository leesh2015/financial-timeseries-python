"""
ARCH(p) Model Optimization

This script demonstrates how to find the optimal ARCH order by testing
various ARCH(p) models and selecting based on AIC/BIC criteria.
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
MAX_P = 10  # Maximum ARCH order to test


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

    # Use AutoARIMA to find the best ARIMA model and fit it
    model = auto_arima(
        log_returns, 
        seasonal=False, 
        trace=True, 
        error_action='ignore', 
        suppress_warnings=True, 
        stepwise=True
    )
    residuals = model.resid()
    
    # Fit and optimize ARCH models of various orders
    aic_values = []
    bic_values = []
    
    print("Testing ARCH models...")
    for p in range(1, MAX_P + 1):
        try:
            arch_model_fitted = arch_model(residuals, vol='ARCH', p=p)
            arch_results = arch_model_fitted.fit(disp="off")
            aic_values.append(arch_results.aic)
            bic_values.append(arch_results.bic)
            print(f"ARCH({p}) - AIC: {arch_results.aic:.4f}, BIC: {arch_results.bic:.4f}")
        except Exception as e:
            print(f"ARCH({p}) - Failed: {e}")
            aic_values.append(np.inf)
            bic_values.append(np.inf)
    
    # Select the optimal order
    optimal_p = np.argmin(aic_values) + 1
    print(f"\nOptimal ARCH order (based on AIC): {optimal_p}")
    
    optimal_p_bic = np.argmin(bic_values) + 1
    print(f"Optimal ARCH order (based on BIC): {optimal_p_bic}")


if __name__ == "__main__":
    main()

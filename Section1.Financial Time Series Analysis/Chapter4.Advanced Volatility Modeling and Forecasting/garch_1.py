"""
GARCH(1,1) Model Implementation

This script demonstrates how to fit a GARCH(1,1) model to ARIMA residuals.
The process involves:
1. Fitting an ARIMA model to remove mean dynamics
2. Fitting a GARCH model to the ARIMA residuals to capture volatility clustering

GARCH(1,1) model: σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}
where:
- ω: constant term
- α: ARCH term (response to shocks)
- β: GARCH term (persistence of volatility)
"""

import yfinance as yf
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

# Configuration
TICKER = 'SPY'
START_DATE = '2020-01-01'
END_DATE = '2024-12-01'
GARCH_P = 1  # ARCH order
GARCH_Q = 1  # GARCH order
DISTRIBUTION = 't'  # Student's t distribution (more realistic for financial data)


def fit_arima_garch(close_prices):
    """
    Fit ARIMA model and then GARCH model to residuals.
    
    Parameters:
    -----------
    close_prices : pd.Series
        Time series of closing prices
        
    Returns:
    --------
    tuple
        (arima_model, garch_results, residuals)
    """
    # Step 1: Fit ARIMA model to capture mean dynamics
    print("="*60)
    print("Step 1: Fitting ARIMA Model")
    print("="*60)
    arima_model = auto_arima(
        close_prices,
        seasonal=False,  # Non-seasonal ARIMA
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    
    print("\nARIMA Model Summary:")
    print(arima_model.summary())
    print(f"\nEstimated ARIMA coefficients:")
    print(arima_model.params())
    
    # Step 2: Extract residuals from ARIMA model
    residuals = arima_model.resid()
    print(f"\nResiduals Statistics:")
    print(f"  Mean: {residuals.mean():.6f}")
    print(f"  Std: {residuals.std():.6f}")
    print(f"  Min: {residuals.min():.6f}")
    print(f"  Max: {residuals.max():.6f}")
    
    # Step 3: Fit GARCH model to residuals
    print("\n" + "="*60)
    print(f"Step 2: Fitting GARCH({GARCH_P},{GARCH_Q}) Model to Residuals")
    print("="*60)
    garch_model = arch_model(
        residuals,
        vol='GARCH',
        dist=DISTRIBUTION,  # Student's t distribution
        p=GARCH_P,
        q=GARCH_Q
    )
    garch_results = garch_model.fit()
    
    print("\nGARCH Model Summary:")
    print(garch_results.summary())
    
    return arima_model, garch_results, residuals


def main():
    """Main function"""
    # Load data
    if USE_UTILS:
        data = load_financial_data(TICKER, START_DATE, END_DATE)
        close_prices = preprocess_data(data, frequency='B', use_adjusted=False)
        close_prices = close_prices['Close']
    else:
        data = yf.download(TICKER, start=START_DATE, end=END_DATE)
        close_prices = data['Close']
        close_prices = close_prices.asfreq('B')  # Business day frequency
        close_prices = close_prices.dropna()
    
    # Fit ARIMA + GARCH model
    arima_model, garch_results, residuals = fit_arima_garch(close_prices)
    
    print("\n" + "="*60)
    print("Analysis Complete")
    print("="*60)
    print(f"\nARIMA Order: {arima_model.order}")
    print(f"GARCH Order: ({GARCH_P}, {GARCH_Q})")
    print(f"Distribution: {DISTRIBUTION}")


if __name__ == "__main__":
    main()

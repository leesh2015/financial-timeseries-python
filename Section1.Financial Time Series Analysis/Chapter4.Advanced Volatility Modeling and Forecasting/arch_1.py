"""
ARCH(1) Model Implementation

This script demonstrates how to fit an ARCH(1) model to ARIMA residuals.
ARCH models capture volatility clustering in financial time series.

ARCH(1) model: σ²_t = ω + α*ε²_{t-1}
"""

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
ARCH_P = 1  # ARCH order
DISTRIBUTION = 't'  # Student's t distribution


def main():
    """Main function"""
    # Load data
    if USE_UTILS:
        data = load_financial_data(TICKER, START_DATE, END_DATE)
        close_prices = preprocess_data(data, frequency='B', use_adjusted=False)
        close_prices = close_prices['Close']
    else:
        data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
        close_prices = data['Close']
        close_prices = close_prices.asfreq('B')
        close_prices = close_prices.dropna()

    # Use AutoARIMA to find the optimal ARIMA model
    model = auto_arima(
        close_prices, 
        seasonal=False,  # Search for a non-seasonal ARIMA model
        trace=True,  # Print the model selection process
        error_action='ignore',  # Ignore errors and proceed
        suppress_warnings=True,  # Suppress warning messages
        stepwise=True  # Perform stepwise search for efficiency
    )
    
    # Print the summary of the model
    print(model.summary())
    
    # Print the estimated coefficients
    print("Estimated coefficients:")
    print(model.params())
    
    # Extract residuals from the ARIMA model
    residuals = model.resid()
    
    # Fit an ARCH model to the residuals
    arch_model_fitted = arch_model(residuals, vol='ARCH', dist=DISTRIBUTION, p=ARCH_P)
    arch_results = arch_model_fitted.fit()
    
    # Print the summary of the ARCH model
    print("\nARCH Model Summary:")
    print(arch_results.summary())


if __name__ == "__main__":
    main()

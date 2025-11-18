"""
ARMA (Autoregressive Moving Average) Model Implementation

This script demonstrates how to fit an ARMA(p,q) model, which combines both
AR and MA components. ARMA(1,1) is a common starting point.

ARMA model: combines AR terms (past values) and MA terms (past errors)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
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
END_DATE = '2023-01-01'
AR_ORDER = 1  # AR order (p)
MA_ORDER = 1  # MA order (q)
FORECAST_STEPS = 10  # Number of steps to forecast


def main():
    """Main function"""
    # Load data - following original code order exactly
    if USE_UTILS:
        data = load_financial_data(TICKER, START_DATE, END_DATE)
        close_prices = data['Close']
    else:
        data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
        close_prices = data['Close']
    
    # Follow original code order exactly:
    # 1. Add frequency information to date index
    close_prices = close_prices.asfreq('B')  # 'B' stands for business day frequency
    
    # 2. Check and remove NaN values
    close_prices = close_prices.dropna()
    
    # 3. Generate log-differenced data of closing prices
    log_diff = np.log(close_prices).diff().dropna()
    
    # 4. Add frequency information again (as in original code)
    log_diff = log_diff.asfreq('B')

    # Fit the ARMA model
    # ARIMA(p, 0, q) is equivalent to ARMA(p, q) model
    print(f"\nFitting ARMA({AR_ORDER},{MA_ORDER}) model (ARIMA({AR_ORDER}, 0, {MA_ORDER}))...")
    model = sm.tsa.ARIMA(log_diff, order=(AR_ORDER, 0, MA_ORDER))
    results = model.fit()
    
    # Print model summary
    print("\n" + "="*60)
    print("ARMA Model Summary")
    print("="*60)
    print(results.summary())
    
    # Forecast for future periods
    print(f"\n{'='*60}")
    print(f"Forecasting next {FORECAST_STEPS} periods")
    print("="*60)
    last_date = log_diff.index[-1]
    forecast = results.get_forecast(steps=FORECAST_STEPS)
    forecast_index = pd.date_range(
        start=last_date + pd.Timedelta(days=1), 
        periods=FORECAST_STEPS, 
        freq='B'
    )
    predictions = forecast.predicted_mean
    predictions.index = forecast_index
    
    # Get confidence intervals
    # Note: conf_int() returns a DataFrame with two columns:
    # - First column: lower bound
    # - Second column: upper bound
    # Column names may vary, so we use iloc for safe access
    conf_int = forecast.conf_int()
    conf_int.index = forecast_index
    
    print("\nForecast Results:")
    print(f"{'Date':<12} {'Forecast':>12} {'Lower CI':>12} {'Upper CI':>12}")
    print("-" * 50)
    
    # Use iloc for safe column access (first column = lower, second = upper)
    for i, date in enumerate(forecast_index):
        pred = predictions[date]
        lower = conf_int.iloc[i, 0]  # First column: lower bound
        upper = conf_int.iloc[i, 1]   # Second column: upper bound
        print(f"{date.strftime('%Y-%m-%d'):<12} {pred:>12.4f} {lower:>12.4f} {upper:>12.4f}")


if __name__ == "__main__":
    main()

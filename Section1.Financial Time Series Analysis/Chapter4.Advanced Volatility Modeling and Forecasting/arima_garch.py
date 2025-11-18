"""
ARIMA-GARCH Hybrid Model with Iterative Forecasting

This script demonstrates iterative forecasting using ARIMA-GARCH hybrid model.
The model is re-trained at each step to incorporate new information.

Note: This is computationally expensive but more realistic for real-time forecasting.
"""

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import sys
import os
import warnings

warnings.filterwarnings("ignore")

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
try:
    from utils.data_loader import load_financial_data, preprocess_data, split_train_test
    from utils.config import config
    USE_UTILS = True
except ImportError:
    USE_UTILS = False
    import yfinance as yf

# Configuration
TICKER = 'SPY'
START_DATE = '2020-01-01'
END_DATE = '2022-12-01'
TRAIN_RATIO = 0.8
GARCH_P = 1
GARCH_Q = 1


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE"""
    mask = (y_true != 0) & np.isfinite(y_pred)
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def main():
    """Main function"""
    # Load and preprocess data
    if USE_UTILS:
        data = load_financial_data(TICKER, START_DATE, END_DATE)
        data = preprocess_data(data, frequency='D', use_adjusted=False)
        close_prices = data['Close']
    else:
        data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
        close_prices = data['Close']
    
    # Ensure close_prices is a Series (not DataFrame)
    # If it's a DataFrame (e.g., from MultiIndex), convert to Series
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.squeeze()  # Convert to Series if single column
    
    # Split the data
    train_size = int(len(close_prices) * TRAIN_RATIO)
    train, test = close_prices[:train_size], close_prices[train_size:]
    
    # Train ARIMA model once
    print("Training initial ARIMA model...")
    arima_model = auto_arima(
        train, 
        seasonal=False, 
        trace=True, 
        error_action='ignore', 
        suppress_warnings=True, 
        stepwise=True
    )
    order = arima_model.order
    arima_residuals = arima_model.resid()
    
    # Train GARCH model once
    print("Training initial GARCH model...")
    garch_model = arch_model(arima_residuals, vol='GARCH', p=GARCH_P, q=GARCH_Q)
    garch_results = garch_model.fit()
    
    # Perform iterative predictions with pre-trained ARIMA + GARCH model
    print(f"\nPerforming iterative forecasting on {len(test)} test points...")
    arima_plus_garch_forecast = []
    history = train.tolist()
    
    for t in range(len(test)):
        if (t + 1) % 50 == 0:
            print(f"  Progress: {t+1}/{len(test)}")
        
        # Predict next step using ARIMA model (re-trained each step)
        model = sm.tsa.ARIMA(history, order=order)
        model_fit = model.fit()
        arima_forecast = model_fit.forecast(steps=1)[0]
        
        # Update GARCH model with new residuals
        arima_residuals = model_fit.resid
        garch_model = arch_model(arima_residuals, vol='GARCH', p=GARCH_P, q=GARCH_Q)
        garch_fit = garch_model.fit(disp="off")
        garch_forecast = garch_fit.forecast(horizon=1)
        
        garch_mean_average = garch_forecast.mean.values[-1, 0]
        hybrid_yhat = arima_forecast + garch_mean_average
        
        arima_plus_garch_forecast.append(hybrid_yhat)
        
        # Update history with the actual value from the test set
        history.append(test.iloc[t])
    
    # Convert arima_plus_garch_forecast to a numpy array
    arima_plus_garch_forecast = np.array(arima_plus_garch_forecast)
    
    # Calculate performance metrics
    mape = mean_absolute_percentage_error(test.values, arima_plus_garch_forecast)
    mse = mean_squared_error(test, arima_plus_garch_forecast)
    mae = mean_absolute_error(test, arima_plus_garch_forecast)
    rmse = np.sqrt(mse)
    
    print(f"\n{'='*60}")
    print('Performance of Iterative ARIMA + GARCH model:')
    print("="*60)
    print(f'MSE: {mse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'MAPE: {mape:.2f}%')


if __name__ == "__main__":
    main()

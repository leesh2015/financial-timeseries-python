"""
ARIMA-GARCH Hybrid Model Backtesting

This script performs backtesting of a trading strategy using an ARIMA-GARCH
hybrid model for single asset trading.

ARIMA Model:
ARIMA(p,d,q): (1 - φ₁B - ... - φₚBᵖ)(1 - B)ᵈY_t = (1 + θ₁B + ... + θₚBᵠ)ε_t

where:
- p: AR order
- d: differencing order
- q: MA order
- B: backshift operator
- ε_t: white noise error term

GARCH Model:
GARCH(p,q): σ²_t = ω + Σᵢ₌₁ᵖ αᵢε²_{t-i} + Σⱼ₌₁ᵠ βⱼσ²_{t-j}

where:
- σ²_t: conditional variance at time t
- ω: constant term
- αᵢ: ARCH coefficients (lagged squared errors)
- βⱼ: GARCH coefficients (lagged variances)

Hybrid Model:
Ŷ_{t+1} = ARIMA_forecast + GARCH_mean

where:
- ARIMA_forecast: ARIMA model prediction for next period
- GARCH_mean: GARCH model mean forecast for ARIMA residuals
- Hybrid combines both mean prediction and volatility adjustment

Trading Strategy:
- Buy when: hybrid_yhat > actual_price (model predicts price increase)
- Sell when: hybrid_yhat < actual_price (model predicts price decrease)
- Re-train models at each step for realistic backtesting
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pmdarima import auto_arima
import statsmodels.api as sm
from arch import arch_model
import matplotlib.pyplot as plt
import sys
import os
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
try:
    from utils.data_loader import load_financial_data, preprocess_data, split_train_test
    from utils.config import config
    USE_UTILS = True
except ImportError:
    USE_UTILS = False

# Configuration
TICKER = 'SPY'
YEARS_BACK = 10
TRAIN_RATIO = 0.7
FREQUENCY = 'B'  # Business Day
GARCH_P = 1
GARCH_Q = 1
INITIAL_CAPITAL = 10000

def main():
    """Main function"""
    # Set dates
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365*YEARS_BACK)).strftime('%Y-%m-%d')
    
    # Load and preprocess data
    print(f"{'='*60}")
    print("Step 1: Loading and Preprocessing Data")
    print("="*60)
    if USE_UTILS:
        df_raw = load_financial_data(TICKER, start_date, end_date, progress=False)
        df = preprocess_data(df_raw, frequency=FREQUENCY, use_adjusted=False)
        
        # Extract Close price
        if isinstance(df, pd.Series):
            close_prices = df.copy()
        elif isinstance(df.columns, pd.MultiIndex):
            # MultiIndex columns - extract first Close column
            if 'Close' in df.columns:
                close_col = df['Close']
                if isinstance(close_col, pd.DataFrame):
                    close_prices = close_col.iloc[:, 0] if close_col.shape[1] > 0 else close_col.squeeze()
                else:
                    close_prices = close_col
            else:
                close_prices = df.iloc[:, 0].copy()
        elif 'Close' in df.columns:
            close_col = df['Close']
            # If it's a DataFrame (single column), squeeze to Series
            if isinstance(close_col, pd.DataFrame):
                close_prices = close_col.squeeze()
            else:
                close_prices = close_col.copy()
        else:
            # No Close column, use first column
            close_prices = df.iloc[:, 0].copy()
            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices.squeeze()
    else:
        df = yf.download(TICKER, start=start_date, end=end_date, progress=False)
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.asfreq(FREQUENCY)
        df = df.dropna()
        close_prices = df['Close'].copy()
    
    # Ensure it's a Series
    if isinstance(close_prices, pd.DataFrame):
        # If DataFrame, squeeze to Series (works for single column)
        close_prices = close_prices.squeeze()
    
    if not isinstance(close_prices, pd.Series):
        # Try to convert to Series
        if hasattr(close_prices, 'index') and hasattr(close_prices, 'values'):
            values = close_prices.values
            # Handle 2D arrays
            if isinstance(values, np.ndarray) and values.ndim > 1:
                if values.shape[1] == 1:
                    values = values.flatten()
                else:
                    raise ValueError(f"Cannot convert 2D array with shape {values.shape} to 1D Series")
            close_prices = pd.Series(values, index=close_prices.index, name=TICKER)
        else:
            raise ValueError(f"close_prices is not a Series and cannot be converted. Type: {type(close_prices)}")
    
    # Final check: ensure it's a Series
    if not isinstance(close_prices, pd.Series):
        raise ValueError(f"Failed to convert close_prices to Series. Type: {type(close_prices)}")
    
    print(f"Loaded {len(close_prices)} observations")
    print(f"Date range: {close_prices.index.min()} to {close_prices.index.max()}")
    
    # Split data into training and testing sets
    print(f"\n{'='*60}")
    print(f"Step 2: Splitting Data (Train: {TRAIN_RATIO*100:.0f}%, Test: {(1-TRAIN_RATIO)*100:.0f}%)")
    print("="*60)
    if USE_UTILS:
        train_data, test_data = split_train_test(close_prices.to_frame(), train_ratio=TRAIN_RATIO)
        train_close = train_data.iloc[:, 0]
        test_close = test_data.iloc[:, 0]
    else:
        split_index = int(len(close_prices) * TRAIN_RATIO)
        train_close = close_prices.iloc[:split_index]
        test_close = close_prices.iloc[split_index:]
    
    print(f"Training data: {len(train_close)} observations")
    print(f"Test data: {len(test_close)} observations")
    
    # Find optimal ARIMA parameters using auto_arima
    print(f"\n{'='*60}")
    print("Step 3: Finding Optimal ARIMA Parameters")
    print("="*60)
    print("ARIMA(p,d,q) model selection using auto_arima...")
    auto_model = auto_arima(train_close, 
                            seasonal=False,
                            trace=False,  # Set to True for detailed output
                            error_action='ignore', 
                            suppress_warnings=True,
                            stepwise=True)
    
    order = auto_model.order
    print(f"Optimal ARIMA order: {order}")
    print(f"  p (AR order): {order[0]}")
    print(f"  d (differencing order): {order[1]}")
    print(f"  q (MA order): {order[2]}")
    
    # Initialize backtesting variables
    print(f"\n{'='*60}")
    print("Step 4: Starting Backtesting")
    print("="*60)
    print(f"Initial capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Trading strategy: Buy when hybrid_yhat > price, Sell when hybrid_yhat < price")
    print(f"Models are re-trained at each step for realistic backtesting\n")
    
    capital = INITIAL_CAPITAL
    results = []
    position = None
    total_shares = 0
    average_price = 0
    
    # Initialize history with training data
    history = train_close.tolist()
    
    # Perform backtesting on test data (step by step)
    for t in range(len(test_close)):
        # Predict next step using ARIMA model
        model = sm.tsa.ARIMA(history, order=order)
        model_fit = model.fit()
        output = model_fit.forecast(steps=1)
        arima_forecast = float(np.mean(output))
        
        # Update GARCH model on ARIMA residuals
        arima_residuals = model_fit.resid
        garch_model = arch_model(arima_residuals, vol='GARCH', p=GARCH_P, q=GARCH_Q)
        garch_fit = garch_model.fit(disp="off")
        garch_forecast = garch_fit.forecast(horizon=1)
        
        garch_mean_average = float(garch_forecast.mean.values[-1, 0])
        
        # Hybrid forecast: ARIMA prediction + GARCH mean adjustment
        hybrid_yhat = arima_forecast + garch_mean_average
        
        # Skip if negative prediction
        if hybrid_yhat < 0:
            actual_price = float(test_close.iloc[t])
            history.append(actual_price)
            total_assets = capital + total_shares * actual_price
            results.append(total_assets)
            continue
        
        actual_price = float(test_close.iloc[t])
        hybrid_yhat = float(hybrid_yhat)
        
        # Trading logic
        shares_to_sell = 0
        shares_to_buy = 0
        
        # Buy signal: model predicts price increase
        if hybrid_yhat > actual_price and capital > 0:
            position = 'long'
            shares_to_buy = capital / actual_price
            total_shares += shares_to_buy
            average_price = actual_price
            capital -= shares_to_buy * actual_price
        
        # Sell signal: model predicts price decrease
        elif hybrid_yhat < actual_price and total_shares > 0:
            shares_to_sell = total_shares
            profit = (actual_price - average_price) * shares_to_sell
            total_shares -= shares_to_sell
            position = None
            capital += shares_to_sell * actual_price
        
        # Calculate total assets
        total_assets = capital + total_shares * actual_price
        results.append(total_assets)
        
        # Logging (every 50 steps to reduce output)
        if t % 50 == 0 or t == len(test_close) - 1:
            date_str = test_close.index[t].strftime('%Y-%m-%d')
            print(f"Date: {date_str:>10} | Hybrid: {hybrid_yhat:>8.2f} | Price: {actual_price:>8.2f} | "
                  f"Capital: ${capital:>10.2f} | Shares: {total_shares:>8.2f} | "
                  f"Assets: ${total_assets:>10.2f} | Position: {str(position):>6}")
        
        # Update history for next iteration
        history.append(actual_price)
    
    # Filter out non-numeric values from results
    results = [x for x in results if isinstance(x, (int, float)) and not np.isnan(x)]
    
    # Calculate performance metrics
    print(f"\n{'='*60}")
    print("Step 5: Performance Analysis")
    print("="*60)
    
    if len(results) < 2:
        print("Error: Insufficient data for performance calculation")
        return
    
    returns = np.diff(results) / results[:-1]
    returns = returns[np.isfinite(returns)]
    
    if len(returns) == 0:
        print("Error: No valid returns calculated")
        return
    
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
    
    # Set simulation dates
    simulation_start_date = test_close.index[0]
    simulation_end_date = test_close.index[-1]
    
    final_assets = float(results[-1]) if results else INITIAL_CAPITAL
    total_return = (final_assets - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    # Calculate annualized return
    simulation_years = (simulation_end_date - simulation_start_date).days / 365.25
    annualized_return = (1 + total_return) ** (1 / simulation_years) - 1 if simulation_years > 0 else 0.0
    
    # Buy and Hold comparison
    initial_price = float(test_close.iloc[0])
    final_price = float(test_close.iloc[-1])
    bnh_return = (final_price - initial_price) / initial_price
    bnh_annualized = (1 + bnh_return) ** (1 / simulation_years) - 1 if simulation_years > 0 else 0.0
    
    print(f"\nSimulation Period: {simulation_start_date.date()} ~ {simulation_end_date.date()}")
    print(f"Simulation Years: {simulation_years:.2f}")
    print(f"\n{'='*60}")
    print("STRATEGY PERFORMANCE:")
    print("="*60)
    print(f"  Final Assets: ${final_assets:,.2f}")
    print(f"  Cumulative Return: {total_return:.2%}")
    print(f"  Annualized Return: {annualized_return:.2%}")
    print(f"  Return Std Dev: {std_return:.4f}")
    print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
    
    print(f"\n{'='*60}")
    print("BUY & HOLD PERFORMANCE:")
    print("="*60)
    print(f"  Initial Price: ${initial_price:.2f}")
    print(f"  Final Price: ${final_price:.2f}")
    print(f"  Cumulative Return: {bnh_return:.2%}")
    print(f"  Annualized Return: {bnh_annualized:.2%}")
    
    print(f"\n{'='*60}")
    print("COMPARISON:")
    print("="*60)
    print(f"  Outperformance: {(total_return - bnh_return):+.2%}")
    print(f"  Annualized Outperformance: {(annualized_return - bnh_annualized):+.2%}")
    
    # Visualization
    print(f"\n{'='*60}")
    print("Step 6: Visualization")
    print("="*60)
    plt.figure(figsize=(14, 8))
    
    # Subplot 1: Capital over time
    plt.subplot(2, 1, 1)
    plt.plot(results, label='Strategy Capital', linewidth=2, color='blue')
    plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', label='Initial Capital', alpha=0.7)
    plt.xlabel('Trade Number', fontsize=11)
    plt.ylabel('Capital (USD)', fontsize=11)
    plt.title(f'ARIMA-GARCH Hybrid Model Backtesting - Capital Over Time', 
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Price and buy/sell signals
    plt.subplot(2, 1, 2)
    plt.plot(test_close.index, test_close.values, label=f'{TICKER} Price', 
             linewidth=1.5, color='black', alpha=0.7)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Price (USD)', fontsize=11)
    plt.title(f'{TICKER} Price During Test Period', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print("Backtesting complete")


if __name__ == "__main__":
    main()

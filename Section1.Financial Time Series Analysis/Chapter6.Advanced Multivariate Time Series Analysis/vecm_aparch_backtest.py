"""
VECM-APARCH Hybrid Model Backtesting

This script performs backtesting of a trading strategy using a VECM-APARCH
hybrid model for multivariate time series trading.

IMPORTANT: Information Leakage Prevention
- Uses data up to time t-1 to predict price at time t
- Models are re-trained at each step using only historical data (walking forward)
- No future data is used in any prediction or optimization step

VECM Model:
ΔY_t = αβ'Y_{t-1} + Γ₁ΔY_{t-1} + ... + Γ_{p-1}ΔY_{t-p+1} + ε_t

where:
- ΔY_t: first difference of Y_t
- α: adjustment coefficients (speed of adjustment to equilibrium)
- β: cointegration vectors (long-run relationships)
- Γᵢ: short-run dynamics coefficients
- ε_t: error terms (residuals)

APARCH Model:
σ^δ_t = ω + Σᵢ₌₁ᵖ αᵢ(|ε_{t-i}| - γᵢε_{t-i})^δ + Σⱼ₌₁ᵠ βⱼσ^δ_{t-j}

where:
- σ^δ_t: conditional variance raised to power δ
- δ: power term (allows for flexible volatility modeling)
- γᵢ: leverage parameters (captures asymmetric effects: negative shocks have larger impact)
- αᵢ: ARCH coefficients (lagged squared errors)
- βⱼ: GARCH coefficients (lagged variances)
- ω: constant term

Hybrid Model:
Ŷ_{t+1} = VECM_forecast + APARCH_mean_adjustment

where:
- VECM_forecast: VECM model prediction for next period
- APARCH_mean_adjustment: APARCH model mean forecast for VECM residuals
- Hybrid combines both mean prediction and volatility adjustment

Trading Strategy:
- Buy when: hybrid_yhat > actual_price (model predicts price increase)
- Sell when: hybrid_yhat < actual_price (model predicts price decrease)
- Dynamic re-optimization: VECM parameters are re-optimized when volatility
  exceeds threshold (using only data up to current time)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VECM
from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
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
TICKERS = ['CL=F', 'YM=F', 'NQ=F', 'DX-Y.NYB', 'GC=F', 'NG=F', '^VIX', 'SPY']
TARGET_INDEX = 'SPY'
YEARS_BACK = 10
TRAIN_RATIO = 0.7
FREQUENCY = 'D'  # Daily frequency
INTERVAL = '1d'
INITIAL_CAPITAL = 10000
MAX_LAGS = 15
VOLATILITY_PERCENTILE = 95
VOLATILITY_MULTIPLIER = 1.65


def find_optimal_aparch_order(residuals, target_index, train_data):
    """
    Find optimal APARCH model order using AIC criterion.
    
    APARCH(p,o,q) model selection:
    - Tests all combinations of p, o, q in range [0, 2)
    - Selects model with minimum AIC
    
    Parameters:
    -----------
    residuals : np.ndarray
        VECM model residuals
    target_index : str
        Target asset symbol
    train_data : pd.DataFrame
        Training data with column names
        
    Returns:
    --------
    tuple: (best_aic, best_order, best_model)
    """
    best_aic = np.inf
    best_order = None
    best_model = None
    
    target_col_idx = train_data.columns.get_loc(target_index)
    target_residuals = residuals[:, target_col_idx]
    
    for p in range(0, 2):
        for o in range(0, 2):
            for q in range(0, 2):
                try:
                    model = arch_model(target_residuals, vol='APARCH', 
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


def initialize_models(train_data, target_index, max_lags=15):
    """
    Initialize VECM and APARCH models with optimal parameters.
    
    Steps:
    1. Find optimal VECM lag order: k_ar_diff_opt
    2. Find optimal cointegration rank: coint_rank_opt
    3. Fit VECM model and extract residuals
    4. Test for ARCH effects in residuals
    5. Find optimal APARCH order
    6. Calculate initial volatility threshold
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    target_index : str
        Target asset symbol
    max_lags : int
        Maximum lags to test for VECM
        
    Returns:
    --------
    tuple: (k_ar_diff_opt, coint_rank_opt, (p_opt, o_opt, q_opt), volatility_threshold)
    """
    print(f"{'='*60}")
    print("Step 1: Finding Optimal VECM Parameters")
    print("="*60)
    
    # Find optimal lag order using AIC
    lag_order = select_order(train_data, maxlags=max_lags, deterministic="colo")
    k_ar_diff_opt = lag_order.aic
    print(f"Optimal lag order (k_ar_diff_opt): {k_ar_diff_opt}")
    
    # Find optimal cointegration rank using Johansen test
    coint_rank_test = select_coint_rank(train_data, det_order=1, 
                                       k_ar_diff=k_ar_diff_opt, method='trace')
    coint_rank_opt = coint_rank_test.rank
    print(f"Optimal cointegration rank: {coint_rank_opt}")
    
    # Fit VECM model
    model = VECM(train_data, k_ar_diff=k_ar_diff_opt, 
                coint_rank=coint_rank_opt, deterministic="colo")
    model_fitted = model.fit()
    
    # Extract residuals for APARCH modeling
    residuals = model_fitted.resid
    
    # ARCH Effect Test
    print(f"\n{'='*60}")
    print("Step 2: Testing for ARCH Effects")
    print("="*60)
    target_col_idx = train_data.columns.get_loc(target_index)
    arch_test = acorr_ljungbox(residuals[:, target_col_idx]**2, lags=10)
    print("ARCH Effect Test Result:")
    print(arch_test)
    
    # Find optimal APARCH order
    print(f"\n{'='*60}")
    print("Step 3: Finding Optimal APARCH Order")
    print("="*60)
    best_aic, best_order, best_model = find_optimal_aparch_order(
        residuals, target_index, train_data)
    print(f"Best AIC: {best_aic:.4f}")
    print(f"Best APARCH order (p, o, q): {best_order}")
    
    # Fit APARCH model with optimal order
    p_opt, o_opt, q_opt = best_order
    garch_model = arch_model(residuals[:, target_col_idx], 
                            vol='APARCH', p=p_opt, o=o_opt, q=q_opt, rescale=True)
    garch_fit = garch_model.fit(disp='off')
    
    # Calculate initial volatility threshold
    garch_forecast = garch_fit.forecast(horizon=1, start=1)
    garch_volatility = np.sqrt(garch_forecast.variance.values)
    volatility_threshold = np.percentile(garch_volatility, VOLATILITY_PERCENTILE) * VOLATILITY_MULTIPLIER
    print(f"\nInitial volatility threshold: {volatility_threshold:.4f}")
    
    return k_ar_diff_opt, coint_rank_opt, (p_opt, o_opt, q_opt), volatility_threshold


def calculate_hybrid_forecast(history, k_ar_diff_opt, coint_rank_opt, 
                             aparch_order, target_index):
    """
    Calculate hybrid VECM-APARCH forecast.
    
    Process:
    1. Fit VECM model on history
    2. Generate VECM prediction: Ŷ_{t+1} = VECM_forecast
    3. Extract VECM residuals
    4. Fit APARCH model on residuals
    5. Generate APARCH mean adjustment
    6. Hybrid forecast: Ŷ_{t+1} = VECM_forecast + APARCH_mean
    
    Parameters:
    -----------
    history : pd.DataFrame
        Historical data up to time t-1
    k_ar_diff_opt : int
        Optimal VECM lag order
    coint_rank_opt : int
        Optimal cointegration rank
    aparch_order : tuple
        (p, o, q) APARCH order
    target_index : str
        Target asset symbol
        
    Returns:
    --------
    tuple: (hybrid_yhat, garch_volatility_current, model_fitted, garch_fit)
    """
    # Fit VECM model
    model = VECM(history, k_ar_diff=k_ar_diff_opt, 
                coint_rank=coint_rank_opt, deterministic="colo")
    model_fitted = model.fit()
    
    # Generate VECM prediction: multiple steps and average
    # This provides more stable predictions
    output = model_fitted.predict(steps=k_ar_diff_opt)
    predicted_mean = output.mean(axis=0)[history.columns.get_loc(target_index)]
    
    # Extract residuals and fit APARCH
    target_col_idx = history.columns.get_loc(target_index)
    residuals = model_fitted.resid[:, target_col_idx]
    
    p_opt, o_opt, q_opt = aparch_order
    garch_model = arch_model(residuals, vol='APARCH', 
                            p=p_opt, o=o_opt, q=q_opt, rescale=True)
    garch_fit = garch_model.fit(disp='off')
    
    # Forecast volatility and mean
    garch_forecast = garch_fit.forecast(horizon=1, start=len(residuals)-p_opt)
    garch_volatility_current = float(np.sqrt(garch_forecast.variance.values[-1, 0]))
    garch_mean_average = float(np.mean(garch_forecast.mean.values))
    
    # Hybrid forecast
    hybrid_yhat = predicted_mean + garch_mean_average
    
    return hybrid_yhat, garch_volatility_current, model_fitted, garch_fit


def run_backtest(train_data, test_data, k_ar_diff_opt, coint_rank_opt, 
                 aparch_order, volatility_threshold, target_index, initial_capital):
    """
    Run backtesting simulation.
    
    CRITICAL: Information Leakage Prevention
    - At time t: Use data up to t-1 to predict price at t
    - Compare prediction with actual price at t
    - Then add price at t to history for next iteration
    
    Trading Logic:
    - Buy when: hybrid_yhat > actual_price (predicts price increase)
    - Sell when: hybrid_yhat < actual_price (predicts price decrease)
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    test_data : pd.DataFrame
        Test data for backtesting
    k_ar_diff_opt : int
        Optimal VECM lag order
    coint_rank_opt : int
        Optimal cointegration rank
    aparch_order : tuple
        (p, o, q) APARCH order
    volatility_threshold : float
        Volatility threshold for re-optimization
    target_index : str
        Target asset symbol
    initial_capital : float
        Initial capital
        
    Returns:
    --------
    tuple: (results, final_capital, total_shares, capital, position)
    """
    from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank
    
    print(f"\n{'='*60}")
    print("Step 4: Starting Backtesting")
    print("="*60)
    print(f"Initial capital: ${initial_capital:,.2f}")
    print(f"Test period: {len(test_data)} days")
    print(f"Trading strategy: Buy when hybrid_yhat > price, Sell when hybrid_yhat < price")
    print(f"Models re-trained at each step (walking forward)\n")
    
    capital = initial_capital
    results = []
    position = None
    average_price = 0
    total_shares = 0
    
    # Initialize history with training data only
    history = train_data.copy()
    
    for t in range(len(test_data)):
        # CRITICAL: Use data up to t-1 to predict price at t
        # Do NOT add test_data[t] to history before prediction
        
        # Calculate hybrid forecast using history (up to t-1)
        hybrid_yhat, garch_volatility_current, model_fitted, garch_fit = \
            calculate_hybrid_forecast(history, k_ar_diff_opt, coint_rank_opt, 
                                   aparch_order, target_index)
        
        # Skip if negative prediction
        if hybrid_yhat < 0:
            # Still need to add current price to history for next iteration
            history = pd.concat([history, test_data.iloc[[t]]])
            actual_price = float(test_data[target_index].iloc[t])
            total_assets = capital + total_shares * actual_price
            results.append(total_assets)
            continue
        
        # Re-optimize if volatility exceeds threshold
        if garch_volatility_current > volatility_threshold:
            # Re-optimize using only current history (no future data)
            coint_rank_opt = select_coint_rank(history, det_order=1, 
                                             k_ar_diff=k_ar_diff_opt, method='trace').rank
            k_ar_diff_opt = select_order(history, maxlags=MAX_LAGS, 
                                        deterministic="colo").aic
            print(f"[Re-optimization at t={t}] k_ar_diff_opt: {k_ar_diff_opt}, "
                  f"coint_rank_opt: {coint_rank_opt}")
            
            # Recalculate volatility threshold
            garch_forecast_all = garch_fit.forecast(horizon=1, start=1)
            garch_volatility_all = np.sqrt(garch_forecast_all.variance.values.flatten())
            volatility_threshold = float(np.percentile(garch_volatility_all, 
                                                      VOLATILITY_PERCENTILE) * 
                                       VOLATILITY_MULTIPLIER)
            print(f"[Re-optimization] Updated volatility_threshold: {volatility_threshold:.4f}")
        
        # Get actual price at time t (this is what we're trying to predict)
        actual_price = float(test_data[target_index].iloc[t])
        
        # Trading logic
        shares_to_sell = 0
        shares_to_buy = 0
        
        # Buy signal: model predicts price increase
        if hybrid_yhat > actual_price and capital > 0:
            position = 'long'
            shares_to_buy = capital / actual_price
            if shares_to_buy * actual_price <= capital:
                total_shares += shares_to_buy
                average_price = (average_price * (total_shares - shares_to_buy) + 
                               actual_price * shares_to_buy) / total_shares
                capital -= shares_to_buy * actual_price
        
        # Sell signal: model predicts price decrease
        elif hybrid_yhat < actual_price and total_shares > 0:
            shares_to_sell = total_shares
            profit = (actual_price - average_price) * shares_to_sell
            total_shares -= shares_to_sell
            
            if total_shares <= 0:
                position = None
            
            capital += shares_to_sell * actual_price
        
        # Calculate total assets
        total_assets = capital + total_shares * actual_price
        results.append(total_assets)
        
        # Logging (every 50 steps to reduce output)
        if t % 50 == 0 or t == len(test_data) - 1:
            date_str = test_data.index[t].strftime('%Y-%m-%d')
            print(f"t={t:4d} | Date: {date_str:>10} | Hybrid: {hybrid_yhat:>8.2f} | "
                  f"Price: {actual_price:>8.2f} | Capital: ${capital:>10.2f} | "
                  f"Shares: {total_shares:>8.2f} | Assets: ${total_assets:>10.2f} | "
                  f"Position: {str(position):>6}")
        
        # NOW add current price to history for next iteration
        # This ensures we use data up to t-1 to predict t
        history = pd.concat([history, test_data.iloc[[t]]])
    
    return results, total_assets, total_shares, capital, position


def calculate_performance_metrics(results, test_data, target_index, 
                                  initial_capital, total_assets):
    """
    Calculate performance metrics and compare with buy-and-hold.
    
    Metrics:
    - Cumulative return
    - Annualized return
    - Sharpe ratio
    - Buy-and-hold comparison
    
    Parameters:
    -----------
    results : list
        List of total assets over time
    test_data : pd.DataFrame
        Test data
    target_index : str
        Target asset symbol
    initial_capital : float
        Initial capital
    total_assets : float
        Final total assets
        
    Returns:
    --------
    dict: Performance metrics
    """
    # Filter valid results
    results = [x for x in results if isinstance(x, (int, float)) and not np.isnan(x)]
    
    if len(results) < 2:
        print("Error: Insufficient data for performance calculation")
        return None
    
    # Calculate returns
    returns = np.diff(results) / results[:-1]
    returns = returns[np.isfinite(returns)]
    
    if len(returns) == 0:
        print("Error: No valid returns calculated")
        return None
    
    # Strategy metrics
    total_return = (total_assets - initial_capital) / initial_capital
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    
    # Annualized metrics
    trading_days_per_year = 252
    annualized_return_simple = mean_return * trading_days_per_year
    annualized_std_return = std_return * np.sqrt(trading_days_per_year)
    sharpe_ratio = annualized_return_simple / annualized_std_return if annualized_std_return > 0 else 0.0
    
    # Time-based annualized return
    simulation_start_date = test_data.index[0]
    simulation_end_date = test_data.index[-1]
    simulation_years = (simulation_end_date - simulation_start_date).days / 365.25
    annualized_return = (1 + total_return) ** (1 / simulation_years) - 1 if simulation_years > 0 else 0.0
    
    # Buy-and-hold comparison
    initial_price = float(test_data[target_index].iloc[0])
    final_price = float(test_data[target_index].iloc[-1])
    bnh_return = (final_price - initial_price) / initial_price
    bnh_annualized = (1 + bnh_return) ** (1 / simulation_years) - 1 if simulation_years > 0 else 0.0
    
    metrics = {
        'total_assets': total_assets,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_std_return': annualized_std_return,
        'sharpe_ratio': sharpe_ratio,
        'simulation_start_date': simulation_start_date,
        'simulation_end_date': simulation_end_date,
        'simulation_years': simulation_years,
        'bnh_return': bnh_return,
        'bnh_annualized': bnh_annualized,
        'outperformance': total_return - bnh_return,
        'annualized_outperformance': annualized_return - bnh_annualized
    }
    
    return metrics


def print_performance_report(metrics, target_index):
    """Print performance report."""
    print(f"\n{'='*60}")
    print("Step 5: Performance Analysis")
    print("="*60)
    
    print(f"\nSimulation Period: {metrics['simulation_start_date'].date()} ~ "
          f"{metrics['simulation_end_date'].date()}")
    print(f"Simulation Years: {metrics['simulation_years']:.2f}")
    
    print(f"\n{'='*60}")
    print("STRATEGY PERFORMANCE:")
    print("="*60)
    print(f"  Final Assets: ${metrics['total_assets']:,.2f}")
    print(f"  Cumulative Return: {metrics['total_return']:.2%}")
    print(f"  Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"  Return Std Dev: {metrics['annualized_std_return']:.4f}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    
    print(f"\n{'='*60}")
    print("BUY & HOLD PERFORMANCE:")
    print("="*60)
    print(f"  Cumulative Return: {metrics['bnh_return']:.2%}")
    print(f"  Annualized Return: {metrics['bnh_annualized']:.2%}")
    
    print(f"\n{'='*60}")
    print("COMPARISON:")
    print("="*60)
    print(f"  Outperformance: {metrics['outperformance']:+.2%}")
    print(f"  Annualized Outperformance: {metrics['annualized_outperformance']:+.2%}")


def visualize_results(results, test_data, target_index, initial_capital):
    """Visualize backtesting results."""
    print(f"\n{'='*60}")
    print("Step 6: Visualization")
    print("="*60)
    
    plt.figure(figsize=(14, 8))
    
    # Subplot 1: Capital over time
    plt.subplot(2, 1, 1)
    plt.plot(results, label='Strategy Capital', linewidth=2, color='blue')
    plt.axhline(y=initial_capital, color='r', linestyle='--', 
               label='Initial Capital', alpha=0.7)
    plt.xlabel('Trade Number', fontsize=11)
    plt.ylabel('Capital (USD)', fontsize=11)
    plt.title('VECM-APARCH Hybrid Model Backtesting - Capital Over Time', 
             fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Price and buy/sell signals
    plt.subplot(2, 1, 2)
    plt.plot(test_data.index, test_data[target_index].values, 
            label=f'{target_index} Price', linewidth=1.5, color='black', alpha=0.7)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Price (USD)', fontsize=11)
    plt.title(f'{target_index} Price During Test Period', 
             fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print("Visualization complete")


def main():
    """Main function"""
    # Set dates
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365*YEARS_BACK)).strftime('%Y-%m-%d')
    
    # Load and preprocess data
    print(f"{'='*60}")
    print("Loading and Preprocessing Data")
    print("="*60)
    
    if USE_UTILS:
        df_raw = load_financial_data(TICKERS, start_date, end_date, 
                                    interval=INTERVAL, progress=False)
        df = preprocess_data(df_raw, frequency=FREQUENCY, use_adjusted=True)
        
        # Extract Close prices
        if isinstance(df, pd.DataFrame):
            if 'Close' in df.columns:
                close_data = df['Close']
            else:
                close_data = df
        else:
            close_data = df
    else:
        df = yf.download(TICKERS, start=start_date, end=end_date, 
                        auto_adjust=True, interval=INTERVAL, progress=False)
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.asfreq(FREQUENCY)
        df = df[(df >= 0).all(axis=1)]
        df = df.dropna()
        close_data = df['Close']
    
    # Ensure it's a DataFrame
    if isinstance(close_data, pd.Series):
        close_data = close_data.to_frame()
    
    print(f"Loaded {len(close_data)} observations")
    print(f"Date range: {close_data.index.min()} to {close_data.index.max()}")
    print(f"Assets: {list(close_data.columns)}")
    
    # Split data
    print(f"\n{'='*60}")
    print(f"Splitting Data (Train: {TRAIN_RATIO*100:.0f}%, Test: {(1-TRAIN_RATIO)*100:.0f}%)")
    print("="*60)
    
    if USE_UTILS:
        train_data, test_data = split_train_test(close_data, train_ratio=TRAIN_RATIO)
    else:
        split_index = int(len(close_data) * TRAIN_RATIO)
        train_data = close_data.iloc[:split_index]
        test_data = close_data.iloc[split_index:]
    
    print(f"Training data: {len(train_data)} observations")
    print(f"Test data: {len(test_data)} observations")
    
    # Initialize models
    k_ar_diff_opt, coint_rank_opt, aparch_order, volatility_threshold = \
        initialize_models(train_data, TARGET_INDEX, MAX_LAGS)
    
    # Run backtest
    results, total_assets, total_shares, capital, position = \
        run_backtest(train_data, test_data, k_ar_diff_opt, coint_rank_opt, 
                    aparch_order, volatility_threshold, TARGET_INDEX, INITIAL_CAPITAL)
    
    # Calculate performance
    metrics = calculate_performance_metrics(results, test_data, TARGET_INDEX, 
                                          INITIAL_CAPITAL, total_assets)
    
    if metrics:
        print_performance_report(metrics, TARGET_INDEX)
        visualize_results(results, test_data, TARGET_INDEX, INITIAL_CAPITAL)
    
    print("\nBacktesting complete")


if __name__ == "__main__":
    main()

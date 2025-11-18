"""
Dynamic Bitcoin Trading Simulation with VECM-EGARCH Hybrid Model

This script implements a dynamic trading simulation for Bitcoin using VECM-EGARCH
hybrid approach. Similar to Chapter1 but optimized for Bitcoin trading.

VECM Model:
ΔY_t = αβ'Y_{t-1} + Γ₁ΔY_{t-1} + ... + Γ_{p-1}ΔY_{t-p+1} + ε_t

EGARCH Model:
log(σ²_t) = ω + Σᵢ₌₁ᵖ (αᵢ|z_{t-i}| + γᵢz_{t-i}) + Σⱼ₌₁ᵠ βⱼlog(σ²_{t-j})

Trading Strategy:
- Long/Short positions based on hybrid forecast
- Commission fees: 0.02% (0.0002)
- Dynamic re-optimization when volatility exceeds threshold
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VECM
from arch import arch_model
import sys
import os
import warnings
from datetime import datetime, timedelta
from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank

warnings.filterwarnings("ignore")

# Add parent directory to path - ensure proper import
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from utils module
from utils.data_loader import load_financial_data, preprocess_data, split_train_test
from utils.model_utils import find_garch, initialize_vecm_models
from utils.metrics import max_drawdown, max_loss, calculate_performance_metrics

# Configuration
TICKERS = ['NQ=F', 'GC=F', 'SI=F', 'NG=F', 'BTC-USD']
TARGET_INDEX = "BTC-USD"
YEARS_BACK = 10
TRAIN_RATIO = 0.7
INTERVAL = '1d'
FREQUENCY = 'D'
PRICE_COLUMN = 'Open'
INITIAL_CAPITAL = 10000
FRACTION = 0.3
COMMISSION_RATE = 0.0002  # 0.02%
MAX_LAGS = 15
VOLATILITY_PERCENTILE = 95
VOLATILITY_MULTIPLIER = 1.65


def initialize_models(train_data, target_index):
    """Initialize VECM and EGARCH models with optimal parameters."""
    print(f"{'='*60}")
    print("Step 1: Initializing Models")
    print("="*60)
    
    # Initialize VECM models
    k_ar_diff_opt, coint_rank_opt = initialize_vecm_models(train_data, MAX_LAGS)
    print(f"Optimal lag order (k_ar_diff_opt): {k_ar_diff_opt}")
    print(f"Optimal cointegration rank: {coint_rank_opt}")
    
    # Fit VECM model
    model = VECM(train_data, k_ar_diff=k_ar_diff_opt, 
                coint_rank=coint_rank_opt, deterministic="colo")
    model_fitted = model.fit()
    
    # Find optimal EGARCH order
    target_col_idx = train_data.columns.get_loc(target_index)
    residuals = model_fitted.resid[:, target_col_idx]
    best_aic, best_order, best_model = find_garch(residuals, vol_type='EGARCH')
    print(f"Best EGARCH AIC: {best_aic:.4f}")
    print(f"Best EGARCH order (p, o, q): {best_order}")
    
    # Fit EGARCH model
    p_opt, o_opt, q_opt = best_order
    garch_model = arch_model(residuals, vol='EGARCH', 
                            p=p_opt, o=o_opt, q=q_opt, rescale=True)
    garch_fit = garch_model.fit(disp='off')
    
    # Calculate volatility threshold
    garch_forecast = garch_fit.forecast(horizon=1, start=1)
    garch_volatility = np.sqrt(garch_forecast.variance.values)
    volatility_threshold = np.percentile(garch_volatility, VOLATILITY_PERCENTILE) * VOLATILITY_MULTIPLIER
    print(f"Volatility threshold: {volatility_threshold:.4f}")
    
    return (k_ar_diff_opt, coint_rank_opt, (p_opt, o_opt, q_opt), 
            volatility_threshold, model_fitted, garch_fit)


def main():
    """Main function"""
    # Set dates
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365*YEARS_BACK)).strftime('%Y-%m-%d')
    
    # Load and preprocess data
    print(f"{'='*60}")
    print("Loading and Preprocessing Data")
    print("="*60)
    
    # Load and preprocess data using utils
    df = load_financial_data(TICKERS, start_date, end_date, 
                            interval=INTERVAL, auto_adjust=False, progress=False)
    df = preprocess_data(df, frequency=FREQUENCY)
    train_data, test_data, ohlc_data = split_train_test(df, TRAIN_RATIO, PRICE_COLUMN)
    
    print(f"Training data: {len(train_data)} observations")
    print(f"Test data: {len(test_data)} observations")
    
    # Initialize models
    (k_ar_diff_opt, coint_rank_opt, egarch_order, volatility_threshold, 
     initial_model_fitted, initial_garch_fit) = initialize_models(train_data, TARGET_INDEX)
    
    p_opt, o_opt, q_opt = egarch_order
    
    # Initialize simulation variables
    print(f"\n{'='*60}")
    print("Step 2: Starting Simulation")
    print("="*60)
    print(f"Initial capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Position fraction: {FRACTION}")
    print(f"Commission rate: {COMMISSION_RATE:.4f} ({COMMISSION_RATE*100:.2f}%)")
    
    capital = INITIAL_CAPITAL
    total_shares = 0
    average_price = 0
    position = None
    results = []
    trade_history = []
    simulation_start_date = test_data.index[0]
    simulation_end_date = test_data.index[-1]
    history = train_data.copy()
    cumulative_commission = 0
    
    # Main simulation loop
    print(f"\n{'='*60}")
    print("Step 3: Running Simulation")
    print("="*60)
    
    for t in range(len(test_data)):
        # Note: We use Open price for analysis, which is known at the start of day t.
        # The trading decision compares prediction with Open price, and execution uses
        # Low/High/Close prices. Since Open is known before Close, adding Open to history
        # before prediction does not cause information leakage when comparing with Close.
        # 
        # Analysis basis: Open price - information known at the start of day t
        # Trading decision: Compare prediction vs Open price
        # Actual trading: Use Low/High/Close prices
        # Therefore, adding Open to history does not cause information leakage
        # when comparing with Close for trading decisions
        
        # Add previous time step's Open price to history (if not first iteration)
        # Safe to add Open price since it's known before Close (no information leakage)
        if t > 0:
            history = pd.concat([history, test_data.iloc[[t-1]]])
        
        # Calculate forecast using history
        model = VECM(history, k_ar_diff=k_ar_diff_opt, coint_rank=coint_rank_opt, deterministic="colo")
        model_fitted = model.fit()
        output, lower_bound, upper_bound = model_fitted.predict(steps=k_ar_diff_opt, alpha=FRACTION)

        # Calculate mean of lower and upper bounds
        target_col_idx = history.columns.get_loc(TARGET_INDEX)
        lower_mean = float(lower_bound.mean(axis=0)[target_col_idx])
        upper_mean = float(upper_bound.mean(axis=0)[target_col_idx])
        predicted_mean = float(output.mean(axis=0)[target_col_idx])

        # Update EGARCH model
        residuals = model_fitted.resid[:, target_col_idx]
        garch_model = arch_model(residuals, vol='EGARCH', p=p_opt, o=o_opt, q=q_opt, rescale=True)
        garch_fit = garch_model.fit(disp='off')
        garch_forecast = garch_fit.forecast(horizon=1, start=len(residuals)-p_opt)
        garch_volatility_current = float(np.sqrt(garch_forecast.variance.values[-1, 0]))
        garch_mean_average = float(np.mean(garch_forecast.mean.values))
        
        # Hybrid forecast
        hybrid_yhat = predicted_mean + garch_mean_average
        
        # Convert to scalar if needed
        if isinstance(hybrid_yhat, np.ndarray):
            hybrid_yhat = float(hybrid_yhat.item())
        else:
            hybrid_yhat = float(hybrid_yhat)

        # Get current Open price - known at start of day t
        actual_price = float(test_data[TARGET_INDEX].iloc[t])
        
        # Skip trade if prediction is negative
        if hybrid_yhat < 0:
            total_assets = capital + abs(total_shares) * actual_price
            results.append(total_assets)
            # History will be updated at the end of loop iteration
            continue
        
        # Dynamic re-optimization if volatility exceeds threshold
        if garch_volatility_current > volatility_threshold:
            # Re-optimize using only current history (no future data)
            coint_rank_opt = select_coint_rank(history, det_order=1, 
                                             k_ar_diff=k_ar_diff_opt, method='trace').rank
            k_ar_diff_opt = select_order(history, maxlags=MAX_LAGS, deterministic="colo").aic
            print(f"[Re-optimization at t={t}] k_ar_diff_opt: {k_ar_diff_opt}, coint_rank_opt: {coint_rank_opt}")

            # Update EGARCH model
            best_aic, best_order, best_model = find_garch(residuals, vol_type='EGARCH')
            p_opt, o_opt, q_opt = best_order
            print(f"[Re-optimization] Best EGARCH AIC: {best_aic:.4f}, Order: {best_order}")

            # Recalculate volatility threshold
            garch_forecast_all = garch_fit.forecast(horizon=1, start=1)
            garch_volatility_all = np.sqrt(garch_forecast_all.variance.values.flatten())
            volatility_threshold = float(np.percentile(garch_volatility_all, VOLATILITY_PERCENTILE) * 
                                       VOLATILITY_MULTIPLIER)
            print(f"[Re-optimization] Updated volatility_threshold: {volatility_threshold:.4f}")

        # Get OHLC prices for trading execution
        # Open price is already retrieved above, used for prediction comparison
        lower_price = float(ohlc_data['Low'][TARGET_INDEX].iloc[t])   # Low price (for condition check only)
        upper_price = float(ohlc_data['High'][TARGET_INDEX].iloc[t])  # High price (for condition check only)
        close_price = float(ohlc_data['Close'][TARGET_INDEX].iloc[t])  # Close price

        # Trading logic
        shares_to_sell = 0
        shares_to_buy = 0

        # Set new LONG position
        if hybrid_yhat > actual_price and capital > 0 and lower_price < lower_mean:
            if position == 'short':
                # Close existing short position
                # [INFO LEAKAGE - OLD CODE] Using lower_price (actual low of day) - not known at decision time
                # shares_to_sell = total_shares
                # profit = (average_price - lower_price) * shares_to_sell
                # commission = shares_to_sell * actual_price * COMMISSION_RATE
                # cumulative_commission += commission
                # capital += profit - commission + shares_to_sell * actual_price
                
                # [FIXED] Use lower_mean (predicted lower barrier) for execution price
                # We cannot know the actual low price of the day in advance,
                # so we use the predicted lower barrier as the execution price
                shares_to_sell = total_shares
                execution_price = lower_mean  # Use predicted lower barrier as execution price
                profit = (average_price - execution_price) * shares_to_sell
                commission = shares_to_sell * execution_price * COMMISSION_RATE
                cumulative_commission += commission
                capital += profit - commission + shares_to_sell * execution_price
                total_shares = 0
                average_price = 0
                position = None

            # Set new long position
            # [INFO LEAKAGE - OLD CODE] Using lower_price (actual low of day) - not known at decision time
            # position = 'long'
            # shares_to_buy = (capital * FRACTION) / lower_price
            # if shares_to_buy * lower_price <= capital:
            #     total_value = (average_price * total_shares) + (lower_price * shares_to_buy)
            #     total_shares += shares_to_buy
            #     average_price = total_value / total_shares
            #     commission = shares_to_buy * lower_price * COMMISSION_RATE
            #     cumulative_commission += commission
            #     capital -= shares_to_buy * lower_price + commission
            
            # [FIXED] Use lower_mean (predicted lower barrier) for execution price
            # We cannot know the actual low price of the day in advance,
            # so we use the predicted lower barrier as the execution price
            # Condition: lower_price < lower_mean (buy when actual low is below predicted barrier)
            position = 'long'
            execution_price = lower_mean  # Use predicted lower barrier as execution price
            shares_to_buy = (capital * FRACTION) / execution_price
            if shares_to_buy * execution_price <= capital:
                total_value = (average_price * total_shares) + (execution_price * shares_to_buy)
                total_shares += shares_to_buy
                average_price = total_value / total_shares
                commission = shares_to_buy * execution_price * COMMISSION_RATE
                cumulative_commission += commission
                capital -= shares_to_buy * execution_price + commission

        # Set new SHORT position
        elif hybrid_yhat < actual_price and capital > 0 and upper_price > upper_mean:
            if position == 'long':
                # Close existing long position
                # [INFO LEAKAGE - OLD CODE] Using upper_price (actual high of day) - not known at decision time
                # shares_to_sell = total_shares
                # profit = (upper_price - average_price) * shares_to_sell
                # commission = shares_to_sell * actual_price * COMMISSION_RATE
                # cumulative_commission += commission
                # capital += profit - commission + shares_to_sell * upper_price
                
                # [FIXED] Use upper_mean (predicted upper barrier) for execution price
                # We cannot know the actual high price of the day in advance,
                # so we use the predicted upper barrier as the execution price
                shares_to_sell = total_shares
                execution_price = upper_mean  # Use predicted upper barrier as execution price
                profit = (execution_price - average_price) * shares_to_sell
                commission = shares_to_sell * execution_price * COMMISSION_RATE
                cumulative_commission += commission
                capital += profit - commission + shares_to_sell * execution_price
                total_shares = 0
                average_price = 0
                position = None

            # Set new short position
            # [INFO LEAKAGE - OLD CODE] Using upper_price (actual high of day) - not known at decision time
            # position = 'short'
            # shares_to_sell = (capital * FRACTION) / upper_price
            # if shares_to_sell * upper_price <= capital:
            #     total_value = (average_price * abs(total_shares)) + (upper_price * shares_to_sell)
            #     total_shares += shares_to_sell
            #     average_price = total_value / abs(total_shares)
            #     commission = shares_to_sell * upper_price * COMMISSION_RATE
            #     cumulative_commission += commission
            #     capital -= shares_to_sell * upper_price + commission
            
            # [FIXED] Use upper_mean (predicted upper barrier) for execution price
            # We cannot know the actual high price of the day in advance,
            # so we use the predicted upper barrier as the execution price
            # Condition: upper_price > upper_mean (sell when actual high is above predicted barrier)
            position = 'short'
            execution_price = upper_mean  # Use predicted upper barrier as execution price
            shares_to_sell = (capital * FRACTION) / execution_price
            if shares_to_sell * execution_price <= capital:
                total_value = (average_price * abs(total_shares)) + (execution_price * shares_to_sell)
                total_shares += shares_to_sell
                average_price = total_value / abs(total_shares)
                commission = shares_to_sell * execution_price * COMMISSION_RATE
                cumulative_commission += commission
                capital -= shares_to_sell * execution_price + commission

        # Calculate unrealized PnL
        if position == 'long':
            unrealized_pnl = (actual_price - average_price) * total_shares
        elif position == 'short':
            unrealized_pnl = (average_price - actual_price) * total_shares
        else:
            unrealized_pnl = 0

        # Calculate total assets
        total_assets = capital + abs(total_shares) * actual_price + unrealized_pnl
        results.append(total_assets)

        # Logging (every 50 steps to reduce output)
        if t % 50 == 0 or t == len(test_data) - 1:
            date_str = test_data.index[t].strftime('%Y-%m-%d')
            print(f"t={t:4d} | Date: {date_str:>10} | Hybrid: {hybrid_yhat:>8.2f} | "
                  f"Price: {actual_price:>8.2f} | Capital: ${capital:>10.2f} | "
                  f"Shares: {total_shares:>8.2f} | Assets: ${total_assets:>10.2f} | "
                  f"Position: {str(position):>6} | PnL: {unrealized_pnl:>10.2f} | "
                  f"Commission: ${cumulative_commission:>10.2f}")

        # Add trade details to history
        date_str = test_data.index[t].strftime('%Y-%m-%d')
        trade_history.append({
            'date': date_str,
            'hybrid_yhat': hybrid_yhat,
            'actual_price': actual_price,  # Open price
            'capital': capital,
            'total_shares': total_shares,
            'total_assets': total_assets,
            'position': position,
            'unrealized_pnl': unrealized_pnl,
            'cumulative_commission': cumulative_commission
        })
        
        # Note: History is updated at the start of next iteration (t+1)
        # No information leakage since we use Open price (known before Close)
    
    # Calculate and display performance metrics
    print(f"\n{'='*60}")
    print("Step 4: Performance Analysis")
    print("="*60)
    
    # Save trade history
    if trade_history:
        trade_history_df = pd.DataFrame(trade_history)
        # Save to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, 'trade_history_btc.xlsx')
        trade_history_df.to_excel(output_file, index=False)
        print(f"Trade history saved to '{output_file}'")
    
    # Calculate performance metrics using utils
    metrics = calculate_performance_metrics(results, INITIAL_CAPITAL)
    if metrics:
        print(f"\nSimulation Period: {simulation_start_date.date()} ~ {simulation_end_date.date()}")
        print(f"Final Capital: ${metrics['final_capital']:,.2f}")
        print(f"Total Commission Paid: ${cumulative_commission:,.2f}")
        print(f"Cumulative Return: {metrics['total_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Return Standard Deviation: {metrics['annualized_std']:.4f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Peak: ${metrics['peak']:,.2f}")
        print(f"Maximum Drawdown: ${metrics['max_drawdown']:,.2f}")
        print(f"Maximum Consecutive Loss Days: {metrics['max_consecutive_loss_days']}")
    
    # Buy-and-hold comparison
    initial_price = float(test_data[TARGET_INDEX].iloc[0])
    final_price = float(test_data[TARGET_INDEX].iloc[-1])
    bnh_return = (final_price - initial_price) / initial_price
    print(f"\nBuy & Hold ({TARGET_INDEX}) Return: {bnh_return:.2%}")
    
    # Visualization
    print(f"\n{'='*60}")
    print("Step 5: Visualization")
    print("="*60)
    plt.figure(figsize=(12, 6))
    plt.plot(results, label='Strategy Capital', linewidth=2, color='blue')
    plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', 
                 label='Initial Capital', alpha=0.7)
    plt.xlabel('Trade Number', fontsize=11)
    plt.ylabel('Capital (USD)', fontsize=11)
    plt.title('Bitcoin Trading Simulation - Capital Over Time', 
             fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save plot to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_file = os.path.join(script_dir, 'simulation_results_btc.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to '{plot_file}'")
    plt.show()
    
    print("Simulation complete")


if __name__ == "__main__":
    main()


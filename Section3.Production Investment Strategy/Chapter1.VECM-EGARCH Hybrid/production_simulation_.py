import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VECM
from arch import arch_model
import warnings
from datetime import datetime, timedelta
import os

# Import utility functions

from utils.vecm_utils import fit_vecm_model, get_alpha_value, get_vecm_confidence
from utils.garch_utils import find_garch
from utils.confidence_utils import calculate_adaptive_threshold, normalize_confidence_to_fraction
from utils.performance_utils import max_drawdown, max_loss

# Ignore warnings
warnings.filterwarnings("ignore")

# Set today's date
end_date = (datetime.today()).strftime('%Y-%m-%d')

# Set start date based on the interval
interval = '1d'  # Daily interval
start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')  

# List of tickers
tickers = ['TQQQ', 'NQ=F', 'ZB=F', 'UNG', 'RB=F', 'BZ=F', 'CL=F']
target_index = "TQQQ"

# Download data
df = yf.download(tickers, start=start_date, end=end_date, 
                 interval=interval, auto_adjust=True, progress=False)
df.reset_index(inplace=True)

# Set index to Date and add frequency information
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.asfreq('D')  # 'D' means daily frequency

# Remove rows with negative values
df = df[(df >= 0).all(axis=1)]

# Remove NaN values
df = df.dropna()

# Split data into training and testing sets
split_index = int(len(df) * 0.7)  # Use 70% for model estimation
train_data = df['Open'].iloc[:split_index]
test_data = df['Open'].iloc[split_index:]
ohlc_data = df.iloc[split_index:]


model, model_fitted, k_ar_diff_opt, coint_rank_opt = fit_vecm_model(train_data, max_lags=15, deterministic="colo")
print(f"Optimal lag order: {k_ar_diff_opt}")
print(f"Optimal cointegration rank: {coint_rank_opt}")

# Alpha calculation method setting (theoretically most accurate method)
ALPHA_METHOD = 'weighted_mean'  # 'weighted_mean', 'sum_negative', 'min_negative'

# Calculate initial alpha value from training data
initial_alpha = get_alpha_value(model_fitted, target_index, train_data, method=ALPHA_METHOD)
print(f"Initial ECT alpha value: {initial_alpha:.6f} (method: {ALPHA_METHOD})")

# Debug: Print alpha values for all cointegration relations
try:
    alpha = model_fitted.alpha
    target_idx = train_data.columns.get_loc(target_index)
    if alpha.ndim == 2:
        print(f"\n[Debug] Alpha matrix shape: {alpha.shape}")
        print(f"[Debug] Target variable index: {target_idx} ({target_index})")
        print(f"[Debug] All alpha values for {target_index}:")
        negative_alphas = []
        for coint_idx in range(alpha.shape[1]):
            alpha_val = alpha[target_idx, coint_idx]
            status = "Converging" if alpha_val < 0 else "Diverging"
            print(f"  Cointegration relation {coint_idx+1}: {alpha_val:>10.6f} {status}")
            if alpha_val < 0:
                negative_alphas.append(alpha_val)
        
        if len(negative_alphas) > 1:
            print(f"\n[Debug] Convergence relations summary:")
            print(f"  - Number of negative alphas: {len(negative_alphas)}")
            print(f"  - Most negative: {np.min(negative_alphas):.6f}")
            print(f"  - Sum: {np.sum(negative_alphas):.6f}")
            if ALPHA_METHOD == 'weighted_mean':
                abs_negative = np.abs(negative_alphas)
                weights = abs_negative / np.sum(abs_negative)
                weighted = np.sum(negative_alphas * weights)
                print(f"  - Weighted average (weighted_mean): {weighted:.6f}")
                print(f"    (Weights calculated by absolute value of each relation)")
except Exception as e:
    print(f"[Debug] Error printing alpha values: {e}")

if initial_alpha >= 0:
    print(f"\n  ⚠️ Warning: Initial alpha is non-negative (should be negative for convergence)")
    print(f"  ⚠️ This combination may not converge properly. Consider using a different combination.")

# Alpha-based re-optimization threshold setting
# Re-optimize if alpha becomes positive or absolute value changes significantly
alpha_change_threshold = 0.5  # Re-optimize if changes by 50% in absolute value

# Determine optimal order for GARCH model
residuals = model_fitted.resid
best_aic, best_order, best_model = find_garch(residuals[:, train_data.columns.get_loc(target_index)])
print(f"Best AIC: {best_aic}")
print(f"Best Order: {best_order}")

# Fit EGARCH model with optimal order
p_opt, o_opt, q_opt = best_order
garch_model = arch_model(residuals[:, train_data.columns.get_loc(target_index)], vol='EGARCH', 
                         p=p_opt, o=o_opt, q=q_opt, rescale=True)
garch_fit = garch_model.fit(disp='off')

# Initialize variables
initial_capital = 10000
capital = initial_capital
long_count = 0
short_count = 0
unrealized_pnl = 0
total_assets = initial_capital
total_shares = 0
average_price = 0
position = None
# Set commission rate as 0.02% (0.0002)
commission_rate = 0.0002
profit_after_commission = 0
results = []
trade_history = []
simulation_start_date = test_data.index[0]
simulation_end_date = test_data.index[-1]
history = train_data.copy()

# Set forecast periods for buy and sell (reflecting Bayesian optimization results)
forecast_steps_buy = 4   # Forecast period for buying
forecast_steps_sell = 7  # Forecast period for selling

# VECM confidence-based variable weight setting
CONFIDENCE_FRACTION_CONFIG = {
    'min_fraction': 0.2,      # Minimum fraction (when confidence is low) - wider range
    'max_fraction': 0.8,      # Maximum fraction (when confidence is high) - wider range
    'window_size': 60,        # Rolling window size (days)
    'method': 'minmax',       # Use Min-Max scaling (more sensitive response)
    # Absolute confidence threshold setting
    # Option 1: Use fixed value
    # 'absolute_threshold': 0.3,
    # Option 2: Dynamic calculation (Recommended) - Set to None to auto-calculate based on history
    'absolute_threshold': None,  # None = Enable dynamic calculation
    'threshold_method': 'percentile',  # 'percentile', 'min', 'mean_std', 'rolling_min'
    'threshold_percentile': 10  # Use lower 10th percentile (when threshold_method='percentile')
}

# Initial confidence calculation (using forecast interval of training data)
# Initial confidence for buy
initial_output_buy, initial_lower_buy, initial_upper_buy = model_fitted.predict(steps=forecast_steps_buy, alpha=0.5)
initial_predicted_mean_buy = initial_output_buy.mean(axis=0)[train_data.columns.get_loc(target_index)]
initial_lower_mean_buy = initial_lower_buy.mean(axis=0)[train_data.columns.get_loc(target_index)]
initial_upper_mean_buy = initial_upper_buy.mean(axis=0)[train_data.columns.get_loc(target_index)]
initial_confidence_buy = get_vecm_confidence(model_fitted, target_index, train_data, 
                                            initial_lower_mean_buy, initial_upper_mean_buy, initial_predicted_mean_buy)

# Initial confidence for sell
initial_output_sell, initial_lower_sell, initial_upper_sell = model_fitted.predict(steps=forecast_steps_sell, alpha=0.5)
initial_predicted_mean_sell = initial_output_sell.mean(axis=0)[train_data.columns.get_loc(target_index)]
initial_lower_mean_sell = initial_lower_sell.mean(axis=0)[train_data.columns.get_loc(target_index)]
initial_upper_mean_sell = initial_upper_sell.mean(axis=0)[train_data.columns.get_loc(target_index)]
initial_confidence_sell = get_vecm_confidence(model_fitted, target_index, train_data, 
                                             initial_lower_mean_sell, initial_upper_mean_sell, initial_predicted_mean_sell)

# Initialize confidence history (separate for buy and sell)
confidence_history_buy = [initial_confidence_buy]
confidence_history_sell = [initial_confidence_sell]
fraction = normalize_confidence_to_fraction(initial_confidence_buy, confidence_history_buy, 
                                            **CONFIDENCE_FRACTION_CONFIG)

# Calculate dynamic threshold (for logging)
if CONFIDENCE_FRACTION_CONFIG.get('absolute_threshold') is None:
    adaptive_threshold_buy = calculate_adaptive_threshold(
        confidence_history_buy,
        method=CONFIDENCE_FRACTION_CONFIG.get('threshold_method', 'percentile'),
        percentile=CONFIDENCE_FRACTION_CONFIG.get('threshold_percentile', 10)
    )
    adaptive_threshold_sell = calculate_adaptive_threshold(
        confidence_history_sell,
        method=CONFIDENCE_FRACTION_CONFIG.get('threshold_method', 'percentile'),
        percentile=CONFIDENCE_FRACTION_CONFIG.get('threshold_percentile', 10)
    )
    threshold_info = f"Dynamic (Buy: {adaptive_threshold_buy:.4f}, Sell: {adaptive_threshold_sell:.4f})"
else:
    threshold_info = f"Fixed: {CONFIDENCE_FRACTION_CONFIG['absolute_threshold']:.4f}"

print(f"\n[VECM confidence-based variable weight system initialization]")
print(f"  Initial Confidence (Buy): {initial_confidence_buy:.6f}")
print(f"  Initial Confidence (Sell): {initial_confidence_sell:.6f}")
print(f"  Initial Fraction: {fraction:.4f}")
print(f"  Absolute Threshold: {threshold_info}")
print(f"  Config: min={CONFIDENCE_FRACTION_CONFIG['min_fraction']}, "
      f"max={CONFIDENCE_FRACTION_CONFIG['max_fraction']}, "
      f"window={CONFIDENCE_FRACTION_CONFIG['window_size']}, "
      f"method={CONFIDENCE_FRACTION_CONFIG['method']}")
unrealized_pnl_history = []
# Initialize cumulative commission
cumulative_commission = 0
# Track shares history for charting
shares_history = []
shares_dates = []
# Track fraction history for analysis
fraction_history = [fraction]

for t in range(len(test_data)):
    # Update history with new test data
    history = pd.concat([history, test_data.iloc[[t]]])
    
    # Print logs only every 100 iterations
    should_log = (t % 100 == 0)
    
    # Calculate upward probability using VECM model
    # Note: Optimization - only re-fit if necessary or periodically
    # Currently re-fitting every step which is expensive but strictly correct for rolling window
    model = VECM(history, k_ar_diff=k_ar_diff_opt, coint_rank=coint_rank_opt, deterministic="colo")
    model_fitted = model.fit()
    
    # Buy prediction (using forecast_steps_buy)
    output_buy, lower_bound_buy, upper_bound_buy = model_fitted.predict(steps=forecast_steps_buy, alpha=0.5)
    lower_mean = lower_bound_buy.mean(axis=0)[history.columns.get_loc(target_index)]
    predicted_mean_buy = output_buy.mean(axis=0)[history.columns.get_loc(target_index)]
    
    # Sell prediction (using forecast_steps_sell)
    output_sell, lower_bound_sell, upper_bound_sell = model_fitted.predict(steps=forecast_steps_sell, alpha=0.5)
    upper_mean = upper_bound_sell.mean(axis=0)[history.columns.get_loc(target_index)]
    predicted_mean_sell = output_sell.mean(axis=0)[history.columns.get_loc(target_index)]
    
    # Debug helper for NaN values (only log causes, no fallback logic here)
    if np.isnan(predicted_mean_buy) or np.isnan(predicted_mean_sell):
        print(f"\n[DEBUG] NaN detected at t={t}, date={test_data.index[t]}")
        print(f"  predicted_mean_buy: {predicted_mean_buy}, predicted_mean_sell: {predicted_mean_sell}")
        print(f"  output_buy shape: {output_buy.shape}, has_nan: {np.isnan(output_buy).any()}")
        print(f"  output_sell shape: {output_sell.shape}, has_nan: {np.isnan(output_sell).any()}")
        print(f"  output_buy sample: {output_buy[:3, :3] if output_buy.size > 0 else 'empty'}")
        print(f"  history length: {len(history)}, history shape: {history.shape}")
        print(f"  target_index: {target_index}, target_index position: {history.columns.get_loc(target_index)}")
    
    # VECM confidence-based variable weight update
    # Calculate buy and sell confidence separately for consistency
    upper_mean_buy = upper_bound_buy.mean(axis=0)[history.columns.get_loc(target_index)]
    lower_mean_sell = lower_bound_sell.mean(axis=0)[history.columns.get_loc(target_index)]
    
    # Calculate buy confidence (using buy prediction interval)
    confidence_buy = get_vecm_confidence(model_fitted, target_index, history, 
                                         lower_mean, upper_mean_buy, predicted_mean_buy)
    
    # Calculate sell confidence (using sell prediction interval)
    confidence_sell = get_vecm_confidence(model_fitted, target_index, history, 
                                          lower_mean_sell, upper_mean, predicted_mean_sell)
    
    # Update confidence history (maintain separate for buy and sell)
    confidence_history_buy.append(confidence_buy)
    confidence_history_sell.append(confidence_sell)
    # Keep only recent window_size (memory efficiency)
    if len(confidence_history_buy) > CONFIDENCE_FRACTION_CONFIG['window_size'] * 2:
        confidence_history_buy = confidence_history_buy[-CONFIDENCE_FRACTION_CONFIG['window_size']:]
    if len(confidence_history_sell) > CONFIDENCE_FRACTION_CONFIG['window_size'] * 2:
        confidence_history_sell = confidence_history_sell[-CONFIDENCE_FRACTION_CONFIG['window_size']:]
    
    # Calculate buy fraction (using buy confidence and buy history)
    fraction_buy = normalize_confidence_to_fraction(confidence_buy, confidence_history_buy, **CONFIDENCE_FRACTION_CONFIG)
    
    # Calculate sell fraction (using sell confidence and sell history)
    # Maintain separate sell confidence history for consistency
    fraction_sell = normalize_confidence_to_fraction(confidence_sell, confidence_history_sell, **CONFIDENCE_FRACTION_CONFIG)
    
    # Current fraction (for logging and recording, using buy fraction)
    fraction = fraction_buy
    fraction_history.append(fraction)
    
    # Use buy prediction for GARCH calculation
    predicted_mean = predicted_mean_buy

    # Update GARCH model
    residuals = model_fitted.resid[:, history.columns.get_loc(target_index)]
    garch_model = arch_model(residuals, vol='EGARCH', p=p_opt, o=o_opt, q=q_opt, rescale=True)
    garch_fit = garch_model.fit(disp='off')
    garch_forecast = garch_fit.forecast(horizon=1, start=len(residuals)-p_opt)
    garch_volatility = np.sqrt(garch_forecast.variance.values[-1, :])
    garch_mean_values = garch_forecast.mean.values
    # Special handling only if there are NaNs, otherwise calculate mean as usual
    if np.isnan(garch_mean_values).any():
        # If any NaN exists, use mean of valid values excluding NaN
        valid_values = garch_mean_values[~np.isnan(garch_mean_values)]
        if len(valid_values) > 0:
            garch_mean_average = np.mean(valid_values)
        else:
            garch_mean_average = 0.0  # If all are NaN, set to 0
    else:
        # If no NaN, calculate full mean as usual
        garch_mean_average = np.mean(garch_mean_values)
    
    # Debug helper for NaN values (GARCH part)
    if np.isnan(garch_mean_average):
        print(f"\n[DEBUG] NaN in garch_mean_average at t={t}, date={test_data.index[t]}")
        print(f"  garch_mean_values: {garch_mean_values}, shape: {garch_mean_values.shape}")
        print(f"  garch_mean_values has_nan: {np.isnan(garch_mean_values).any()}")
        print(f"  residuals length: {len(residuals)}, residuals has_nan: {np.isnan(residuals).any()}")
        print(f"  garch_forecast.variance shape: {garch_forecast.variance.shape if hasattr(garch_forecast, 'variance') else 'N/A'}")
    
    # Hybrid yhat for buying (for buy decision)
    var_hat_buy = predicted_mean_buy + garch_mean_average
    if isinstance(var_hat_buy, np.ndarray):
        hybrid_yhat_buy = var_hat_buy.item()
    else:
        hybrid_yhat_buy = var_hat_buy
    
    # Hybrid yhat for selling (for sell decision)
    var_hat_sell = predicted_mean_sell + garch_mean_average
    if isinstance(var_hat_sell, np.ndarray):
        hybrid_yhat_sell = var_hat_sell.item()
    else:
        hybrid_yhat_sell = var_hat_sell
    
    # Check for NaN in final hybrid_yhat
    if np.isnan(hybrid_yhat_buy) or np.isnan(hybrid_yhat_sell):
        print(f"\n[DEBUG] Final hybrid_yhat NaN at t={t}, date={test_data.index[t]}")
        print(f"  hybrid_yhat_buy: {hybrid_yhat_buy}, hybrid_yhat_sell: {hybrid_yhat_sell}")
        print(f"  predicted_mean_buy: {predicted_mean_buy}, predicted_mean_sell: {predicted_mean_sell}")
        print(f"  garch_mean_average: {garch_mean_average}")

    # Determine whether to re-optimize based on ECT alpha value
    current_alpha = get_alpha_value(model_fitted, target_index, history, method=ALPHA_METHOD)
    
    # Re-optimization check conditions
    # If Alpha > 0, cointegration is broken, so skip trading
    # Checking initial_alpha < 0 is unnecessary: if alpha > 0 after re-optimization, it's still problematic
    should_reoptimize = False
    reoptimize_reason = ""
    
    # If Alpha > 0, cointegration is broken → Skip trading (no re-balancing)
    if current_alpha > 0:
        should_reoptimize = True
        reoptimize_reason = f"Alpha is positive (convergence lost): {current_alpha:.6f} (initial: {initial_alpha:.6f})"

    # Execute re-optimization
    if should_reoptimize:
        if should_log:
            print(f"\n[Re-optimization triggered by ECT alpha] {reoptimize_reason}")
            print(f"  ⚠️ Maintaining position but re-optimizing model contents (no full liquidation)")
        
        # Full Re-optimization Logic
        model, model_fitted, k_ar_diff_opt, coint_rank_opt = fit_vecm_model(history, max_lags=15, deterministic="colo")
        
        if should_log:
            print(f"  Re-optimized k_ar_diff_opt: {k_ar_diff_opt}")
            print(f"  Re-optimized coint_rank_opt: {coint_rank_opt}")
        
        # Re-optimize GARCH model
        residuals_new = model_fitted.resid[:, history.columns.get_loc(target_index)]
        best_aic_new, best_order_new, best_model_new = find_garch(residuals_new)
        p_opt, o_opt, q_opt = best_order_new
        
        if should_log:
            print(f"  Re-optimized GARCH order: {best_order_new} (AIC: {best_aic_new})")
        
        # Update initial alpha with new fitted model
        initial_alpha = get_alpha_value(model_fitted, target_index, history, method=ALPHA_METHOD)
        
        # 6. Calculate confidence even during re-optimization period (Update fraction, but do not trade)
        # Calculate separate confidence for buy and sell
        # Recalculate prediction with re-optimized model
        output_buy, lower_bound_buy, upper_bound_buy = model_fitted.predict(steps=forecast_steps_buy, alpha=0.5)
        lower_mean = lower_bound_buy.mean(axis=0)[history.columns.get_loc(target_index)]
        predicted_mean_buy = output_buy.mean(axis=0)[history.columns.get_loc(target_index)]
        upper_mean_buy = upper_bound_buy.mean(axis=0)[history.columns.get_loc(target_index)]
        
        output_sell, lower_bound_sell, upper_bound_sell = model_fitted.predict(steps=forecast_steps_sell, alpha=0.5)
        upper_mean = upper_bound_sell.mean(axis=0)[history.columns.get_loc(target_index)]
        predicted_mean_sell = output_sell.mean(axis=0)[history.columns.get_loc(target_index)]
        lower_mean_sell = lower_bound_sell.mean(axis=0)[history.columns.get_loc(target_index)]
        
        confidence_buy = get_vecm_confidence(model_fitted, target_index, history, 
                                             lower_mean, upper_mean_buy, predicted_mean_buy)
        confidence_sell = get_vecm_confidence(model_fitted, target_index, history, 
                                              lower_mean_sell, upper_mean, predicted_mean_sell)
        confidence_history_buy.append(confidence_buy)
        confidence_history_sell.append(confidence_sell)
        if len(confidence_history_buy) > CONFIDENCE_FRACTION_CONFIG['window_size'] * 2:
            confidence_history_buy = confidence_history_buy[-CONFIDENCE_FRACTION_CONFIG['window_size']:]
        if len(confidence_history_sell) > CONFIDENCE_FRACTION_CONFIG['window_size'] * 2:
            confidence_history_sell = confidence_history_sell[-CONFIDENCE_FRACTION_CONFIG['window_size']:]
        fraction_buy = normalize_confidence_to_fraction(confidence_buy, confidence_history_buy, **CONFIDENCE_FRACTION_CONFIG)
        fraction_sell = normalize_confidence_to_fraction(confidence_sell, confidence_history_sell, **CONFIDENCE_FRACTION_CONFIG)
        fraction = fraction_buy
        fraction_history.append(fraction)
        
        if should_log:
            print(f"  Updated initial_alpha: {initial_alpha:.6f} (method: {ALPHA_METHOD})")
            print(f"  Confidence (Buy): {confidence_buy:.6f}, Confidence (Sell): {confidence_sell:.6f}")
            print(f"  Fraction (Buy): {fraction_buy:.4f}, Fraction (Sell): {fraction_sell:.4f}")
            print(f"  ⚠️ Skipping trading at this step due to re-optimization\n")
        
        # Skip trading during re-optimization period
        # Only update portfolio value and move to next loop
        date_str = test_data.index[t].strftime('%Y-%m-%d')
        actual_price = test_data[target_index].iloc[t]
        
        # Calculate portfolio value (if position held)
        if position == 'long':
            total_assets = capital + total_shares * actual_price
        else:
            total_assets = capital
        
        results.append(total_assets)
        # Record shares
        shares_history.append(total_shares)
        shares_dates.append(test_data.index[t])
        
        # Record portfolio only, without trading
        trade_history.append({
            'date': date_str,
            'confidence_buy': confidence_buy,
            'confidence_sell': confidence_sell,
            'fraction_buy': fraction_buy,
            'fraction_sell': fraction_sell,
            'fraction': fraction,
            'hybrid_yhat_buy': None,  # No prediction value during re-optimization
            'hybrid_yhat_sell': None,  # No prediction value during re-optimization
            'actual_price': actual_price,
            'capital': capital,
            'total_shares': total_shares,
            'total_assets': total_assets,
            'position': position,
            'unrealized_pnl': (actual_price - average_price) * total_shares if position == 'long' else 0,
            'note': 'Re-optimization'
        })
        
        if should_log:
            print(f"date: {date_str:>10} | [RE-OPTIMIZATION] | actual_price: {actual_price:>10.2f} | "
                  f"capital: {capital:>10.2f} | total_shares: {total_shares:>10.2f} | total_assets: {total_assets:>10.2f} | "
                  f"position: {str(position):>6}")
        
        continue  # Skip trading in this loop and proceed to next

    # Get current actual price
    actual_price = test_data[target_index].iloc[t]
    lower_price = ohlc_data['Low'][target_index].iloc[t]
    upper_price = ohlc_data['High'][target_index].iloc[t]
    close_price = ohlc_data['Close'][target_index].iloc[t]

    # Initialize shares to sell and buy
    shares_to_sell = 0
    shares_to_buy = 0

    # Long Position Entry - Modified to integer unit ordering (Using Buy prediction)
    if hybrid_yhat_buy > actual_price and capital > 0 and lower_price < lower_mean:
        # Set new long position
        position = 'long'
        
        # Round down to integer after decimal calculation
        # Execution price uses lower boundary mean (lower_mean) (Low price cannot be executed)
        # Use buy fraction
        shares_to_buy_float = (capital * fraction_buy) / lower_mean
        shares_to_buy = int(shares_to_buy_float)  # Floor to integer
        
        # Check minimum order unit (0 shares cannot be ordered)
        if shares_to_buy >= 1 and shares_to_buy * lower_mean <= capital:
            total_value = (average_price * total_shares) + (lower_mean * shares_to_buy)
            total_shares += shares_to_buy
            average_price = total_value / total_shares
            commission = shares_to_buy * lower_mean * commission_rate
            cumulative_commission += commission
            capital -= shares_to_buy * lower_mean + commission
            
            if should_log:
                print(f"Buying {shares_to_buy} shares at {lower_mean:.2f} (wanted to buy {shares_to_buy_float:.2f}, lower_price: {lower_price:.2f})")

    # Long Position Exit - Modified to integer unit ordering (Using Sell prediction)
    # Do not sell when the model predicts an upward move
    if position == 'long' and upper_price > upper_mean :#and hybrid_yhat_sell <= actual_price:
        # Round down to integer after decimal calculation
        # Use sell fraction
        shares_to_sell_float = total_shares * fraction_sell
        shares_to_sell = int(shares_to_sell_float)  # Floor to integer
        
        # Check minimum order unit (0 shares cannot be ordered)
        if shares_to_sell >= 1:
            # Sell execution: Sell at upper boundary mean (upper_mean)
            sell_value = shares_to_sell * upper_mean
            commission = shares_to_sell * upper_mean * commission_rate
            cumulative_commission += commission
            total_shares -= shares_to_sell
            # Capital received upon sale: Sale amount - commission
            capital += sell_value - commission
            
            if should_log:
                print(f"Selling {shares_to_sell} shares at {upper_mean:.2f} (wanted to sell {shares_to_sell_float:.2f})")
            
            if total_shares <= 0:
                position = None
                total_shares = 0
                average_price = 0

    # Print key metrics for debugging and logging
    date_str = test_data.index[t].strftime('%Y-%m-%d')
    
    # Calculate total assets
    # If position exists: Cash + Share Value (unrealized P&L already included)
    # If no position: Cash only
    if position == 'long':
        # Long position: Cash + (Current Price * Shares Held)
        total_assets = capital + total_shares * actual_price
    elif position == 'short':
        # Short position: Cash + (Avg Price * Shares Held) + (Avg Price - Current Price) * Shares Held
        # Simplified: Cash + Avg Price * Shares Held + Avg Price * Shares Held - Current Price * Shares Held
        # = Cash + Avg Price * Shares Held + Avg Price * Shares Held - Current Price * Shares Held
        # For short positions, total_shares might be negative, so use absolute value
        total_assets = capital + abs(total_shares) * average_price + (average_price - actual_price) * total_shares
    else:
        # No position: Cash only
        total_assets = capital
    
    # Calculate unrealized P&L (for reference)
    if position == 'long':
        unrealized_pnl = (actual_price - average_price) * total_shares
    elif position == 'short':
        unrealized_pnl = (average_price - actual_price) * total_shares
    else:
        unrealized_pnl = 0
    results.append(total_assets)
    # Record shares
    shares_history.append(total_shares)
    shares_dates.append(test_data.index[t])

    # Print metrics including cumulative commission
    if should_log:
        print(f"date: {date_str:>10} | Conf(Buy): {confidence_buy:>6.4f} | Conf(Sell): {confidence_sell:>6.4f} | "
              f"Frac(Buy): {fraction_buy:>5.3f} | Frac(Sell): {fraction_sell:>5.3f} | "
              f"hybrid_yhat_buy: {hybrid_yhat_buy:>10.2f} | hybrid_yhat_sell: {hybrid_yhat_sell:>10.2f} | actual_price: {actual_price:>10.2f} | "
              f"capital: {capital:>10.2f} | total_shares: {total_shares:>10.2f} | total_assets: {total_assets:>10.2f} | "
              f"position: {str(position):>6} | unrealized_pnl: {unrealized_pnl:>10.2f} | cumulative_commission: {cumulative_commission:>10.2f}")

    # Add trade details to the trade history list
    trade_history.append({
        'date': date_str,
        'confidence_buy': confidence_buy,
        'confidence_sell': confidence_sell,
        'fraction_buy': fraction_buy,
        'fraction_sell': fraction_sell,
        'fraction': fraction,
        'hybrid_yhat_buy': hybrid_yhat_buy,
        'hybrid_yhat_sell': hybrid_yhat_sell,
        'actual_price': close_price,
        'capital': capital,
        'total_shares': total_shares,
        'total_assets': total_assets,
        'position': position,
        'unrealized_pnl': unrealized_pnl
    })

# Convert trade history to DataFrame
trade_history_df = pd.DataFrame(trade_history)
# Save trade history to Excel file in results folder
# Save trade history to Excel file in results folder
# Save results to the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
trade_history_path = os.path.join(results_dir, f'trade_history_{timestamp}.xlsx')
trade_history_df.to_excel(trade_history_path, index=False)
print(f"Trade history saved to: {trade_history_path}")
# Calculate returns
returns = np.diff(results) / results[:-1]
# Convert daily returns to annual returns
trading_days_per_year = 252  # Typically 252 trading days in a year
annualized_return = np.mean(returns) * trading_days_per_year
annualized_std_return = np.std(returns) * np.sqrt(trading_days_per_year)
# Calculate Sharpe ratio (assuming risk-free rate is 0)
sharpe_ratio = annualized_return / annualized_std_return

# Calculate maximum drawdown
strategy_peak, strategy_max_drawdown = max_drawdown(results)
max_loss_streak = max_loss(results)

# Calculate total return over the entire period
total_return = (total_assets - initial_capital) / initial_capital
# Calculate simulation period in years
simulation_years = (simulation_end_date - simulation_start_date).days / 365.25
# Calculate annualized expected return
annualized_return = (1 + total_return) ** (1 / simulation_years) - 1

# Buy and Hold Performance Metrics (TQQQ)
initial_price = test_data[target_index].iloc[0]
final_price = test_data[target_index].iloc[-1]
bnh_total_return = (final_price - initial_price) / initial_price
bnh_annualized_return = (1 + bnh_total_return) ** (1 / simulation_years) - 1

# Calculate Buy and Hold returns for Sharpe ratio
bnh_returns = test_data[target_index].pct_change().dropna()
bnh_annualized_std = bnh_returns.std() * np.sqrt(trading_days_per_year)
bnh_sharpe_ratio = (bnh_annualized_return / bnh_annualized_std) if bnh_annualized_std > 0 else 0

# Calculate Buy and Hold drawdown
bnh_values = test_data[target_index].values
bnh_peak, bnh_max_drawdown = max_drawdown(bnh_values)

print("")
print(f"Simulation Performance: {simulation_start_date.date()} ~ {simulation_end_date.date()}")
print(f"=" * 80)

print(f"STRATEGY PERFORMANCE:")
print(f"  Final Capital: ${total_assets:,.2f} USD")
print(f"  Cumulative Return: {total_return:.2%}")
print(f"  Annualized Return: {annualized_return:.2%}")
print(f"  Return Standard Deviation: {annualized_std_return:.4f}")
print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"  Peak: ${strategy_peak:,.2f}")
print(f"  Maximum Drawdown: ${strategy_max_drawdown:,.2f}")
print(f"  Maximum Consecutive Loss Days: {max_loss_streak}")

print(f"\nBUY & HOLD ({target_index}) PERFORMANCE:")
print(f"  Final Price: ${final_price:,.2f} USD (Initial: ${initial_price:,.2f} USD)")
print(f"  Cumulative Return: {bnh_total_return:.2%}")
print(f"  Annualized Return: {bnh_annualized_return:.2%}")
print(f"  Return Standard Deviation: {bnh_annualized_std:.4f}")
print(f"  Sharpe Ratio: {bnh_sharpe_ratio:.4f}")
print(f"  Peak: ${bnh_peak:,.2f}")
print(f"  Maximum Drawdown: ${bnh_max_drawdown:,.2f}")

# Comparison
print(f"\nCOMPARISON:")
print(f"  Outperformance (Return): {(total_return - bnh_total_return):+.2%}")
print(f"  Outperformance (Annualized): {(annualized_return - bnh_annualized_return):+.2%}")
print(f"  Sharpe Ratio Difference: {sharpe_ratio - bnh_sharpe_ratio:+.4f}")
print(f"=" * 80)

# Prepare chart data
chart_df = pd.DataFrame({
    'date': [test_data.index[t] for t in range(len(test_data))],
    'price': [test_data[target_index].iloc[t] for t in range(len(test_data))],
})

# Match shares_history by date
shares_dict = dict(zip(shares_dates, shares_history))
chart_df['shares'] = [shares_dict.get(test_data.index[t], 0) for t in range(len(test_data))]

# Match portfolio_value to dates
portfolio_value_list = []
results_idx = 0
for t in range(len(test_data)):
    current_date = test_data.index[t]
    if results_idx < len(results):
        portfolio_value_list.append(results[results_idx])
        if results_idx + 1 < len(results):
            results_idx += 1
    else:
        portfolio_value_list.append(results[-1] if results else initial_capital)
chart_df['portfolio_value'] = portfolio_value_list
chart_df.set_index('date', inplace=True)

# Calculate Buy and Hold
initial_price = test_data[target_index].iloc[0]
buy_hold_shares = initial_capital / initial_price
buy_hold_commission_buy = buy_hold_shares * initial_price * commission_rate
buy_hold_actual_shares = (initial_capital - buy_hold_commission_buy) / initial_price
buy_hold_values = [buy_hold_actual_shares * price for price in chart_df['price']]
chart_df['buy_hold_value'] = buy_hold_values

# Calculate background color intensity based on quantity (normalized)
max_shares = chart_df['shares'].max() if chart_df['shares'].max() > 0 else 1
chart_df['shares_normalized'] = chart_df['shares'] / max_shares  # Normalize to 0-1 range

# Visualization: 3-panel vertical layout
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

# Panel 1: Price Chart + Background Color by Quantity
ax1.set_ylabel('Price (USD)', fontsize=12)
ax1.set_title(f'{target_index} Price with Holdings Background', fontsize=14, fontweight='bold')

# Apply background color gradient based on quantity
# Darker green for larger quantities, lighter green for smaller quantities
# Alpha restricted to 0.15~0.35 range (not too dark)
for i in range(len(chart_df)):
    date = chart_df.index[i]
    shares_norm = chart_df['shares_normalized'].iloc[i]
    # Color intensity: max 0.35 alpha, min 0.15 alpha
    alpha_value = 0.15 + (shares_norm * 0.20)  # 0.15 ~ 0.35 range
    if i < len(chart_df) - 1:
        ax1.axvspan(date, chart_df.index[i+1], alpha=alpha_value, color='green', zorder=0)

# Price Line
ax1.plot(chart_df.index, chart_df['price'], label=f'{target_index} Price', 
         linewidth=2, color='black', zorder=2)
ax1.grid(True, alpha=0.3, zorder=1)
ax1.legend(loc='upper left', fontsize=10)

# Panel 2: Shares Held Bar Chart
ax2.set_ylabel('Shares Held', fontsize=12)
ax2.set_title('Holdings Quantity', fontsize=14, fontweight='bold')
# Display quantity as bar chart
ax2.bar(chart_df.index, chart_df['shares'], width=1, alpha=0.6, color='orange', label='Shares Held')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend(loc='upper left', fontsize=10)

# Panel 3: Portfolio Value Trend + Buy & Hold Comparison
ax3.set_ylabel('Portfolio Value (USD)', fontsize=12)
ax3.set_xlabel('Date', fontsize=12)
ax3.set_title('Portfolio Value vs Buy & Hold', fontsize=14, fontweight='bold')
# Strategy Portfolio Value
ax3.plot(chart_df.index, chart_df['portfolio_value'], label='Strategy Portfolio', 
         linewidth=2, color='blue')
# Buy & Hold
ax3.plot(chart_df.index, chart_df['buy_hold_value'], label='Buy & Hold', 
         linewidth=2, color='red', linestyle='--')
# Initial Capital Baseline
ax3.axhline(y=initial_capital, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Initial Capital')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper left', fontsize=10)

# Performance comparison text (placed at bottom right to avoid overlap with legend)
strategy_return = (chart_df['portfolio_value'].iloc[-1] - initial_capital) / initial_capital * 100
buy_hold_return = (chart_df['buy_hold_value'].iloc[-1] - initial_capital) / initial_capital * 100
outperformance = strategy_return - buy_hold_return

ax3.text(0.98, 0.02, 
         f'Strategy: {strategy_return:.2f}%\nBuy & Hold: {buy_hold_return:.2f}%\nOutperformance: {outperformance:+.2f}%',
         transform=ax3.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
# Save chart to results folder
chart_path = os.path.join(results_dir, f'qqq_trading_simulation_chart_{timestamp}.png')
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
print(f"Chart saved to: {chart_path}")
plt.show()

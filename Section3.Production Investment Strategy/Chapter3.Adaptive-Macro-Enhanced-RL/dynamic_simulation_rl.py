import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VECM
from arch import arch_model
import warnings
from datetime import datetime, timedelta
from statsmodels.tsa.vector_ar.vecm import select_order
from statsmodels.tsa.vector_ar.vecm import select_coint_rank
import sys
import os
from matplotlib.patches import Patch

# Import local modules
from functions_ import find_garch, max_drawdown, calculate_mdd_details, max_loss
from trading_utils import (
    get_alpha_value, 
    get_vecm_confidence, 
    calculate_adaptive_threshold, 
    normalize_confidence_to_fraction
)

try:
    from config import get_config
    SYSTEM_CONFIG = get_config()
except ImportError:
    print("Warning: config.py not found. Using defaults.")
    SYSTEM_CONFIG = {}

# Regime Detection System Integration
try:
    from regime_detector import ModelBasedRegimeDetector
    USE_REGIME_DETECTION = SYSTEM_CONFIG.get('use_regime_detection', True)
except ImportError as e:
    print(f"Warning: Could not import regime detector: {e}")
    print(f"  Make sure 'regime_detector.py' is in the same directory.")
    USE_REGIME_DETECTION = False
    ModelBasedRegimeDetector = None

# RL System Integration (Production Policy Only)
USE_RL_AGENT = SYSTEM_CONFIG.get('use_rl_agent', True)  # Set to False to disable RL
# RL_BLEND_FACTOR is read from env or defaults to 0.6
RL_BLEND_FACTOR = float(os.environ.get('RL_BLEND_FACTOR', str(SYSTEM_CONFIG.get('rl_blend_factor', 0.6)))) 

try:
    from rl_agent import VECMRLAgent
    if not USE_RL_AGENT:
        VECMRLAgent = None
except ImportError as e:
    print(f"Warning: Could not import RL agent: {e}")
    print(f"  Make sure 'rl_agent.py' is in the same directory.")
    USE_RL_AGENT = False
    VECMRLAgent = None

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

# Optimal Parameter Estimation
lag_order = select_order(train_data, maxlags=15, deterministic="cili")
time_lag = lag_order.aic
print(f"time_lag: {time_lag}")

# Johansen Test for Cointegration Rank
coint_rank_test = select_coint_rank(train_data, det_order=1, k_ar_diff=time_lag, method='trace')
coint_rank_opt = coint_rank_test.rank
k_ar_diff_opt = time_lag

# Fit VECM Model
model = VECM(train_data, k_ar_diff=k_ar_diff_opt, coint_rank=coint_rank_opt, deterministic="cili")
model_fitted = model.fit()
print(f"k_ar_diff_opt: {k_ar_diff_opt}")
print(f"coint_rank_opt: {coint_rank_opt}")

# (get_alpha_value moved to trading_utils.py)

# Alpha Calculation Configuration
ALPHA_METHOD = 'weighted_mean'  # Options: 'weighted_mean', 'sum_negative', 'min_negative'

# Initial Alpha Calculation
initial_alpha = get_alpha_value(model_fitted, target_index, train_data, method=ALPHA_METHOD)
print(f"Initial ECT alpha value: {initial_alpha:.6f} (method: {ALPHA_METHOD})")

# Debugging: Print all cointegration alpha values
try:
    alpha = model_fitted.alpha
    target_idx = train_data.columns.get_loc(target_index)
    if alpha.ndim == 2:
        print(f"\n[Debug] Alpha matrix shape: {alpha.shape}")
        print(f"[Debug] Target variable index: {target_idx} ({target_index})")
        print(f"[Debug] Alpha values for {target_index}:")
        negative_alphas = []
        for coint_idx in range(alpha.shape[1]):
            val = alpha[target_idx, coint_idx]
            status = "✓ Conv" if val < 0 else "✗ Div"
            print(f"  Relation {coint_idx+1}: {val:>10.6f} {status}")
            if val < 0:
                negative_alphas.append(val)
        
        if len(negative_alphas) > 1:
            print(f"\n[Debug] Convergence Summary:")
            print(f"  - Count of negative alphas: {len(negative_alphas)}")
            print(f"  - Min: {np.min(negative_alphas):.6f}")
            print(f"  - Sum: {np.sum(negative_alphas):.6f}")
            if ALPHA_METHOD == 'weighted_mean':
                abs_neg = np.abs(negative_alphas)
                w = abs_neg / np.sum(abs_neg)
                print(f"  - Weighted Mean: {np.sum(negative_alphas * w):.6f}")
except Exception as e:
    print(f"[Debug] Alpha print error: {e}")

if initial_alpha >= 0:
    print(f"\n  ⚠️ Warning: Initial alpha is non-negative (should be negative for convergence)")
    print(f"  ⚠️ This combination may not converge properly. Consider using a different combination.")

# ECT Alpha-based Re-optimization Thresholds
alpha_change_threshold = SYSTEM_CONFIG.get('alpha_change_threshold', 0.5)

# (get_vecm_confidence moved to trading_utils.py)
# (calculate_adaptive_threshold moved to trading_utils.py)
# (normalize_confidence_to_fraction moved to trading_utils.py)

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

# Initialize Trading State
initial_capital = SYSTEM_CONFIG.get('initial_capital', 10000)
commission_rate = SYSTEM_CONFIG.get('commission_rate', 0.0002)

capital = initial_capital
total_shares = 0
average_price = 0
position = None
peak_price = 0
cumulative_commission = 0

# Load Risk Management
risk_cfg = SYSTEM_CONFIG.get('risk_management', {})
USE_RISK_MANAGEMENT = risk_cfg.get('use_risk_management', False)
STOP_LOSS_MULT = risk_cfg.get('stop_loss_mult', 8.0)
TRAILING_STOP_BASE = risk_cfg.get('trailing_stop_base', 0.25)
MIN_STOP_LOSS = risk_cfg.get('min_stop_loss', 0.15)
MAX_STOP_LOSS = risk_cfg.get('max_stop_loss', 0.50)
ENTRY_EDGE_PCT = risk_cfg.get('entry_edge_pct', 0.0)

results = []
trade_history = []
simulation_start_date = test_data.index[0]
simulation_end_date = test_data.index[-1]
history = train_data.copy()

# Forecasting windows (Optimized for TQQQ)
forecast_steps_buy = SYSTEM_CONFIG.get('forecast_steps_buy', 4)
forecast_steps_sell = SYSTEM_CONFIG.get('forecast_steps_sell', 7)

# Configuration for VECM Confidence based Dynamic Fraction
CONFIDENCE_FRACTION_CONFIG = SYSTEM_CONFIG.get('confidence_fraction_config_base', {
    'min_fraction': 0.2,
    'max_fraction': 0.8,
    'window_size': 60,
    'method': 'minmax',
    'absolute_threshold': None,
    'threshold_method': 'percentile',
    'threshold_percentile': 10
})

# Regime-specific Fraction Configuration (Loaded from config.py)
REGIME_FRACTION_CONFIG = SYSTEM_CONFIG.get('regime_fraction_config', {
    'bull': {'min_fraction': 0.68, 'max_fraction': 0.97, 'sell_ratio_cap': 0.4414},
    'bear': {'min_fraction': 0.48, 'max_fraction': 0.77},
    'sideways': {'min_fraction': 0.43, 'max_fraction': 0.86},
    'high_vol': {'min_fraction': 0.47, 'max_fraction': 0.70}
})

# Initialize Regime Detection System
if USE_REGIME_DETECTION and ModelBasedRegimeDetector is not None:
    regime_detector = ModelBasedRegimeDetector(window=60, hysteresis_threshold=0.05)
    print(f"\n[Regime Detection System Initialized]")
    print(f"  Window: 60 days")
    print(f"  Hysteresis Threshold: 0.05")
else:
    regime_detector = None
    print(f"\n[Regime Detection System Disabled]")

# RL Agent Initialization
rl_agent = None
if USE_RL_AGENT and VECMRLAgent is not None:
    try:
        simple_policy_params = SYSTEM_CONFIG.get('rl_policy_params', {})
        print(f"\n[RL Agent Initialized]")
        print(f"  Mode: Simple Policy (VECM Confidence Based)")
        print(f"  Blend Factor: {RL_BLEND_FACTOR * 100:.1f}%")
        
        rl_agent = VECMRLAgent(simple_policy_params=simple_policy_params)
    except Exception as e:
        print(f"\n[RL Agent Initialization Failed] {e}")
        USE_RL_AGENT = False
        rl_agent = None
else:
    rl_agent = None
    USE_RL_AGENT = False
    print(f"\n[RL System Disabled]")

# Initialize Confidence & Fractions
# Base configuration is already loaded into CONFIDENCE_FRACTION_CONFIG

# Initial Buy Confidence
initial_output_buy, initial_lower_buy, initial_upper_buy = model_fitted.predict(steps=forecast_steps_buy, alpha=0.5)
initial_predicted_mean_buy = initial_output_buy.mean(axis=0)[train_data.columns.get_loc(target_index)]
initial_lower_mean_buy = initial_lower_buy.mean(axis=0)[train_data.columns.get_loc(target_index)]
initial_upper_mean_buy = initial_upper_buy.mean(axis=0)[train_data.columns.get_loc(target_index)]
initial_confidence_buy = get_vecm_confidence(model_fitted, target_index, train_data, 
                                            initial_lower_mean_buy, initial_upper_mean_buy, initial_predicted_mean_buy)

# Initial Sell Confidence
initial_output_sell, initial_lower_sell, initial_upper_sell = model_fitted.predict(steps=forecast_steps_sell, alpha=0.5)
initial_predicted_mean_sell = initial_output_sell.mean(axis=0)[train_data.columns.get_loc(target_index)]
initial_lower_mean_sell = initial_lower_sell.mean(axis=0)[train_data.columns.get_loc(target_index)]
initial_upper_mean_sell = initial_upper_sell.mean(axis=0)[train_data.columns.get_loc(target_index)]
initial_confidence_sell = get_vecm_confidence(model_fitted, target_index, train_data, 
                                             initial_lower_mean_sell, initial_upper_mean_sell, initial_predicted_mean_sell)

confidence_history_buy = [initial_confidence_buy]
confidence_history_sell = [initial_confidence_sell]
fraction = normalize_confidence_to_fraction(initial_confidence_buy, confidence_history_buy, 
                                            **CONFIDENCE_FRACTION_CONFIG)

# Dynamic Threshold Info
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

print(f"\n[VECM Confidence-based Dynamic Fraction System Initialized]")
print(f"  Initial Confidence (Buy): {initial_confidence_buy:.6f}")
print(f"  Initial Confidence (Sell): {initial_confidence_sell:.6f}")
print(f"  Initial Fraction: {fraction:.4f}")
print(f"  Absolute Threshold: {threshold_info}")
print(f"  Config: min={CONFIDENCE_FRACTION_CONFIG['min_fraction']}, max={CONFIDENCE_FRACTION_CONFIG['max_fraction']}")

if USE_REGIME_DETECTION and regime_detector is not None:
    print(f"\n[Regime-specific Fraction Ranges]")
    for regime, config in REGIME_FRACTION_CONFIG.items():
        print(f"  {regime.upper()}: min={config['min_fraction']:.2f}, max={config['max_fraction']:.2f}")

# Backtest Tracking
cumulative_commission = 0
shares_history = []
shares_dates = []
fraction_history = [fraction]

# --- Main Simulation Loop ---
for t in range(len(test_data)):
    current_action = 'HOLD'
    # Update sliding window history
    history = pd.concat([history, test_data.iloc[[t]]])
    should_log = (t % 100 == 0)
    
    # Fit VECM with updated history
    model = VECM(history, k_ar_diff=k_ar_diff_opt, coint_rank=coint_rank_opt, deterministic="cili")
    model_fitted = model.fit()
    
    # --- 1. GARCH Modeling for Hybrid Prediction ---
    try:
        # Update GARCH model using residuals from VECM
        target_resids = model_fitted.resid[:, history.columns.get_loc(target_index)]
        garch_model_obj = arch_model(target_resids, vol='EGARCH', p=p_opt, o=o_opt, q=q_opt, rescale=True)
        garch_fit = garch_model_obj.fit(disp='off')
        
        # Forecast 1 step ahead (using specific start point to be consistent with original)
        garch_forecast = garch_fit.forecast(horizon=1, start=len(target_resids)-p_opt)
        garch_volatility = np.sqrt(garch_forecast.variance.values[-1, :])
        garch_mean_values = garch_forecast.mean.values
        
        # NaN handling for GARCH mean values (matches Chapter 2 exactly)
        if np.isnan(garch_mean_values).any():
            valid_values = garch_mean_values[~np.isnan(garch_mean_values)]
            garch_mean_average = np.mean(valid_values) if len(valid_values) > 0 else 0.0
        else:
            garch_mean_average = np.mean(garch_mean_values)
    except Exception as e:
        if should_log:
            print(f"  Warning: GARCH modeling failed: {e}")
        garch_mean_average = 0.0
        garch_volatility = 0.1
        garch_vol_mean = 0.1

    # --- 2. VECM Forecasting (Buy/Sell Intervals) ---
    # Prediction for Buy (forecast_steps_buy)
    output_buy, lower_bound_buy, upper_bound_buy = model_fitted.predict(steps=forecast_steps_buy, alpha=0.5)
    lower_mean = lower_bound_buy.mean(axis=0)[history.columns.get_loc(target_index)]
    predicted_mean_buy = output_buy.mean(axis=0)[history.columns.get_loc(target_index)]
    upper_mean_buy = upper_bound_buy.mean(axis=0)[history.columns.get_loc(target_index)]
    
    # Prediction for Sell (forecast_steps_sell)
    output_sell, lower_bound_sell, upper_bound_sell = model_fitted.predict(steps=forecast_steps_sell, alpha=0.5)
    upper_mean = upper_bound_sell.mean(axis=0)[history.columns.get_loc(target_index)]
    predicted_mean_sell = output_sell.mean(axis=0)[history.columns.get_loc(target_index)]
    lower_mean_sell = lower_bound_sell.mean(axis=0)[history.columns.get_loc(target_index)]
    
    # --- 3. Regime Detection & Fraction Management ---
    current_regime = 'sideways'
    if USE_REGIME_DETECTION and regime_detector is not None:
        try:
            garch_vol_mean = float(np.mean(garch_volatility)) if isinstance(garch_volatility, np.ndarray) else float(garch_volatility)
            current_regime = regime_detector.detect_regime(
                vecm_pred=predicted_mean_buy,
                kalman_beta=None,
                garch_vol=garch_vol_mean,
                model_confidences={'vecm': confidence_history_buy[-1] if confidence_history_buy else 0.5},
                copula_risk=None
            )
            
            # Apply regime-specific fraction ranges
            if current_regime in REGIME_FRACTION_CONFIG:
                reg_cfg = REGIME_FRACTION_CONFIG[current_regime]
                CONFIDENCE_FRACTION_CONFIG['min_fraction'] = reg_cfg['min_fraction']
                CONFIDENCE_FRACTION_CONFIG['max_fraction'] = reg_cfg['max_fraction']
                
                if should_log:
                    print(f"  [Regime] {current_regime.upper()}: min={reg_cfg['min_fraction']:.2f}, max={reg_cfg['max_fraction']:.2f}")
        except Exception as e:
            if should_log: print(f"  Warning: Regime detection failed: {e}")
            current_regime = 'sideways'
            pass # Ranges are already handled by reg_cfg
    
    # Calculate separate confidences for Buy and Sell
    confidence_buy = get_vecm_confidence(model_fitted, target_index, history, 
                                         lower_mean, upper_mean_buy, predicted_mean_buy)
    confidence_sell = get_vecm_confidence(model_fitted, target_index, history, 
                                          lower_mean_sell, upper_mean, predicted_mean_sell)
    confidence_history_buy.append(confidence_buy)
    confidence_history_sell.append(confidence_sell)
    
    # Normalize confidence to base fractions
    base_fraction_buy = normalize_confidence_to_fraction(confidence_buy, confidence_history_buy, **CONFIDENCE_FRACTION_CONFIG)
    base_fraction_sell = normalize_confidence_to_fraction(confidence_sell, confidence_history_sell, **CONFIDENCE_FRACTION_CONFIG)
    
    # --- 4. RL Agent Integration (Hybrid Signal) ---
    fraction_buy = base_fraction_buy
    fraction_sell = base_fraction_sell

    if USE_RL_AGENT and rl_agent is not None and t >= 10:
        try:
            reg_map = {'bull': 1.0, 'bear': -1.0, 'sideways': 0.0, 'high_vol': 0.5}
            reg_val = reg_map.get(current_regime, 0.0)
            cur_p = test_data[target_index].iloc[t]
            
            # Position and Capital ratios
            pos_ratio = (total_shares * cur_p) / (capital + total_shares * cur_p) if (capital + total_shares * cur_p) > 0 else 0.0
            cap_ratio = capital / initial_capital if initial_capital > 0 else 1.0
            
            # Macro indicators
            try:
                cl_p = history['CL=F'].iloc[-21] if len(history) > 20 else history['CL=F'].iloc[0]
                cl_c = history['CL=F'].iloc[-1]
                cl_mom = (cl_c - cl_p) / cl_p if cl_p > 0 else 0.0
                zb_p = history['ZB=F'].iloc[-21] if len(history) > 20 else history['ZB=F'].iloc[0]
                zb_c = history['ZB=F'].iloc[-1]
                zb_mom = (zb_c - zb_p) / zb_p if zb_p > 0 else 0.0
            except:
                cl_mom, zb_mom = 0.0, 0.0
            
            cur_alpha = get_alpha_value(model_fitted, target_index, history, method=ALPHA_METHOD)
            
            obs = np.array([predicted_mean_buy, confidence_buy, garch_vol_mean, reg_val, pos_ratio, cap_ratio, cl_mom, zb_mom, cur_alpha], dtype=np.float32)
            
            rl_action = rl_agent.predict(obs, deterministic=True)
            rl_pos_size = float(np.clip(rl_action[0], 0.0, 1.0))
            rl_signal = float(np.clip(rl_action[1], -1.0, 1.0))

            # Always reflect RL's position sizing judgment (Removed 0.5 Hurdle)
            if rl_signal >= 0:
                # RL suggests a buying weight based on macro/confidence
                rl_fraction_buy = np.clip(rl_pos_size, CONFIDENCE_FRACTION_CONFIG['min_fraction'], CONFIDENCE_FRACTION_CONFIG['max_fraction'])
                rl_fraction_sell = base_fraction_sell
            else:
                # RL suggests a selling weight based on macro/confidence
                rl_fraction_sell = np.clip(rl_pos_size, CONFIDENCE_FRACTION_CONFIG['min_fraction'], CONFIDENCE_FRACTION_CONFIG['max_fraction'])
                rl_fraction_buy = base_fraction_buy
            
            blend = RL_BLEND_FACTOR
            fraction_buy = np.clip(blend * rl_fraction_buy + (1 - blend) * base_fraction_buy, CONFIDENCE_FRACTION_CONFIG['min_fraction'], CONFIDENCE_FRACTION_CONFIG['max_fraction'])
            fraction_sell = np.clip(blend * rl_fraction_sell + (1 - blend) * base_fraction_sell, CONFIDENCE_FRACTION_CONFIG['min_fraction'], CONFIDENCE_FRACTION_CONFIG['max_fraction'])
            
            if should_log:
                print(f"  [RL] Buy Frac: {fraction_buy:.4f}, Sell Frac: {fraction_sell:.4f}, Sig: {rl_signal:.2f}")
        except Exception as e:
            if should_log: print(f"  Warning: RL integration failed: {e}")

    # Current fraction for logging
    fraction = fraction_buy
    fraction_history.append(fraction)
    
    # Hybrid Prediction calculation
    hybrid_yhat_buy = predicted_mean_buy + garch_mean_average
    hybrid_yhat_sell = predicted_mean_sell + garch_mean_average
    
    # --- 5. ECT Cointegration Check & Trading skip ---
    current_alpha = get_alpha_value(model_fitted, target_index, history, method=ALPHA_METHOD)
    if current_alpha > 0:
        if should_log: print(f"\n[Re-optimization] Alpha lost convergence: {current_alpha:.6f}. Skipping trade.")
        lag_new = select_order(history, maxlags=15, deterministic="cili")
        k_ar_diff_opt = lag_new.aic
        coint_new = select_coint_rank(history, det_order=1, k_ar_diff=k_ar_diff_opt, method='trace')
        coint_rank_opt = coint_new.rank
        model = VECM(history, k_ar_diff=k_ar_diff_opt, coint_rank=coint_rank_opt, deterministic="cili")
        model_fitted = model.fit()
        continue 

    # --- 6. Trading Execution Logic (Limit Order Protocol) ---
    actual_price = test_data[target_index].iloc[t]
    lower_price = ohlc_data['Low'][target_index].iloc[t]
    upper_price = ohlc_data['High'][target_index].iloc[t]
    close_price = ohlc_data['Close'][target_index].iloc[t]
    is_bull_regime = (current_regime == 'bull')

    # 1. Buy Execution (Limit Order at lower_mean)
    # Entry condition requires current low to touch/break the lower bound forecast.
    entry_condition = (lower_price < lower_mean)
    expected_return = (hybrid_yhat_buy - actual_price) / actual_price if actual_price > 0 else 0
    has_edge = expected_return > ENTRY_EDGE_PCT

    if hybrid_yhat_buy > actual_price and capital > 0 and entry_condition and has_edge:
        if position != 'long':
            peak_price = actual_price  # Reset peak for new position
        position = 'long'
        
        # Calculate shares using Buy Fraction and Execution Price (lower_mean)
        shares_to_buy_float = (capital * fraction_buy) / lower_mean
        shares_to_buy = int(shares_to_buy_float)
        
        if shares_to_buy >= 1 and shares_to_buy * lower_mean <= capital:
            buy_value = shares_to_buy * lower_mean
            commission = buy_value * commission_rate
            cumulative_commission += commission
            # Update average purchase price for PnL tracking
            total_purchase_val = (average_price * total_shares) + buy_value
            total_shares += shares_to_buy
            average_price = total_purchase_val / total_shares
            capital -= (buy_value + commission)
            
            current_action = 'BUY'
            if should_log:
                print(f"  [Trade] Buying {shares_to_buy} shares at {lower_mean:.2f} (Limit Order filled)")

    # 2. Risk Management (Trailing Stop Protection)
    # Note: Optimization follows regime-based scaling; this is an extra safety layer.
    if USE_RISK_MANAGEMENT and position == 'long' and total_shares >= 1:
        current_ret = (actual_price - average_price) / average_price if average_price > 0 else 0
        if current_ret > 0.15: # Trigger only if in profit
            dynamic_stop_pct = np.clip(garch_volatility * (STOP_LOSS_MULT * 0.7), MIN_STOP_LOSS, MAX_STOP_LOSS)
            trailing_stop_price = peak_price * (1 - TRAILING_STOP_BASE * 0.8)
            guaranteed_price = average_price * 1.05
            exit_trigger = max(trailing_stop_price, guaranteed_price)
            
            if lower_price < exit_trigger:
                # Use open or trigger price, whichever is lower (gap down protection)
                sell_p = min(actual_price, exit_trigger)
                sell_val = total_shares * sell_p
                comm = sell_val * commission_rate
                cumulative_commission += comm
                capital += sell_val - comm
                print(f"!!! PROFIT PROTECTION !!! Sold {total_shares} at {sell_p:.2f} (Return: {current_ret:.2%})")
                current_action = 'SELL_PROT'
                total_shares, average_price, position, peak_price = 0, 0, None, 0

    # 3. Sell Execution (Limit Order at upper_mean)
    should_sell = (upper_price > upper_mean)
    
    # Bull Regime Persistence: Adaptive Reversal Threshold (Golden Ratio: 0.5007 * Vol)
    if is_bull_regime:
        expected_sell_ret = (hybrid_yhat_sell - actual_price) / actual_price
        # Detect if predicted reversal is stronger than 0.5x current volatility
        adaptive_reversal_threshold = -(garch_volatility * 0.5007)
        should_sell = should_sell and (expected_sell_ret < adaptive_reversal_threshold)

    if position == 'long' and should_sell and total_shares >= 1:
        if is_bull_regime:
            # Optimized profit taking in bull regime to maximize capital efficiency
            sell_ratio = reg_cfg.get('sell_ratio_cap', 0.4414)
            shares_to_sell_float = total_shares * sell_ratio
        else:
            # Full logic weighting in other regimes
            shares_to_sell_float = total_shares * fraction_sell
            
        shares_to_sell = int(shares_to_sell_float)
        
        if shares_to_sell >= 1:
            sell_value = shares_to_sell * upper_mean
            commission = sell_value * commission_rate
            cumulative_commission += commission
            total_shares -= shares_to_sell
            capital += sell_value - commission
            
            current_action = 'SELL'
            if should_log:
                print(f"  [Trade] Selling {shares_to_sell} shares at {upper_mean:.2f} (Target Reached)")
            if total_shares <= 0:
                position, total_shares, average_price = None, 0, 0

    # Valuation and Logging
    # Evaluated at Daily Close to match standardized reporting
    if position == 'long':
        total_assets = capital + total_shares * close_price
    else:
        total_assets = capital
    
    unrealized_pnl = (close_price - average_price) * total_shares if position == 'long' else 0
    results.append(total_assets)
    shares_history.append(total_shares)
    shares_dates.append(test_data.index[t])

    if should_log:
        date_str_loop = test_data.index[t].strftime('%Y-%m-%d')
        # 매수/매도 여력(Edge) 계산 로그 강화
        buy_edge = (hybrid_yhat_buy - actual_price) / actual_price if actual_price > 0 else 0
        sell_edge = (hybrid_yhat_sell - actual_price) / actual_price if actual_price > 0 else 0
        
        print(f"date: {date_str_loop} | Conf(B/S): {confidence_buy:.3f}/{confidence_sell:.3f} | Edge(B/S): {buy_edge:+.2%}/{sell_edge:+.2%} | "
              f"Yhat(B): {hybrid_yhat_buy:7.2f} | Prc: {actual_price:7.2f} | Assets: {total_assets:10.2f} | Pos: {str(position):>6} | "
              f"Regime: {current_regime:<8}")
        
        if hybrid_yhat_buy <= actual_price and position != 'long' and capital > 0:
            if t % 50 == 0:
                print(f"  [Info] Buy skipped: Bearish forecast (Yhat {hybrid_yhat_buy:.2f} <= Actual {actual_price:.2f})")

    # Track High-Water Mark for Risk Management
    if position == 'long':
        peak_price = max(peak_price, upper_price)
    else:
        peak_price = 0

    trade_history.append({
        'date': test_data.index[t].strftime('%Y-%m-%d'),
        'action': current_action,
        'confidence_buy': confidence_buy,
        'confidence_sell': confidence_sell,
        'fraction_buy': fraction_buy,
        'fraction_sell': fraction_sell,
        'hybrid_yhat_buy': hybrid_yhat_buy,
        'hybrid_yhat_sell': hybrid_yhat_sell,
        'open_price': actual_price,
        'close_price': close_price,
        'capital': capital,
        'total_shares': total_shares,
        'total_assets': total_assets,
        'position': position,
        'unrealized_pnl': unrealized_pnl,
        'regime': current_regime
    })

# Final Performance Metrics & Visualization
trade_history_df = pd.DataFrame(trade_history)

# Source code directory-based result folder path settings
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
trade_history_path = os.path.join(results_dir, f'trade_history_{timestamp}.csv')
trade_history_df.to_csv(trade_history_path, index=False, encoding='utf-8-sig')
print(f"\nTrade history saved to: {trade_history_path}")

# Calculate simulation parameters
simulation_start_date = test_data.index[0]
simulation_end_date = test_data.index[-1]
simulation_years = (simulation_end_date - simulation_start_date).days / 365.25

# 1. Strategy Performance Metrics
total_return = (total_assets - initial_capital) / initial_capital
annualized_return_calc = (1 + total_return) ** (1 / simulation_years) - 1 if (1 + total_return) > 0 else -1.0

daily_returns = pd.Series(results).pct_change().dropna()
annualized_vol = daily_returns.std() * np.sqrt(252)
sharpe_ratio = (annualized_return_calc / annualized_vol) if annualized_vol > 0 else 0

strategy_dates_list = [test_data.index[t] for t in range(len(test_data))]
strat_peak_at_mdd, strat_mdd_pct, strat_mdd_date, strat_mdd2_pct, strat_mdd2_date = calculate_mdd_details(results, strategy_dates_list)
max_loss_streak_val = max_loss(results)

# 2. Buy & Hold (TQQQ) Benchmark
initial_price_val = test_data[target_index].iloc[0]
final_price_val = test_data[target_index].iloc[-1]
bnh_total_return = (final_price_val - initial_price_val) / initial_price_val
bnh_annual_return = (1 + bnh_total_return) ** (1 / simulation_years) - 1 if (1 + bnh_total_return) > 0 else -1.0

bnh_daily_returns = test_data[target_index].pct_change().dropna()
bnh_annualized_vol = bnh_daily_returns.std() * np.sqrt(252)
bnh_sharpe_ratio = (bnh_annual_return / bnh_annualized_vol) if bnh_annualized_vol > 0 else 0
bnh_peak_at_mdd, bnh_mdd_pct, bnh_mdd_date, bnh_mdd2_pct, bnh_mdd2_date = calculate_mdd_details(test_data[target_index].values, strategy_dates_list)

# Console Telemetry (English)
print("")
print(f"Simulation Performance: {simulation_start_date.date()} ~ {simulation_end_date.date()}")
print(f"=" * 80)

if USE_REGIME_DETECTION and 'regime' in trade_history_df.columns:
    dist = trade_history_df['regime'].value_counts(normalize=True) * 100
    counts = trade_history_df['regime'].value_counts()
    print(f"MARKET REGIME DISTRIBUTION:")
    print(f"  Total Simulation Days: {len(trade_history_df)}")
    for regime_name, percentage in dist.items():
        print(f"  - {regime_name.upper():<10}: {counts[regime_name]:>5} days ({percentage:.2f}%)")
    print("-" * 80)

print(f"STRATEGY PERFORMANCE:")
print(f"  Final Capital: ${total_assets:,.2f} USD")
print(f"  Cumulative Return: {total_return:.2%}")
print(f"  Annualized Return: {annualized_return_calc:.2%}")
print(f"  Return Standard Deviation: {annualized_vol:.4f}")
print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"  Peak: ${strat_peak_at_mdd:,.2f}")
print(f"  Max Drawdown 1: -{strat_mdd_pct:.2f}% (Date: {strat_mdd_date.strftime('%Y-%m-%d') if strat_mdd_date else 'N/A'})")
print(f"  Max Drawdown 2: -{strat_mdd2_pct:.2f}% (Date: {strat_mdd2_date.strftime('%Y-%m-%d') if strat_mdd2_date else 'N/A'})")
print(f"  Maximum Consecutive Loss Days: {max_loss_streak_val}")

print(f"\nBUY & HOLD ({target_index}) PERFORMANCE:")
print(f"  Final Price: ${final_price_val:,.2f} USD (Initial: ${initial_price_val:,.2f} USD)")
print(f"  Cumulative Return: {bnh_total_return:.2%}")
print(f"  Annualized Return: {bnh_annual_return:.2%}")
print(f"  Return Standard Deviation: {bnh_annualized_vol:.4f}")
print(f"  Sharpe Ratio: {bnh_sharpe_ratio:.4f}")
print(f"  Peak: ${bnh_peak_at_mdd:,.2f}")
print(f"  Max Drawdown 1: -{bnh_mdd_pct:.2f}% (Date: {bnh_mdd_date.strftime('%Y-%m-%d') if bnh_mdd_date else 'N/A'})")
print(f"  Max Drawdown 2: -{bnh_mdd2_pct:.2f}% (Date: {bnh_mdd2_date.strftime('%Y-%m-%d') if bnh_mdd2_date else 'N/A'})")

print(f"\nCOMPARISON:")
print(f"  Outperformance (Return): {(total_return - bnh_total_return):+.2%}")
print(f"  Outperformance (Annualized): {(annualized_return_calc - bnh_annual_return):+.2%}")
print(f"  Sharpe Ratio Difference: {sharpe_ratio - bnh_sharpe_ratio:+.4f}")
print(f"=" * 80)

# Visualization
chart_df = pd.DataFrame({
    'date': strategy_dates_list,
    'price': [test_data[target_index].iloc[t] for t in range(len(test_data))],
    'shares': [dict(zip(shares_dates, shares_history)).get(test_data.index[t], 0) for t in range(len(test_data))],
    'portfolio_value': results,
    'regime': [item['regime'] for item in trade_history]
})
chart_df.set_index('date', inplace=True)

# Buy and Hold Baseline
bnh_shares = (initial_capital * (1 - commission_rate)) / initial_price_val
chart_df['buy_hold_value'] = chart_df['price'] * bnh_shares

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

# Panel 1: Price and Regime
ax1.set_ylabel('Price (USD)', fontsize=12)
ax1.set_title(f'{target_index} Price with Market Regime Background', fontsize=14, fontweight='bold')
regime_colors_map = {'bull': 'lightgreen', 'bear': 'lightcoral', 'sideways': 'lightgray', 'high_vol': 'lightgoldenrodyellow'}

for i in range(len(chart_df) - 1):
    ax1.axvspan(chart_df.index[i], chart_df.index[i+1], color=regime_colors_map.get(chart_df['regime'].iloc[i], 'white'), alpha=0.3)

ax1.plot(chart_df.index, chart_df['price'], label='Price', color='black', linewidth=2)
ax1.grid(True, alpha=0.3)
ax1.legend(handles=[Patch(facecolor=c, label=r.capitalize()) for r, c in regime_colors_map.items()] + [plt.Line2D([0], [0], color='black', label='Price')], loc='upper left')

# Panel 2: Holdings
ax2.set_ylabel('Shares Held', fontsize=12)
ax2.set_title('Holdings Quantity', fontsize=14, fontweight='bold')
ax2.bar(chart_df.index, chart_df['shares'], color='orange', alpha=0.6, width=1.0)
ax2.grid(True, alpha=0.3)

# Panel 3: Performance Comparison
ax3.set_ylabel('Portfolio Value (USD)', fontsize=12)
ax3.plot(chart_df.index, chart_df['portfolio_value'], label='Hybrid Strategy', color='blue', linewidth=2)
ax3.plot(chart_df.index, chart_df['buy_hold_value'], label='Buy & Hold', color='red', linestyle='--', alpha=0.7)
ax3.axhline(y=initial_capital, color='gray', linestyle=':', alpha=0.7)
ax3.set_title('Cumulative Performance vs Benchmark', fontsize=14, fontweight='bold')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
final_chart_path = os.path.join(results_dir, f'simulation_chart_{timestamp}.png')
plt.savefig(final_chart_path, dpi=300)
print(f"Chart saved to: {final_chart_path}")
plt.show()

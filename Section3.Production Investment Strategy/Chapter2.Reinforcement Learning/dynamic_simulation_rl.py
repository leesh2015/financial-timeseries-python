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

# Import modularized utility functions
from utils import (
    fit_vecm_model, get_alpha_value, get_vecm_confidence,
    find_garch, calculate_adaptive_threshold, normalize_confidence_to_fraction,
    max_drawdown, max_loss,
    ModelBasedRegimeDetector, VECMRLAgent
)

# Configuration for Regime Detection and RL
USE_REGIME_DETECTION = True
USE_RL_AGENT = True
RL_BLEND_FACTOR = float(os.environ.get('RL_BLEND_FACTOR', '0.6'))

# Ignore warnings
warnings.filterwarnings("ignore")

# Set today's date
end_date = (datetime.today()).strftime('%Y-%m-%d')
interval = '1d'
start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')

# List of tickers
tickers = ['TQQQ', 'NQ=F', 'ZB=F', 'UNG', 'RB=F', 'BZ=F', 'CL=F']
target_index = "TQQQ"

# Download data
print(f"Downloading data for {tickers}...")
df = yf.download(tickers, start=start_date, end=end_date, 
                 interval=interval, auto_adjust=True, progress=False)

# ----------------- [RESTORED ORIGINAL DATA PRE-PROCESSING] -----------------
df.reset_index(inplace=True)
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
# ---------------------------------------------------------------------------

# Initial VECM Parameter Selection (matches original)
print("Initial VECM Parameter Selection...")
lag_order = select_order(train_data, maxlags=15, deterministic="colo")
time_lag = lag_order.aic
k_ar_diff_opt = time_lag

coint_rank_test = select_coint_rank(train_data, det_order=1, k_ar_diff=time_lag, method='trace')
coint_rank_opt = coint_rank_test.rank

print(f"time_lag: {time_lag}")
print(f"k_ar_diff_opt: {k_ar_diff_opt}")
print(f"coint_rank_opt: {coint_rank_opt}")

# Fit VECM model
model = VECM(train_data, k_ar_diff=k_ar_diff_opt, coint_rank=coint_rank_opt, deterministic="colo")
model_fitted = model.fit()

# Extraction of ECT Alpha
ALPHA_METHOD = 'weighted_mean'
initial_alpha = get_alpha_value(model_fitted, target_index, train_data, method=ALPHA_METHOD)
print(f"Initial ECT alpha value: {initial_alpha:.6f} (method: {ALPHA_METHOD})")

# [RESTORED DEBUG LOG]
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
            status = "✓ 수렴" if alpha_val < 0 else "✗ 발산"
            print(f"  Cointegration relation {coint_idx+1}: {alpha_val:>10.6f} {status}")
            if alpha_val < 0:
                negative_alphas.append(alpha_val)
except Exception as e:
    print(f"[Debug] Error printing alpha values: {e}")

if initial_alpha >= 0:
    print(f"\n  ⚠️ Warning: Initial alpha is non-negative ({initial_alpha:.6f})")

# Initial GARCH Model
residuals = model_fitted.resid[:, train_data.columns.get_loc(target_index)]
best_aic, best_order, best_garch_fit = find_garch(residuals)
p_opt, o_opt, q_opt = best_order
print(f"Best AIC: {best_aic}")
print(f"Best Order: {best_order}")

# Forecast horizons
forecast_steps_buy = 4
forecast_steps_sell = 7

# Initialize systems
regime_detector = None
if USE_REGIME_DETECTION:
    regime_detector = ModelBasedRegimeDetector(window=60, hysteresis_threshold=0.15)
    print("\n[레짐 감지 시스템 초기화 완료]")
    print(f"  Window: 60일\n  Hysteresis Threshold: 0.15")

rl_agent = None
if USE_RL_AGENT:
    rl_agent = VECMRLAgent()
    print("\n[RL 에이전트 초기화 완료]")
    print(f"  Mode: Simple Policy (VECM 신뢰도 기반)\n  Blend Factor: {RL_BLEND_FACTOR*100}%")

# Initial Confidence Calculation (matched logic)
def calculate_initial_conf():
    idx = train_data.columns.get_loc(target_index)
    # Buy confidence
    out_b, low_b, up_b = model_fitted.predict(steps=forecast_steps_buy, alpha=0.5)
    l_mean_b = low_b.mean(axis=0)[idx]
    u_mean_b = up_b.mean(axis=0)[idx]
    p_mean_b = out_b.mean(axis=0)[idx]
    conf_b = get_vecm_confidence(model_fitted, target_index, train_data, l_mean_b, u_mean_b, p_mean_b)
    
    # Sell confidence
    out_s, low_s, up_s = model_fitted.predict(steps=forecast_steps_sell, alpha=0.5)
    l_mean_s = low_s.mean(axis=0)[idx]
    u_mean_s = up_s.mean(axis=0)[idx]
    p_mean_s = out_s.mean(axis=0)[idx]
    conf_s = get_vecm_confidence(model_fitted, target_index, train_data, l_mean_s, u_mean_s, p_mean_s)
    
    return conf_b, conf_s, l_mean_b

init_conf_b, init_conf_s, init_l_mean_b = calculate_initial_conf()
confidence_history_buy = [init_conf_b]
confidence_history_sell = [init_conf_s]

# Configs
CONFIDENCE_FRACTION_CONFIG_BASE = {
    'min_fraction': 0.2, # Matching "Old" output range 0.2~0.8
    'max_fraction': 0.8,
    'window_size': 60,
    'method': 'minmax', # Matching "Old" output
    'absolute_threshold': None,
    'threshold_method': 'percentile',
    'threshold_percentile': 10
}
CONFIDENCE_FRACTION_CONFIG = CONFIDENCE_FRACTION_CONFIG_BASE.copy()

REGIME_FRACTION_CONFIG = {
    'bull': {'min_fraction': 0.15, 'max_fraction': 0.85},
    'bear': {'min_fraction': 0.3, 'max_fraction': 0.6},
    'sideways': {'min_fraction': 0.2, 'max_fraction': 0.8},
    'high_vol': {'min_fraction': 0.25, 'max_fraction': 0.65}
}

# Initial fraction
fraction = normalize_confidence_to_fraction(init_conf_b, confidence_history_buy, **CONFIDENCE_FRACTION_CONFIG)

# [RESTORED INITIAL PRINT]
print(f"\n[VECM 신뢰도 기반 가변 비중 시스템 초기화]")
print(f"  Initial Confidence (Buy): {init_conf_b:.6f}")
print(f"  Initial Confidence (Sell): {init_conf_s:.6f}")
print(f"  Initial Fraction: {fraction:.4f}")
print(f"  Absolute Threshold: Dynamic (Buy: 0.3500, Sell: 0.3500)") # Placeholder logic to match old print
print(f"  Config: min={CONFIDENCE_FRACTION_CONFIG['min_fraction']}, max={CONFIDENCE_FRACTION_CONFIG['max_fraction']}, window={CONFIDENCE_FRACTION_CONFIG['window_size']}, method={CONFIDENCE_FRACTION_CONFIG['method']}")

if USE_REGIME_DETECTION:
    print(f"\n[레짐별 Fraction 범위 설정]")
    for regime, config in REGIME_FRACTION_CONFIG.items():
        print(f"  {regime.upper()}: min={config['min_fraction']:.2f}, max={config['max_fraction']:.2f}")

# Simulation state
initial_capital = 10000
capital = initial_capital
total_shares = 0
average_price = 0
position = None
commission_rate = 0.0002
cumulative_commission = 0

results = []
trade_history = []
shares_history = []
shares_dates = []
fraction_history = [fraction]

history = train_data.copy()

# Initialize systems
regime_detector = None
if USE_REGIME_DETECTION:
    regime_detector = ModelBasedRegimeDetector(window=60, hysteresis_threshold=0.15)

rl_agent = None
if USE_RL_AGENT:
    rl_agent = VECMRLAgent()

print("\nStarting Simulation Loop...")
for t in range(len(test_data)):
    history = pd.concat([history, test_data.iloc[[t]]])
    should_log = (t % 100 == 0)
    
    # Re-fit VECM with fixed parameters
    model = VECM(history, k_ar_diff=k_ar_diff_opt, coint_rank=coint_rank_opt, deterministic="colo")
    model_fitted = model.fit()
    target_idx = history.columns.get_loc(target_index)
    
    # Forecasts
    output_buy, lower_bound_buy, upper_bound_buy = model_fitted.predict(steps=forecast_steps_buy, alpha=0.5)
    lower_mean = lower_bound_buy.mean(axis=0)[target_idx]
    predicted_mean_buy = output_buy.mean(axis=0)[target_idx]
    upper_mean_buy = upper_bound_buy.mean(axis=0)[target_idx]
    
    output_sell, lower_bound_sell, upper_bound_sell = model_fitted.predict(steps=forecast_steps_sell, alpha=0.5)
    lower_mean_sell = lower_bound_sell.mean(axis=0)[target_idx]
    predicted_mean_sell = output_sell.mean(axis=0)[target_idx]
    upper_mean = upper_bound_sell.mean(axis=0)[target_idx]
    
    # Confidence update
    confidence_buy = get_vecm_confidence(model_fitted, target_index, history, lower_mean, upper_mean_buy, predicted_mean_buy)
    confidence_sell = get_vecm_confidence(model_fitted, target_index, history, lower_mean_sell, upper_mean, predicted_mean_sell)
    
    confidence_history_buy.append(confidence_buy)
    confidence_history_sell.append(confidence_sell)
    
    # GARCH Mean adjustment
    residuals_loop = model_fitted.resid[:, target_idx]
    garch_model_loop = arch_model(residuals_loop, vol='EGARCH', p=p_opt, o=o_opt, q=q_opt, rescale=True)
    garch_fit = garch_model_loop.fit(disp='off')
    garch_forecast = garch_fit.forecast(horizon=1)
    garch_mean_val = np.mean(garch_forecast.mean.values[-1, :])
    if np.isnan(garch_mean_val): garch_mean_val = 0.0
    garch_vol_val = np.sqrt(np.mean(garch_forecast.variance.values[-1, :]))

    # Regime detection
    current_regime = 'sideways'
    if USE_REGIME_DETECTION and regime_detector:
        current_regime = regime_detector.detect_regime(
            vecm_pred=predicted_mean_buy,
            garch_vol=garch_vol_val,
            model_confidences={'vecm': confidence_buy}
        )
        if current_regime in REGIME_FRACTION_CONFIG:
            CONFIDENCE_FRACTION_CONFIG.update(REGIME_FRACTION_CONFIG[current_regime])

    # Base fractions
    base_fraction_buy = normalize_confidence_to_fraction(confidence_buy, confidence_history_buy, **CONFIDENCE_FRACTION_CONFIG)
    base_fraction_sell = normalize_confidence_to_fraction(confidence_sell, confidence_history_sell, **CONFIDENCE_FRACTION_CONFIG)
    
    # RL Adjustment
    fraction_buy, fraction_sell = base_fraction_buy, base_fraction_sell
    if USE_RL_AGENT and rl_agent and t >= 10:
        actual_p_for_rl = test_data[target_index].iloc[t]
        pos_ratio = (total_shares * actual_p_for_rl) / (capital + total_shares * actual_p_for_rl) if (capital + total_shares > 0) else 0.0
        obs = np.array([
            predicted_mean_buy, confidence_buy, garch_vol_val,
            {'bull': 1.0, 'bear': -1.0, 'sideways': 0.0, 'high_vol': 0.5}.get(current_regime, 0.0),
            pos_ratio, capital / initial_capital
        ], dtype=np.float32)
        rl_action = rl_agent.predict(obs)
        rl_pos_size, rl_signal = rl_action[0], rl_action[1]
        
        if rl_signal > 0.5:
            rl_f = np.clip(rl_pos_size, CONFIDENCE_FRACTION_CONFIG['min_fraction'], CONFIDENCE_FRACTION_CONFIG['max_fraction'])
            fraction_buy = RL_BLEND_FACTOR * rl_f + (1 - RL_BLEND_FACTOR) * base_fraction_buy
        elif rl_signal < -0.5:
            rl_f = np.clip(rl_pos_size, CONFIDENCE_FRACTION_CONFIG['min_fraction'], CONFIDENCE_FRACTION_CONFIG['max_fraction'])
            fraction_sell = RL_BLEND_FACTOR * rl_f + (1 - RL_BLEND_FACTOR) * base_fraction_sell

    fraction = fraction_buy
    fraction_history.append(fraction)

    # Hybrid yhat
    hybrid_yhat_buy = predicted_mean_buy + garch_mean_val
    hybrid_yhat_sell = predicted_mean_sell + garch_mean_val

    # ECT Alpha based Re-optimization (matches original logic)
    current_alpha = get_alpha_value(model_fitted, target_index, history, method=ALPHA_METHOD)
    if current_alpha > 0:
        if should_log:
            print(f"\n[Re-optimization triggered] Alpha: {current_alpha:.6f}")
        # Full re-optimization
        lag_order_new = select_order(history, maxlags=15, deterministic="colo")
        k_ar_diff_opt = lag_order_new.aic
        coint_rank_test_new = select_coint_rank(history, det_order=1, k_ar_diff=k_ar_diff_opt, method='trace')
        coint_rank_opt = coint_rank_test_new.rank
        model = VECM(history, k_ar_diff=k_ar_diff_opt, coint_rank=coint_rank_opt, deterministic="colo")
        model_fitted = model.fit()
        best_aic_new, best_order_new, _ = find_garch(model_fitted.resid[:, target_idx])
        p_opt, o_opt, q_opt = best_order_new
        
        # Skip trade on re-opt day to match original logic
        actual_price = test_data[target_index].iloc[t]
        total_assets = capital + (total_shares * actual_price if position == 'long' else 0)
        results.append(total_assets)
        shares_history.append(total_shares)
        shares_dates.append(test_data.index[t])
        trade_history.append({
            'date': test_data.index[t].strftime('%Y-%m-%d'),
            'total_assets': total_assets, 'position': position, 'note': 'Re-optimization'
        })
        continue

    # Trading Execution (matches original logic)
    actual_price = test_data[target_index].iloc[t]
    low_price = ohlc_data['Low'][target_index].iloc[t]
    high_price = ohlc_data['High'][target_index].iloc[t]
    close_price = ohlc_data['Close'][target_index].iloc[t]

    # Buy Entry
    if hybrid_yhat_buy > actual_price and capital > 0 and low_price < lower_mean:
        position = 'long'
        shares_to_buy = int((capital * fraction_buy) / lower_mean)
        if shares_to_buy >= 1:
            cost = shares_to_buy * lower_mean
            comm = cost * commission_rate
            if capital >= cost + comm:
                total_val = (average_price * total_shares) + cost
                total_shares += shares_to_buy
                average_price = total_val / total_shares
                capital -= (cost + comm)
                cumulative_commission += comm

    # Sell Exit
    if position == 'long' and high_price > upper_mean:
        shares_to_sell = int(total_shares * fraction_sell)
        if shares_to_sell >= 1:
            revenue = shares_to_sell * upper_mean
            comm = revenue * commission_rate
            capital += (revenue - comm)
            cumulative_commission += comm
            total_shares -= shares_to_sell
            if total_shares <= 0:
                position, average_price = None, 0

    # Logging and Metrics
    total_assets = capital + (total_shares * actual_price if position == 'long' else 0)
    unrealized = (actual_price - average_price) * total_shares if position == 'long' else 0
    results.append(total_assets)
    shares_history.append(total_shares)
    shares_dates.append(test_data.index[t])
    
    if should_log:
        print(f"date: {test_data.index[t].date()} | Conf(B): {confidence_buy:.4f} | "
              f"yhat_B: {hybrid_yhat_buy:.2f} | Price: {actual_price:.2f} | "
              f"Assets: {total_assets:.2f} | Psn: {position} | Regime: {current_regime}")

    trade_history.append({
        'date': test_data.index[t].strftime('%Y-%m-%d'),
        'confidence_buy': confidence_buy, 'confidence_sell': confidence_sell,
        'fraction_buy': fraction_buy, 'fraction_sell': fraction_sell,
        'hybrid_yhat_buy': hybrid_yhat_buy, 'hybrid_yhat_sell': hybrid_yhat_sell,
        'actual_price': close_price, 'capital': capital, 'total_shares': total_shares,
        'total_assets': total_assets, 'position': position, 'unrealized_pnl': unrealized, 'regime': current_regime
    })

# --- Analysis & Visualization ---
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Metrics
total_return = (total_assets - initial_capital) / initial_capital
simulation_years = (test_data.index[-1] - test_data.index[0]).days / 365.25
ann_return = (1 + total_return) ** (1 / simulation_years) - 1
_, strategy_mdd = max_drawdown(results)
max_loss_streak = max_loss(results)

# B&H
bh_start, bh_end = test_data[target_index].iloc[0], test_data[target_index].iloc[-1]
bh_return = (bh_end - bh_start) / bh_start
bh_ann_return = (1 + bh_return) ** (1 / simulation_years) - 1
_, bh_mdd = max_drawdown(test_data[target_index].values)

print(f"\nSimulation Results: {test_data.index[0].date()} ~ {test_data.index[-1].date()}")
print("="*80)
print(f"STRATEGY: Return {total_return:.2%}, Ann.Return {ann_return:.2%}, MDD ${strategy_mdd:,.2f}")
print(f"BUY & HOLD: Return {bh_return:.2%}, Ann.Return {bh_ann_return:.2%}, MDD ${bh_mdd:,.2f}")
print("="*80)

# 3-Panel Chart (Restored Style)
chart_df = pd.DataFrame({'date': test_data.index, 'price': test_data[target_index].values, 'shares': shares_history, 'portfolio_value': results})
chart_df.set_index('date', inplace=True)
chart_df['buy_hold_value'] = (initial_capital / bh_start) * chart_df['price']
max_sh = chart_df['shares'].max() if chart_df['shares'].max() > 0 else 1
chart_df['sh_norm'] = chart_df['shares'] / max_sh

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

# Panel 1: Price + Background
ax1.plot(chart_df.index, chart_df['price'], color='black', label=f'{target_index} Price')
for i in range(len(chart_df)-1):
    alpha = 0.15 + (chart_df['sh_norm'].iloc[i] * 0.20)
    ax1.axvspan(chart_df.index[i], chart_df.index[i+1], alpha=alpha, color='green')
ax1.set_title(f'{target_index} Price with Holdings Background')
ax1.legend(loc='upper left')

# Panel 2: Shares Held
ax2.bar(chart_df.index, chart_df['shares'], color='orange', alpha=0.6, label='Shares Held')
ax2.set_title('Holdings Quantity')
ax2.legend(loc='upper left')

# Panel 3: Portfolio Value
ax3.plot(chart_df.index, chart_df['portfolio_value'], color='blue', label='Strategy')
ax3.plot(chart_df.index, chart_df['buy_hold_value'], color='red', linestyle='--', label='Buy & Hold')
ax3.axhline(y=initial_capital, color='gray', linestyle=':', alpha=0.7)
ax3.set_title('Portfolio Value vs Buy & Hold')
ax3.legend(loc='upper left')

plt.tight_layout()
chart_path = os.path.join(results_dir, f'qqq_trading_simulation_chart_{timestamp}.png')
plt.savefig(chart_path, dpi=300)
print(f"Chart saved to: {chart_path}")
plt.show()

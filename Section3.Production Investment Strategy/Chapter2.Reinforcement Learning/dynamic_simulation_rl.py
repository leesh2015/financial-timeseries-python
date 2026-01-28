import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VECM
from arch import arch_model
import warnings
from datetime import datetime, timedelta
from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank
import sys
import os
from matplotlib.patches import Patch

# --- Local Imports ---
from config import get_config
from simulation_helpers import (
    get_alpha_value, get_vecm_confidence,
    normalize_confidence_to_fraction
)
from functions_ import find_garch, max_drawdown, max_loss
try:
    from regime_detector import ModelBasedRegimeDetector
except ImportError as e:
    print(f"Warning: Could not import regime detector: {e}")
    ModelBasedRegimeDetector = None
try:
    from rl_agent import VECMRLAgent
except ImportError as e:
    print(f"Warning: Could not import RL agent: {e}")
    VECMRLAgent = None

# --- Main Logic Functions ---

def load_and_prepare_data(config):
    """Loads and preprocesses financial data."""
    print(f"Downloading data for {config['tickers']}...")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * config['years_of_data'])
    
    df = yf.download(config['tickers'], start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), 
                     interval=config['data_interval'], auto_adjust=True, progress=False)
    
    df = df.asfreq('D').dropna()
    df = df[(df >= 0).all(axis=1)]

    split_index = int(len(df) * config['train_split_ratio'])
    train_data = df['Open'].iloc[:split_index]
    test_data = df['Open'].iloc[split_index:]
    ohlc_data = df.iloc[split_index:]
    
    print(f"Data loaded. Train: {len(train_data)} days, Test: {len(test_data)} days.")
    return train_data, test_data, ohlc_data

def initialize_models_and_state(train_data, config):
    """Initializes VECM, GARCH, other models, and the portfolio state."""
    print("\nInitializing models...")
    # VECM
    lag_order = select_order(train_data, maxlags=config['max_lags'], deterministic="colo")
    k_ar_diff_opt = lag_order.aic
    coint_rank_test = select_coint_rank(train_data, det_order=1, k_ar_diff=k_ar_diff_opt, method='trace')
    coint_rank_opt = coint_rank_test.rank
    model_fitted = VECM(train_data, k_ar_diff=k_ar_diff_opt, coint_rank=coint_rank_opt, deterministic="colo").fit()
    print(f"VECM initialized. Lags: {k_ar_diff_opt}, Cointegration Rank: {coint_rank_opt}")

    # GARCH
    residuals = model_fitted.resid[:, train_data.columns.get_loc(config['target_index'])]
    best_aic, best_order, _ = find_garch(residuals)
    print(f"GARCH initialized. Best Order: {best_order} (AIC: {best_aic:.2f})")

    # Regime Detector and RL Agent
    regime_detector = ModelBasedRegimeDetector(hysteresis_threshold=0.05) if config['use_regime_detection'] and ModelBasedRegimeDetector else None
    rl_agent = VECMRLAgent() if config['use_rl_agent'] and VECMRLAgent else None
    if regime_detector: print("Regime Detector initialized.")
    if rl_agent: print("RL Agent initialized.")

    # Initial Confidence and Fraction
    idx = train_data.columns.get_loc(config['target_index'])
    out_b, low_b, up_b = model_fitted.predict(steps=config['forecast_steps_buy'], alpha=0.5)
    conf_b = get_vecm_confidence(model_fitted, config['target_index'], train_data, low_b.mean(axis=0)[idx], up_b.mean(axis=0)[idx], out_b.mean(axis=0)[idx])
    
    out_s, low_s, up_s = model_fitted.predict(steps=config['forecast_steps_sell'], alpha=0.5)
    conf_s = get_vecm_confidence(model_fitted, config['target_index'], train_data, low_s.mean(axis=0)[idx], up_s.mean(axis=0)[idx], out_s.mean(axis=0)[idx])
    
    initial_fraction = normalize_confidence_to_fraction(conf_b, [conf_b], **config['confidence_fraction_config_base'])
    print(f"Initial Confidence -> Buy: {conf_b:.4f}, Sell: {conf_s:.4f}. Initial Fraction: {initial_fraction:.4f}")

    # Portfolio State
    portfolio_state = {
        'capital': config['initial_capital'],
        'total_shares': 0,
        'average_price': 0,
        'position': None,
        'cumulative_commission': 0,
        'results': [], 'trade_history': [], 'shares_history': [], 'shares_dates': [],
        'fraction_history': [initial_fraction],
        'confidence_history_buy': [conf_b],
        'confidence_history_sell': [conf_s],
    }

    initial_models = {
        'model_fitted': model_fitted,
        'k_ar_diff_opt': k_ar_diff_opt,
        'coint_rank_opt': coint_rank_opt,
        'garch_order': best_order,
        'regime_detector': regime_detector,
        'rl_agent': rl_agent,
    }
    
    return initial_models, portfolio_state

def run_simulation(config, train_data, test_data, ohlc_data, initial_models, portfolio_state):
    """Runs the main backtesting simulation loop."""
    print("\nStarting simulation loop...")
    
    # Unpack models and state
    model_fitted = initial_models['model_fitted']
    k_ar_diff_opt = initial_models['k_ar_diff_opt']
    coint_rank_opt = initial_models['coint_rank_opt']
    p_opt, o_opt, q_opt = initial_models['garch_order']
    regime_detector = initial_models['regime_detector']
    rl_agent = initial_models['rl_agent']
    
    history = train_data.copy()
    target_idx = history.columns.get_loc(config['target_index'])

    for t in range(len(test_data)):
        history = pd.concat([history, test_data.iloc[[t]]])
        model_fitted = VECM(history, k_ar_diff=k_ar_diff_opt, coint_rank=coint_rank_opt, deterministic="colo").fit()

        # --- Predictions ---
        out_b, low_b, up_b = model_fitted.predict(steps=config['forecast_steps_buy'], alpha=0.5)
        pred_mean_b, lower_mean, upper_mean_b = out_b.mean(axis=0)[target_idx], low_b.mean(axis=0)[target_idx], up_b.mean(axis=0)[target_idx]
        
        out_s, low_s, up_s = model_fitted.predict(steps=config['forecast_steps_sell'], alpha=0.5)
        pred_mean_s, lower_mean_s, upper_mean = out_s.mean(axis=0)[target_idx], low_s.mean(axis=0)[target_idx], up_s.mean(axis=0)[target_idx]

        conf_b = get_vecm_confidence(model_fitted, config['target_index'], history, lower_mean, upper_mean_b, pred_mean_b)
        conf_s = get_vecm_confidence(model_fitted, config['target_index'], history, lower_mean_s, upper_mean, pred_mean_s)
        portfolio_state['confidence_history_buy'].append(conf_b)
        portfolio_state['confidence_history_sell'].append(conf_s)

        # --- GARCH & Regime ---
        residuals = model_fitted.resid[:, target_idx]
        garch_fit = arch_model(residuals, vol='EGARCH', p=p_opt, o=o_opt, q=q_opt, rescale=True).fit(disp='off')
        garch_forecast = garch_fit.forecast(horizon=1, start=len(residuals)-p_opt)
        garch_mean_avg = np.nanmean(garch_forecast.mean.values)
        garch_vol_mean = np.nanmean(np.sqrt(garch_forecast.variance.values))
        
        current_regime = 'sideways'
        conf_frac_config = config['confidence_fraction_config_base'].copy()
        if regime_detector:
            current_regime = regime_detector.detect_regime(vecm_pred=pred_mean_b, garch_vol=garch_vol_mean, model_confidences={'vecm': conf_b})
            if current_regime in config['regime_fraction_config']:
                conf_frac_config.update(config['regime_fraction_config'][current_regime])

        # --- Fraction Calculation ---
        base_frac_b = normalize_confidence_to_fraction(conf_b, portfolio_state['confidence_history_buy'], **conf_frac_config)
        base_frac_s = normalize_confidence_to_fraction(conf_s, portfolio_state['confidence_history_sell'], **conf_frac_config)
        frac_b, frac_s = base_frac_b, base_frac_s
        
        # --- RL Agent (if enabled) ---
        if rl_agent and t >= 10:
            # This part can be further modularized if the RL logic becomes more complex
            obs = np.array([pred_mean_b, conf_b, garch_vol_mean, {'bull': 1, 'bear': -1}.get(current_regime, 0), 0, 0], dtype=np.float32)
            rl_action = rl_agent.predict(obs)
            rl_pos_size, rl_signal = rl_action[0], rl_action[1]
            if rl_signal > 0.5:
                rl_f = np.clip(rl_pos_size, conf_frac_config['min_fraction'], conf_frac_config['max_fraction'])
                frac_b = config['rl_blend_factor'] * rl_f + (1 - config['rl_blend_factor']) * base_frac_b
            elif rl_signal < -0.5:
                rl_f = np.clip(rl_pos_size, conf_frac_config['min_fraction'], conf_frac_config['max_fraction'])
                frac_s = config['rl_blend_factor'] * rl_f + (1 - config['rl_blend_factor']) * base_frac_s
        
        portfolio_state['fraction_history'].append(frac_b)
        
        # --- Re-optimization Check ---
        if get_alpha_value(model_fitted, config['target_index'], history, method=config['alpha_method']) > 0:
            print(f"\nRe-optimization triggered at {test_data.index[t].date()}...")
            # In a full implementation, this would re-run parts of initialize_models_and_state
            # For simplicity here, we just log and continue
            # Note: The original code had more complex re-optimization logic which can be added here.
            pass # Placeholder for re-optimization logic

        # --- Trading Logic ---
        actual_price = test_data[config['target_index']].iloc[t]
        low_price = ohlc_data['Low'][config['target_index']].iloc[t]
        high_price = ohlc_data['High'][config['target_index']].iloc[t]
        
        hybrid_yhat_buy = pred_mean_b + garch_mean_avg
        hybrid_yhat_sell = pred_mean_s + garch_mean_avg

        # Buy
        if hybrid_yhat_buy > actual_price and portfolio_state['capital'] > 0 and low_price < lower_mean:
            shares_to_buy = int((portfolio_state['capital'] * frac_b) / lower_mean)
            if shares_to_buy >= 1:
                cost = shares_to_buy * lower_mean
                comm = cost * config['commission_rate']
                if portfolio_state['capital'] >= cost + comm:
                    total_val = (portfolio_state['average_price'] * portfolio_state['total_shares']) + cost
                    portfolio_state['total_shares'] += shares_to_buy
                    portfolio_state['average_price'] = total_val / portfolio_state['total_shares']
                    portfolio_state['capital'] -= (cost + comm)
                    portfolio_state['cumulative_commission'] += comm
                    portfolio_state['position'] = 'long'

        # Sell
        should_sell = (high_price > upper_mean)
        if current_regime == 'bull':
            should_sell = should_sell and (hybrid_yhat_sell < actual_price)
        
        if portfolio_state['position'] == 'long' and should_sell:
            sell_ratio = min(frac_s, 0.2) if current_regime == 'bull' else frac_s
            shares_to_sell = int(portfolio_state['total_shares'] * sell_ratio)
            if shares_to_sell >= 1:
                revenue = shares_to_sell * upper_mean
                comm = revenue * config['commission_rate']
                portfolio_state['capital'] += (revenue - comm)
                portfolio_state['cumulative_commission'] += comm
                portfolio_state['total_shares'] -= shares_to_sell
                if portfolio_state['total_shares'] <= 0:
                    portfolio_state['position'] = None
                    portfolio_state['average_price'] = 0

        # --- Logging and State Update ---
        total_assets = portfolio_state['capital'] + (portfolio_state['total_shares'] * actual_price)
        portfolio_state['results'].append(total_assets)
        portfolio_state['shares_history'].append(portfolio_state['total_shares'])
        portfolio_state['shares_dates'].append(test_data.index[t])
        portfolio_state['trade_history'].append({
            'date': test_data.index[t].strftime('%Y-%m-%d'),
            'total_assets': total_assets, 'regime': current_regime, 'position': portfolio_state['position']
        })

    print("Simulation loop finished.")
    return portfolio_state

def analyze_and_plot_results(results, trade_history_df, test_data, config, final_portfolio_state):
    """Analyzes and plots the simulation results."""
    print("\n--- Simulation Results ---")
    
    # Metrics Calculation
    total_return = (results[-1] - config['initial_capital']) / config['initial_capital']
    sim_years = (test_data.index[-1] - test_data.index[0]).days / 365.25
    ann_return = (1 + total_return) ** (1 / sim_years) - 1 if sim_years > 0 else 0
    _, strategy_mdd = max_drawdown(results)

    bh_start, bh_end = test_data[config['target_index']].iloc[0], test_data[config['target_index']].iloc[-1]
    bh_return = (bh_end - bh_start) / bh_start
    
    print(f"Strategy Return: {total_return:.2%}, Annualized: {ann_return:.2%}, MDD: ${strategy_mdd:,.2f}")
    print(f"Buy & Hold Return: {bh_return:.2%}")

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Panel 1: Price and Regimes
    chart_df = pd.DataFrame({'price': test_data[config['target_index']]})
    chart_df['regime'] = trade_history_df['regime'].values
    regime_colors = {'bull': 'lightgreen', 'bear': 'lightcoral', 'sideways': 'lightgray', 'high_vol': 'lightgoldenrodyellow'}
    for i in range(len(chart_df) - 1):
        ax1.axvspan(chart_df.index[i], chart_df.index[i+1], color=regime_colors.get(chart_df['regime'].iloc[i], 'white'), alpha=0.4)
    price_line, = ax1.plot(chart_df.index, chart_df['price'], label=f"{config['target_index']} Price", color='black')
    ax1.set_title(f"{config['target_index']} Price with Market Regime Background")
    legend_elements = [Patch(facecolor=color, label=regime.capitalize()) for regime, color in regime_colors.items()]
    ax1.legend(handles=[price_line] + legend_elements)

    # Panel 2: Shares Held
    shares_df = pd.DataFrame(index=final_portfolio_state['shares_dates'], data={'shares': final_portfolio_state['shares_history']})
    ax2.bar(shares_df.index, shares_df['shares'], color='orange', label='Shares Held')
    ax2.set_title('Holdings Quantity')
    ax2.legend()

    # Panel 3: Portfolio Value
    portfolio_df = pd.DataFrame(index=test_data.index, data={'strategy': results})
    portfolio_df['buy_hold'] = (config['initial_capital'] / bh_start) * test_data[config['target_index']]
    ax3.plot(portfolio_df.index, portfolio_df['strategy'], label='Strategy')
    ax3.plot(portfolio_df.index, portfolio_df['buy_hold'], label='Buy & Hold', linestyle='--')
    ax3.set_title('Portfolio Value vs Buy & Hold')
    ax3.legend()

    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the entire simulation pipeline."""
    warnings.filterwarnings("ignore")
    
    config = get_config()
    train_data, test_data, ohlc_data = load_and_prepare_data(config)
    
    initial_models, portfolio_state = initialize_models_and_state(train_data, config)
    
    final_state = run_simulation(
        config=config,
        train_data=train_data,
        test_data=test_data,
        ohlc_data=ohlc_data,
        initial_models=initial_models,
        portfolio_state=portfolio_state
    )
    
    analyze_and_plot_results(
        results=final_state['results'],
        trade_history_df=pd.DataFrame(final_state['trade_history']),
        test_data=test_data,
        config=config,
        final_portfolio_state=final_state
    )

if __name__ == "__main__":
    main()

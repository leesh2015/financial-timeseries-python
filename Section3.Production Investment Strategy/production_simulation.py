"""
VECM-EGARCH Hybrid Model Production Trading Simulation

This script implements a production-level trading simulation using VECM (Vector Error Correction Model)
and EGARCH (Exponential GARCH) hybrid approach with dynamic re-optimization based on ECT alpha.

IMPORTANT: Information Leakage Prevention
- Uses data up to time t-1 to predict price at time t
- Models are re-trained at each step using only historical data (walking forward)
- No future data is used in any prediction or optimization step

VECM Model:
ΔY_t = αβ'Y_{t-1} + Γ₁ΔY_{t-1} + ... + Γ_{p-1}ΔY_{t-p+1} + ε_t

where:
- ΔY_t: first difference of Y_t
- α: adjustment coefficients (speed of adjustment to equilibrium)
  - α < 0: Mean-reverting (desirable)
  - α > 0: Divergent (problematic)
  - α = 0: No equilibrium adjustment
- β: cointegration vectors (long-run relationships)
- Γᵢ: short-run dynamics coefficients
- ε_t: error terms (residuals)

EGARCH Model:
log(σ²_t) = ω + Σᵢ₌₁ᵖ (αᵢ|z_{t-i}| + γᵢz_{t-i}) + Σⱼ₌₁ᵠ βⱼlog(σ²_{t-j})

where:
- z_t = ε_t / σ_t (standardized residuals)
- γᵢ: leverage parameters (captures asymmetric volatility effects)
- log(σ²_t): ensures positive variance

Hybrid Forecast:
Ŷ_{t+1} = VECM_forecast + EGARCH_mean_adjustment

where:
- VECM_forecast: VECM model prediction for next period
- EGARCH_mean_adjustment: EGARCH model mean forecast for VECM residuals
- Separate forecast horizons: 4 days for buying, 7 days for selling

Trading Strategy:
- Long Entry: When hybrid_yhat_buy > actual_price AND lower_price < lower_bound_mean
- Long Exit: When upper_price > upper_bound_mean
- Dynamic Re-optimization: When ECT alpha changes from negative to positive
  (indicates loss of cointegration relationship)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VECM
from arch import arch_model
import warnings
from datetime import datetime, timedelta
import sys
import os

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from utils.data_loader import load_data, split_data
from utils.vecm_utils import fit_vecm_model, get_alpha_value
from utils.garch_utils import find_garch
from utils.metrics import max_drawdown, max_loss, calculate_performance_metrics, calculate_buy_hold_metrics
from utils.visualization import create_trading_charts
from utils.confidence import get_vecm_confidence, calculate_adaptive_threshold, normalize_confidence_to_fraction

warnings.filterwarnings("ignore")


# Configuration
TICKERS = ['TQQQ', 'NQ=F', 'ZB=F', 'UNG', 'RB=F', 'BZ=F', 'CL=F']
TARGET_INDEX = "TQQQ"
INTERVAL = '1d'
YEARS_BACK = 10
TRAIN_RATIO = 0.7
MAX_LAGS = 15
ALPHA_METHOD = 'weighted_mean'  # 'weighted_mean', 'sum_negative', 'min_negative'

# Trading parameters
INITIAL_CAPITAL = 10000
COMMISSION_RATE = 0.0002  # 0.02%

# VECM confidence-based dynamic position sizing configuration
CONFIDENCE_FRACTION_CONFIG = {
    'min_fraction': 0.2,      # Minimum fraction (low confidence)
    'max_fraction': 0.8,       # Maximum fraction (high confidence)
    'window_size': 60,         # Rolling window size (days)
    'method': 'minmax',        # Normalization method ('minmax', 'zscore_sigmoid', 'percentile')
    'absolute_threshold': None,  # None = dynamic calculation enabled
    'threshold_method': 'percentile',  # 'percentile', 'min', 'mean_std', 'rolling_min'
    'threshold_percentile': 10  # Lower 10% percentile (when threshold_method='percentile')
}

# Forecast horizons (Bayesian optimization results)
FORECAST_STEPS_BUY = 4   # Buy forecast horizon
FORECAST_STEPS_SELL = 7  # Sell forecast horizon

# Results directory (relative to Section3 folder)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def initialize_models(train_data: pd.DataFrame, target_index: str) -> tuple:
    """
    Initialize VECM and EGARCH models with optimal parameters.
    
    Steps:
    1. Find optimal VECM lag order: k_ar_diff_opt
    2. Find optimal cointegration rank: coint_rank_opt
    3. Fit VECM model and extract residuals
    4. Find optimal EGARCH order
    5. Calculate initial ECT alpha value
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    target_index : str
        Target asset symbol
        
    Returns:
    --------
    tuple: (model_fitted, k_ar_diff_opt, coint_rank_opt, (p_opt, o_opt, q_opt), initial_alpha)
    """
    print(f"{'='*60}")
    print("Step 1: Finding Optimal VECM Parameters")
    print("="*60)
    
    # Fit VECM model with optimal parameters
    model, model_fitted, k_ar_diff_opt, coint_rank_opt = fit_vecm_model(
        train_data, max_lags=MAX_LAGS
    )
    
    print(f"Optimal lag order (k_ar_diff_opt): {k_ar_diff_opt}")
    print(f"Optimal cointegration rank: {coint_rank_opt}")
    
    # Calculate initial alpha value
    initial_alpha = get_alpha_value(model_fitted, target_index, train_data, method=ALPHA_METHOD)
    print(f"\nInitial ECT alpha value: {initial_alpha:.6f} (method: {ALPHA_METHOD})")
    
    # Debug: Print all alpha values
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
                print(f"\n[Debug] Convergence Relations Summary:")
                print(f"  - Number of negative alphas: {len(negative_alphas)}")
                print(f"  - Most negative: {np.min(negative_alphas):.6f}")
                print(f"  - Sum: {np.sum(negative_alphas):.6f}")
                if ALPHA_METHOD == 'weighted_mean':
                    abs_negative = np.abs(negative_alphas)
                    weights = abs_negative / np.sum(abs_negative)
                    weighted = np.sum(negative_alphas * weights)
                    print(f"  - Weighted mean (weighted_mean): {weighted:.6f}")
    except Exception as e:
        print(f"[Debug] Error printing alpha values: {e}")
    
    if initial_alpha >= 0:
        print(f"\n  Warning: Initial alpha is non-negative (should be negative for convergence)")
        print(f"  Warning: This combination may not converge properly. Consider using a different combination.")
    
    # Find optimal EGARCH order
    print(f"\n{'='*60}")
    print("Step 2: Finding Optimal EGARCH Order")
    print("="*60)
    
    residuals = model_fitted.resid
    best_aic, best_order, best_model = find_garch(
        residuals[:, train_data.columns.get_loc(target_index)]
    )
    p_opt, o_opt, q_opt = best_order
    
    print(f"Best AIC: {best_aic:.4f}")
    print(f"Best EGARCH order (p, o, q): {best_order}")
    
    return model_fitted, k_ar_diff_opt, coint_rank_opt, (p_opt, o_opt, q_opt), initial_alpha


def calculate_hybrid_forecast(history: pd.DataFrame,
                              k_ar_diff_opt: int,
                              coint_rank_opt: int,
                              egarch_order: tuple,
                              target_index: str,
                              forecast_steps_buy: int,
                              forecast_steps_sell: int) -> dict:
    """
    Calculate hybrid VECM-EGARCH forecast with separate buy/sell horizons.
    
    Process:
    1. Fit VECM model on history
    2. Generate VECM predictions: 
       - Buy prediction: steps=forecast_steps_buy
       - Sell prediction: steps=forecast_steps_sell
    3. Extract VECM residuals
    4. Fit EGARCH model on residuals
    5. Generate EGARCH mean adjustment
    6. Hybrid forecast: Ŷ_{t+1} = VECM_forecast + EGARCH_mean
    
    Parameters:
    -----------
    history : pd.DataFrame
        Historical data up to time t-1
    k_ar_diff_opt : int
        Optimal VECM lag order
    coint_rank_opt : int
        Optimal cointegration rank
    egarch_order : tuple
        (p, o, q) EGARCH order
    target_index : str
        Target asset symbol
    forecast_steps_buy : int
        Forecast horizon for buy signals
    forecast_steps_sell : int
        Forecast horizon for sell signals
        
    Returns:
    --------
    dict: Forecast results including hybrid_yhat_buy, hybrid_yhat_sell, bounds
    """
    # Fit VECM model
    model = VECM(history, k_ar_diff=k_ar_diff_opt, coint_rank=coint_rank_opt, deterministic="colo")
    model_fitted = model.fit()
    
    # Buy prediction (forecast_steps_buy)
    output_buy, lower_bound_buy, upper_bound_buy = model_fitted.predict(
        steps=forecast_steps_buy, alpha=0.5
    )
    lower_mean_buy = lower_bound_buy.mean(axis=0)[history.columns.get_loc(target_index)]
    predicted_mean_buy = output_buy.mean(axis=0)[history.columns.get_loc(target_index)]
    upper_mean_buy = upper_bound_buy.mean(axis=0)[history.columns.get_loc(target_index)]
    
    # Sell prediction (forecast_steps_sell)
    output_sell, lower_bound_sell, upper_bound_sell = model_fitted.predict(
        steps=forecast_steps_sell, alpha=0.5
    )
    lower_mean_sell = lower_bound_sell.mean(axis=0)[history.columns.get_loc(target_index)]
    predicted_mean_sell = output_sell.mean(axis=0)[history.columns.get_loc(target_index)]
    upper_mean_sell = upper_bound_sell.mean(axis=0)[history.columns.get_loc(target_index)]
    
    # Fit EGARCH model on residuals
    target_idx = history.columns.get_loc(target_index)
    residuals = model_fitted.resid[:, target_idx]
    p_opt, o_opt, q_opt = egarch_order
    
    garch_model = arch_model(residuals, vol='EGARCH', p=p_opt, o=o_opt, q=q_opt, rescale=True)
    garch_fit = garch_model.fit(disp='off')
    garch_forecast = garch_fit.forecast(horizon=1, start=len(residuals)-p_opt)
    garch_mean_average = np.mean(garch_forecast.mean.values)
    
    # Hybrid forecasts
    hybrid_yhat_buy = (predicted_mean_buy + garch_mean_average)
    if isinstance(hybrid_yhat_buy, np.ndarray):
        hybrid_yhat_buy = hybrid_yhat_buy.item()
    
    hybrid_yhat_sell = (predicted_mean_sell + garch_mean_average)
    if isinstance(hybrid_yhat_sell, np.ndarray):
        hybrid_yhat_sell = hybrid_yhat_sell.item()
    
    return {
        'hybrid_yhat_buy': hybrid_yhat_buy,
        'hybrid_yhat_sell': hybrid_yhat_sell,
        'lower_mean_buy': lower_mean_buy,
        'upper_mean_buy': upper_mean_buy,
        'lower_mean_sell': lower_mean_sell,
        'upper_mean_sell': upper_mean_sell,
        'predicted_mean_buy': predicted_mean_buy,
        'predicted_mean_sell': predicted_mean_sell,
        'model_fitted': model_fitted
    }


def run_simulation(train_data: pd.DataFrame,
                  test_data: pd.DataFrame,
                  ohlc_data: pd.DataFrame,
                  k_ar_diff_opt: int,
                  coint_rank_opt: int,
                  egarch_order: tuple,
                  initial_alpha: float,
                  target_index: str) -> dict:
    """
    Run trading simulation with dynamic re-optimization.
    
    Trading Logic:
    - Long Entry: hybrid_yhat_buy > actual_price AND lower_price < lower_bound_mean
    - Long Exit: upper_price > upper_bound_mean
    - Re-optimization: When ECT alpha changes from negative to positive
    
    Position Sizing:
    - Buy: shares = int((capital * fraction_buy) / lower_mean)
    - Sell: shares = int(total_shares * fraction_sell)
    - fraction_buy, fraction_sell: Dynamic based on VECM confidence (0.2~0.8)
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    test_data : pd.DataFrame
        Test data
    ohlc_data : pd.DataFrame
        OHLC data for test period
    k_ar_diff_opt : int
        Optimal VECM lag order
    coint_rank_opt : int
        Optimal cointegration rank
    egarch_order : tuple
        (p, o, q) EGARCH order
    initial_alpha : float
        Initial ECT alpha value
    target_index : str
        Target asset symbol
        
    Returns:
    --------
    dict: Simulation results including trade_history, results, shares_history, etc.
    """
    print(f"\n{'='*60}")
    print("Step 3: Starting Trading Simulation")
    print("="*60)
    print(f"Initial capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Test period: {len(test_data)} days")
    print(f"Commission rate: {COMMISSION_RATE*100:.2f}%")
    print(f"Forecast horizons: Buy={FORECAST_STEPS_BUY} days, Sell={FORECAST_STEPS_SELL} days")
    print(f"Position sizing: Dynamic (confidence-based, {CONFIDENCE_FRACTION_CONFIG['min_fraction']:.1f}~{CONFIDENCE_FRACTION_CONFIG['max_fraction']:.1f})\n")
    
    # Initialize trading state
    capital = INITIAL_CAPITAL
    total_shares = 0
    average_price = 0
    position = None
    cumulative_commission = 0
    
    results = []
    trade_history = []
    shares_history = []
    shares_dates = []
    
    # Initialize history with training data
    history = train_data.copy()
    
    # Initialize confidence-based position sizing
    # Calculate initial confidence from training data
    model_init = VECM(train_data, k_ar_diff=k_ar_diff_opt, coint_rank=coint_rank_opt, deterministic="colo")
    model_init_fitted = model_init.fit()
    
    # Initial confidence for buy (using buy forecast horizon)
    init_output_buy, init_lower_buy, init_upper_buy = model_init_fitted.predict(steps=FORECAST_STEPS_BUY, alpha=0.5)
    init_predicted_mean_buy = init_output_buy.mean(axis=0)[train_data.columns.get_loc(target_index)]
    init_lower_mean_buy = init_lower_buy.mean(axis=0)[train_data.columns.get_loc(target_index)]
    init_upper_mean_buy = init_upper_buy.mean(axis=0)[train_data.columns.get_loc(target_index)]
    initial_confidence_buy = get_vecm_confidence(model_init_fitted, target_index, train_data,
                                                  init_lower_mean_buy, init_upper_mean_buy, init_predicted_mean_buy)
    
    # Initial confidence for sell (using sell forecast horizon)
    init_output_sell, init_lower_sell, init_upper_sell = model_init_fitted.predict(steps=FORECAST_STEPS_SELL, alpha=0.5)
    init_predicted_mean_sell = init_output_sell.mean(axis=0)[train_data.columns.get_loc(target_index)]
    init_lower_mean_sell = init_lower_sell.mean(axis=0)[train_data.columns.get_loc(target_index)]
    init_upper_mean_sell = init_upper_sell.mean(axis=0)[train_data.columns.get_loc(target_index)]
    initial_confidence_sell = get_vecm_confidence(model_init_fitted, target_index, train_data,
                                                  init_lower_mean_sell, init_upper_mean_sell, init_predicted_mean_sell)
    
    # Initialize confidence history
    confidence_history_buy = [initial_confidence_buy]
    confidence_history_sell = [initial_confidence_sell]
    
    # Calculate initial fractions
    fraction_buy = normalize_confidence_to_fraction(initial_confidence_buy, confidence_history_buy, 
                                                    **CONFIDENCE_FRACTION_CONFIG)
    fraction_sell = normalize_confidence_to_fraction(initial_confidence_sell, confidence_history_sell, 
                                                     **CONFIDENCE_FRACTION_CONFIG)
    
    # Log initialization
    print(f"\n[VECM Confidence-Based Dynamic Position Sizing Initialized]")
    print(f"  Initial Confidence (Buy): {initial_confidence_buy:.6f}")
    print(f"  Initial Confidence (Sell): {initial_confidence_sell:.6f}")
    print(f"  Initial Fraction (Buy): {fraction_buy:.4f}")
    print(f"  Initial Fraction (Sell): {fraction_sell:.4f}")
    print(f"  Config: min={CONFIDENCE_FRACTION_CONFIG['min_fraction']}, "
          f"max={CONFIDENCE_FRACTION_CONFIG['max_fraction']}, "
          f"window={CONFIDENCE_FRACTION_CONFIG['window_size']}, "
          f"method={CONFIDENCE_FRACTION_CONFIG['method']}\n")
    
    for t in range(len(test_data)):
        # Update history with new test data
        history = pd.concat([history, test_data.iloc[[t]]])
        
        # Log only every 100 iterations
        should_log = (t % 100 == 0)
        
        # Calculate hybrid forecast
        forecast_results = calculate_hybrid_forecast(
            history, k_ar_diff_opt, coint_rank_opt, egarch_order,
            target_index, FORECAST_STEPS_BUY, FORECAST_STEPS_SELL
        )
        
        hybrid_yhat_buy = forecast_results['hybrid_yhat_buy']
        hybrid_yhat_sell = forecast_results['hybrid_yhat_sell']
        lower_mean = forecast_results['lower_mean_buy']
        upper_mean_buy = forecast_results['upper_mean_buy']
        lower_mean_sell = forecast_results['lower_mean_sell']
        upper_mean = forecast_results['upper_mean_sell']
        predicted_mean_buy = forecast_results['predicted_mean_buy']
        predicted_mean_sell = forecast_results['predicted_mean_sell']
        model_fitted = forecast_results['model_fitted']
        
        # Calculate VECM confidence for buy and sell
        confidence_buy = get_vecm_confidence(model_fitted, target_index, history,
                                            lower_mean, upper_mean_buy, predicted_mean_buy)
        confidence_sell = get_vecm_confidence(model_fitted, target_index, history,
                                             lower_mean_sell, upper_mean, predicted_mean_sell)
        
        # Update confidence history
        confidence_history_buy.append(confidence_buy)
        confidence_history_sell.append(confidence_sell)
        
        # Keep only recent window_size
        if len(confidence_history_buy) > CONFIDENCE_FRACTION_CONFIG['window_size'] * 2:
            confidence_history_buy = confidence_history_buy[-CONFIDENCE_FRACTION_CONFIG['window_size']:]
        if len(confidence_history_sell) > CONFIDENCE_FRACTION_CONFIG['window_size'] * 2:
            confidence_history_sell = confidence_history_sell[-CONFIDENCE_FRACTION_CONFIG['window_size']:]
        
        # Calculate dynamic fractions
        fraction_buy = normalize_confidence_to_fraction(confidence_buy, confidence_history_buy, 
                                                       **CONFIDENCE_FRACTION_CONFIG)
        fraction_sell = normalize_confidence_to_fraction(confidence_sell, confidence_history_sell, 
                                                        **CONFIDENCE_FRACTION_CONFIG)
        
        # Check for re-optimization based on ECT alpha
        current_alpha = get_alpha_value(model_fitted, target_index, history, method=ALPHA_METHOD)
        
        # Re-optimization condition: Alpha changes from negative to positive (original logic)
        should_reoptimize = False
        if initial_alpha < 0 and current_alpha > 0:
            should_reoptimize = True
            if should_log:
                print(f"\n[Re-optimization triggered] Alpha: {initial_alpha:.6f} -> {current_alpha:.6f}")
            # Update initial_alpha with new fitted model (original logic)
            initial_alpha = get_alpha_value(model_fitted, target_index, history, method=ALPHA_METHOD)
            
            # Update confidence and fractions during re-optimization (no trading)
            confidence_history_buy.append(confidence_buy)
            confidence_history_sell.append(confidence_sell)
            if len(confidence_history_buy) > CONFIDENCE_FRACTION_CONFIG['window_size'] * 2:
                confidence_history_buy = confidence_history_buy[-CONFIDENCE_FRACTION_CONFIG['window_size']:]
            if len(confidence_history_sell) > CONFIDENCE_FRACTION_CONFIG['window_size'] * 2:
                confidence_history_sell = confidence_history_sell[-CONFIDENCE_FRACTION_CONFIG['window_size']:]
            fraction_buy = normalize_confidence_to_fraction(confidence_buy, confidence_history_buy, 
                                                           **CONFIDENCE_FRACTION_CONFIG)
            fraction_sell = normalize_confidence_to_fraction(confidence_sell, confidence_history_sell, 
                                                            **CONFIDENCE_FRACTION_CONFIG)
            
            if should_log:
                print(f"  Updated initial_alpha: {initial_alpha:.6f}")
                print(f"  Confidence (Buy): {confidence_buy:.6f}, Confidence (Sell): {confidence_sell:.6f}")
                print(f"  Fraction (Buy): {fraction_buy:.4f}, Fraction (Sell): {fraction_sell:.4f}")
                print(f"  Warning: Skipping trading at this step due to re-optimization\n")
            
            # Skip trading during re-optimization
            date_str = test_data.index[t].strftime('%Y-%m-%d')
            actual_price = test_data[target_index].iloc[t]
            
            if position == 'long':
                total_assets = capital + total_shares * actual_price
            else:
                total_assets = capital
            
            results.append(total_assets)
            shares_history.append(total_shares)
            shares_dates.append(test_data.index[t])
            
            trade_history.append({
                'date': date_str,
                'confidence_buy': confidence_buy,
                'confidence_sell': confidence_sell,
                'fraction_buy': fraction_buy,
                'fraction_sell': fraction_sell,
                'hybrid_yhat_buy': None,
                'hybrid_yhat_sell': None,
                'actual_price': actual_price,
                'capital': capital,
                'total_shares': total_shares,
                'total_assets': total_assets,
                'position': position,
                'unrealized_pnl': (actual_price - average_price) * total_shares if position == 'long' else 0,
                'note': 'Re-optimization'
            })
            continue
        
        # Get current prices
        actual_price = test_data[target_index].iloc[t]
        lower_price = ohlc_data['Low'][target_index].iloc[t]
        upper_price = ohlc_data['High'][target_index].iloc[t]
        close_price = ohlc_data['Close'][target_index].iloc[t]
        
        # Long entry: hybrid_yhat_buy > actual_price AND lower_price < lower_bound_mean
        if hybrid_yhat_buy > actual_price and capital > 0 and lower_price < lower_mean:
            position = 'long'
            
            # Calculate shares to buy (integer shares) - use dynamic fraction_buy
            shares_to_buy_float = (capital * fraction_buy) / lower_mean
            shares_to_buy = int(shares_to_buy_float)
            
            if shares_to_buy >= 1 and shares_to_buy * lower_mean <= capital:
                # Update average price: weighted average
                total_value = (average_price * total_shares) + (lower_mean * shares_to_buy)
                total_shares += shares_to_buy
                average_price = total_value / total_shares
                
                # Apply commission
                commission = shares_to_buy * lower_mean * COMMISSION_RATE
                cumulative_commission += commission
                capital -= shares_to_buy * lower_mean + commission
                
                if should_log:
                    print(f"Buying {shares_to_buy} shares at {lower_mean:.2f} "
                          f"(wanted: {shares_to_buy_float:.2f}, lower_price: {lower_price:.2f})")
        
        # Long exit: upper_price > upper_bound_mean
        if position == 'long' and upper_price > upper_mean:
            shares_to_sell_float = total_shares * fraction_sell
            shares_to_sell = int(shares_to_sell_float)
            
            if shares_to_sell >= 1:
                sell_value = shares_to_sell * upper_mean
                commission = shares_to_sell * upper_mean * COMMISSION_RATE
                cumulative_commission += commission
                total_shares -= shares_to_sell
                capital += sell_value - commission
                
                if should_log:
                    print(f"Selling {shares_to_sell} shares at {upper_mean:.2f} "
                          f"(wanted: {shares_to_sell_float:.2f})")
                
                if total_shares <= 0:
                    position = None
                    total_shares = 0
                    average_price = 0
        
        # Calculate total assets
        if position == 'long':
            total_assets = capital + total_shares * actual_price
            unrealized_pnl = (actual_price - average_price) * total_shares
        else:
            total_assets = capital
            unrealized_pnl = 0
        
        results.append(total_assets)
        shares_history.append(total_shares)
        shares_dates.append(test_data.index[t])
        
        # Log metrics
        if should_log:
            date_str = test_data.index[t].strftime('%Y-%m-%d')
            print(f"date: {date_str:>10} | Conf(Buy): {confidence_buy:>6.4f} | Conf(Sell): {confidence_sell:>6.4f} | "
                  f"Frac(Buy): {fraction_buy:>5.3f} | Frac(Sell): {fraction_sell:>5.3f} | "
                  f"hybrid_yhat_buy: {hybrid_yhat_buy:>10.2f} | hybrid_yhat_sell: {hybrid_yhat_sell:>10.2f} | "
                  f"actual_price: {actual_price:>10.2f} | capital: {capital:>10.2f} | total_shares: {total_shares:>10.2f} | "
                  f"total_assets: {total_assets:>10.2f} | position: {str(position):>6} | "
                  f"unrealized_pnl: {unrealized_pnl:>10.2f} | cumulative_commission: {cumulative_commission:>10.2f}")
        
        # Record trade history
        date_str = test_data.index[t].strftime('%Y-%m-%d')
        trade_history.append({
            'date': date_str,
            'confidence_buy': confidence_buy,
            'confidence_sell': confidence_sell,
            'fraction_buy': fraction_buy,
            'fraction_sell': fraction_sell,
            'hybrid_yhat_buy': hybrid_yhat_buy,
            'hybrid_yhat_sell': hybrid_yhat_sell,
            'actual_price': close_price,
            'capital': capital,
            'total_shares': total_shares,
            'total_assets': total_assets,
            'position': position,
            'unrealized_pnl': unrealized_pnl
        })
    
    return {
        'trade_history': trade_history,
        'results': results,
        'shares_history': shares_history,
        'shares_dates': shares_dates,
        'final_capital': capital,
        'total_shares': total_shares,
        'cumulative_commission': cumulative_commission
    }


def print_performance_report(strategy_metrics: dict, bnh_metrics: dict, 
                            simulation_start_date: pd.Timestamp,
                            simulation_end_date: pd.Timestamp,
                            target_index: str):
    """
    Print comprehensive performance report.
    
    Parameters:
    -----------
    strategy_metrics : dict
        Strategy performance metrics
    bnh_metrics : dict
        Buy-and-hold performance metrics
    simulation_start_date : pd.Timestamp
        Simulation start date
    simulation_end_date : pd.Timestamp
        Simulation end date
    target_index : str
        Target asset symbol
    """
    print("")
    print(f"Simulation Performance: {simulation_start_date.date()} ~ {simulation_end_date.date()}")
    print(f"=" * 80)
    
    print(f"STRATEGY PERFORMANCE:")
    print(f"  Final Capital: ${strategy_metrics['final_capital']:,.2f} USD")
    print(f"  Cumulative Return: {strategy_metrics['total_return']:.2%}")
    print(f"  Annualized Return: {strategy_metrics['annualized_return_time']:.2%}")
    print(f"  Return Standard Deviation: {strategy_metrics['annualized_std_return']:.4f}")
    print(f"  Sharpe Ratio: {strategy_metrics['sharpe_ratio']:.4f}")
    print(f"  Peak: ${strategy_metrics['peak']:,.2f}")
    print(f"  Maximum Drawdown: ${strategy_metrics['max_drawdown']:,.2f}")
    print(f"  Maximum Consecutive Loss Days: {strategy_metrics['max_consecutive_loss_days']}")
    
    print(f"\nBUY & HOLD ({target_index}) PERFORMANCE:")
    print(f"  Final Price: ${bnh_metrics['final_price']:,.2f} USD (Initial: ${bnh_metrics['initial_price']:,.2f} USD)")
    print(f"  Cumulative Return: {bnh_metrics['total_return']:.2%}")
    print(f"  Annualized Return: {bnh_metrics['annualized_return']:.2%}")
    print(f"  Return Standard Deviation: {bnh_metrics['annualized_std']:.4f}")
    print(f"  Sharpe Ratio: {bnh_metrics['sharpe_ratio']:.4f}")
    print(f"  Peak: ${bnh_metrics['peak']:,.2f}")
    print(f"  Maximum Drawdown: ${bnh_metrics['max_drawdown']:,.2f}")
    
    # Comparison
    print(f"\nCOMPARISON:")
    print(f"  Outperformance (Return): {(strategy_metrics['total_return'] - bnh_metrics['total_return']):+.2%}")
    print(f"  Outperformance (Annualized): {(strategy_metrics['annualized_return_time'] - bnh_metrics['annualized_return']):+.2%}")
    print(f"  Sharpe Ratio Difference: {strategy_metrics['sharpe_ratio'] - bnh_metrics['sharpe_ratio']:+.4f}")
    print(f"=" * 80)


def main():
    """Main function to run production simulation"""
    # Set dates
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365*YEARS_BACK)).strftime('%Y-%m-%d')
    
    # Load and preprocess data
    print(f"{'='*60}")
    print("Loading and Preprocessing Data")
    print("="*60)
    
    df = load_data(TICKERS, start_date, end_date, INTERVAL)
    print(f"Loaded {len(df)} observations")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Split data
    train_data, test_data, ohlc_data = split_data(df, TARGET_INDEX, TRAIN_RATIO)
    print(f"\nTraining data: {len(train_data)} observations")
    print(f"Test data: {len(test_data)} observations")
    
    # Initialize models
    model_fitted, k_ar_diff_opt, coint_rank_opt, egarch_order, initial_alpha = \
        initialize_models(train_data, TARGET_INDEX)
    
    # Run simulation
    simulation_results = run_simulation(
        train_data, test_data, ohlc_data,
        k_ar_diff_opt, coint_rank_opt, egarch_order,
        initial_alpha, TARGET_INDEX
    )
    
    # Calculate performance metrics
    simulation_start_date = test_data.index[0]
    simulation_end_date = test_data.index[-1]
    
    strategy_metrics = calculate_performance_metrics(
        simulation_results['results'],
        INITIAL_CAPITAL,
        simulation_start_date,
        simulation_end_date
    )
    
    bnh_metrics = calculate_buy_hold_metrics(
        test_data, TARGET_INDEX, INITIAL_CAPITAL, COMMISSION_RATE,
        simulation_start_date, simulation_end_date
    )
    
    # Print performance report
    print_performance_report(strategy_metrics, bnh_metrics,
                            simulation_start_date, simulation_end_date, TARGET_INDEX)
    
    # Save trade history to Excel
    os.makedirs(RESULTS_DIR, exist_ok=True)
    trade_history_df = pd.DataFrame(simulation_results['trade_history'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trade_history_path = os.path.join(RESULTS_DIR, f'trade_history_{timestamp}.xlsx')
    trade_history_df.to_excel(trade_history_path, index=False)
    print(f"\nTrade history saved to: {trade_history_path}")
    
    # Create and save charts
    chart_path = create_trading_charts(
        test_data, TARGET_INDEX,
        simulation_results['shares_history'],
        simulation_results['shares_dates'],
        simulation_results['results'],
        INITIAL_CAPITAL,
        COMMISSION_RATE,
        RESULTS_DIR,
        timestamp
    )
    print(f"Chart saved to: {chart_path}")
    plt.show()
    
    print("\nSimulation complete")


if __name__ == "__main__":
    main()

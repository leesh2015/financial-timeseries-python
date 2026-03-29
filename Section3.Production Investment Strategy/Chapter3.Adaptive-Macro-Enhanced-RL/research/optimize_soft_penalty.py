import numpy as np
import pandas as pd
import optuna
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_trading_simulation(signals_df, penalty_config, regime_config=None):
    """
    High-speed trading simulation using precomputed signals.
    Allows for 100+ trials per minute.
    """
    if regime_config is None:
        # Default regime-specific fraction ranges
        regime_config = {
            'bull': {'min_fraction': 0.68, 'max_fraction': 0.97},
            'bear': {'min_fraction': 0.48, 'max_fraction': 0.77},
            'sideways': {'min_fraction': 0.43, 'max_fraction': 0.86},
            'high_vol': {'min_fraction': 0.47, 'max_fraction': 0.70}
        }
        
    initial_capital = 10000
    capital = initial_capital
    total_shares = 0
    average_price = 0
    position = None
    commission_rate = 0.0002
    
    results = []
    confidence_history_buy = []
    confidence_history_sell = []
    
    # Strategy parameters
    buy_confidence_threshold = 0.65
    sell_confidence_threshold = 0.55
    max_position_ratio = 0.90
    min_position_ratio_for_sell = 0.20
    max_position_size = 0.95
    RL_BLEND_FACTOR = 0.6
    
    for i, row in signals_df.iterrows():
        actual_price = row['actual_price']
        lower_price = row['lower_price']
        upper_price = row['upper_price']
        close_price = row['close_price']
        predicted_mean_buy = row['predicted_mean_buy']
        predicted_mean_sell = row['predicted_mean_sell']
        confidence_buy = row['confidence_buy']
        confidence_sell = row['confidence_sell']
        lower_mean = row['lower_mean']
        upper_mean = row['upper_mean']
        garch_mean_average = row['garch_mean_average']
        current_regime = row['regime']
        cl_momentum = row['CL=F_momentum']
        zb_momentum = row['ZB=F_momentum']
        ect_alpha = row.get('ECT_Alpha', 0.0)
        
        confidence_history_buy.append(confidence_buy)
        confidence_history_sell.append(confidence_sell)
        
        # Calculate dynamic thresholds
        abs_threshold_buy = np.percentile(confidence_history_buy[-60:], 10) if len(confidence_history_buy) >= 2 else 0.35
        abs_threshold_sell = np.percentile(confidence_history_sell[-60:], 10) if len(confidence_history_sell) >= 2 else 0.35
        
        def get_fraction(conf, hist, reg_min, reg_max, abs_thresh):
            if conf < abs_thresh: return reg_min
            if len(hist) < 2: return (reg_min + reg_max) / 2
            window = hist[-60:]
            c_min, c_max = min(window), max(window)
            if c_max == c_min: return (reg_min + reg_max) / 2
            norm = (conf - c_min) / (c_max - c_min)
            return reg_min + (reg_max - reg_min) * norm

        reg_cfg = regime_config.get(current_regime, {'min_fraction': 0.2, 'max_fraction': 0.8})
        base_fraction_buy = get_fraction(confidence_buy, confidence_history_buy, reg_cfg['min_fraction'], reg_cfg['max_fraction'], abs_threshold_buy)
        base_fraction_sell = get_fraction(confidence_sell, confidence_history_sell, reg_cfg['min_fraction'], reg_cfg['max_fraction'], abs_threshold_sell)
        
        fraction_buy = base_fraction_buy
        fraction_sell = base_fraction_sell
        
        if i >= 10:
            pos_ratio = (total_shares * actual_price) / (capital + total_shares * actual_price) if (capital + total_shares * actual_price) > 0 else 0
            
            # --- Soft Penalty Logic ---
            cl_penalty = max(0.0, float(cl_momentum)) * penalty_config['cl_multiplier']
            zb_penalty = max(0.0, -float(zb_momentum)) * penalty_config['zb_multiplier']
            macro_penalty_ratio = cl_penalty + zb_penalty
            
            # Structural shock check (Stagflationary proxy)
            if cl_momentum > penalty_config['stag_cl_threshold'] and zb_momentum < penalty_config['stag_zb_threshold']:
                macro_penalty_ratio = max(macro_penalty_ratio, penalty_config['stag_penalty_floor'])
                
            ect_bonus = 0.0
            if ect_alpha > 0:
                macro_penalty_ratio += float(ect_alpha) * penalty_config['ect_penalty_multiplier']
            elif ect_alpha < 0:
                ect_bonus = -float(ect_alpha) * penalty_config['ect_bonus_multiplier']
                
            # Limit total penalty to avoid total capital lock in minor volatility
            macro_penalty_ratio = min(penalty_config['max_penalty_cap'], macro_penalty_ratio)
            
            # Apply to confidence
            penalized_confidence = min(1.0, max(0.0, confidence_buy * (1.0 - macro_penalty_ratio) + ect_bonus))
            
            # Action Signal
            if confidence_buy > buy_confidence_threshold and pos_ratio < max_position_ratio:
                position_size = min(max_position_size, penalized_confidence)
                rl_signal = 1.0
            elif confidence_buy < sell_confidence_threshold and pos_ratio > min_position_ratio_for_sell:
                position_size = pos_ratio
                rl_signal = -1.0
            else:
                position_size = max(0.0, pos_ratio * (1.0 - macro_penalty_ratio))
                rl_signal = 0.0
                
            # Blend RL into base VECM strategy
            if rl_signal > 0.5:
                rl_frac_buy = np.clip(position_size, reg_cfg['min_fraction'], reg_cfg['max_fraction'])
                fraction_buy = RL_BLEND_FACTOR * rl_frac_buy + (1 - RL_BLEND_FACTOR) * base_fraction_buy
            elif rl_signal < -0.5:
                rl_frac_sell = np.clip(position_size, reg_cfg['min_fraction'], reg_cfg['max_fraction'])
                fraction_sell = RL_BLEND_FACTOR * rl_frac_sell + (1 - RL_BLEND_FACTOR) * base_fraction_sell
                
        hybrid_yhat_buy = predicted_mean_buy + garch_mean_average
        
        # Execution
        if hybrid_yhat_buy > actual_price and capital > 0 and lower_price < lower_mean:
            position = 'long'
            shares = int((capital * fraction_buy) / lower_mean)
            if shares >= 1 and shares * lower_mean <= capital:
                total_shares += shares
                average_price = (average_price * (total_shares - shares) + lower_mean * shares) / total_shares
                capital -= (shares * lower_mean * (1 + commission_rate))
        
        if position == 'long' and total_shares >= 1:
            is_bull = (current_regime == 'bull')
            should_sell = (upper_price > upper_mean)
            if should_sell:
                ratio = min(fraction_sell, 0.2) if is_bull else fraction_sell
                shares = int(total_shares * ratio)
                if shares >= 1:
                    capital += (shares * upper_mean * (1 - commission_rate))
                    total_shares -= shares
                    if total_shares <= 0: position, average_price = None, 0

        val = capital + total_shares * close_price
        results.append(val)
            
    return results

def objective(trial):
    """
    Optuna objective function for maximizing the investment quality score.
    """
    penalty_config = {
        'cl_multiplier': trial.suggest_float('cl_multiplier', 1.0, 10.0),
        'zb_multiplier': trial.suggest_float('zb_multiplier', 1.0, 10.0),
        'ect_penalty_multiplier': trial.suggest_float('ect_penalty_multiplier', 1.0, 10.0),
        'ect_bonus_multiplier': trial.suggest_float('ect_bonus_multiplier', 0.1, 5.0),
        'max_penalty_cap': trial.suggest_float('max_penalty_cap', 0.20, 0.95),
        'stag_cl_threshold': trial.suggest_float('stag_cl_threshold', 0.02, 0.10),
        'stag_zb_threshold': trial.suggest_float('stag_zb_threshold', -0.06, -0.01),
        'stag_penalty_floor': trial.suggest_float('stag_penalty_floor', 0.30, 0.95)
    }
    
    try:
        signals_df = pd.read_csv('precomputed_signals.csv')
        results = run_trading_simulation(signals_df, penalty_config)
        
        if not results: return -1e9
        
        returns = np.diff(results) / results[:-1]
        final_return = (results[-1] - 10000) / 10000
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Calculate MDD
        peak = results[0]
        max_dd = 0
        for r in results:
            if r > peak: peak = r
            dd = (peak - r) / peak
            if dd > max_dd: max_dd = dd
            
        safe_dd = max_dd if max_dd > 0 else 0.01
        
        # Composite score prioritizing Risk-Adjusted Return and DD suppression
        score = (sharpe * max(0.1, final_return)) / (safe_dd ** 2)
        return score
    except:
        return -1e9

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    print("Starting Bayesian Optimization using Optuna (100 trials)...")
    print("Goal: Maximize Risk-Adjusted Composite Score [Sharpe * Return / (MDD^2)]")
    
    study = optuna.create_study(direction='maximize')
    
    # Check if precomputed signals exist
    if not os.path.exists('precomputed_signals.csv'):
        print("Error: precomputed_signals.csv not found in the current directory.")
        print("Please run precompute_signals.py first.")
        sys.exit(1)
        
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    
    print("\n" + "="*50)
    print("BEST SOFT PENALTY CONFIGURATION FOUND")
    print("="*50)
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.4f}")
    
    # Execution with best params
    best_config = study.best_params
    signals_df = pd.read_csv('precomputed_signals.csv')
    best_results = run_trading_simulation(signals_df, best_config)
    
    best_ret = (best_results[-1] - 10000) / 10000
    returns = np.diff(best_results) / best_results[:-1]
    best_sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    peak = best_results[0]
    best_max_dd = 0
    for r in best_results:
        if r > peak: peak = r
        dd = (peak - r) / peak
        if dd > best_max_dd: best_max_dd = dd
        
    print(f"\nSIMULATION OUTCOME WITH BEST PARAMS:")
    print(f"  Final Return:  {best_ret*100:.2f}%")
    print(f"  Max Drawdown:  {best_max_dd*100:.2f}%")
    print(f"  Sharpe Ratio:  {best_sharpe:.4f}")
    print("="*50)

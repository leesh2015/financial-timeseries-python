import numpy as np
import pandas as pd
import optuna
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from regime_detector import ModelBasedRegimeDetector

def run_trading_simulation(signals_df, penalty_config, regime_params, rl_params):
    """
    High-speed trading simulation using precomputed signals and dynamic regime detection.
    Includes adaptive volatility-based sell logic for Bull regimes.
    """
    initial_capital = 10000
    capital = initial_capital
    total_shares = 0
    average_price = 0
    position = None
    commission_rate = 0.0002
    
    # Initialize Regime Detector
    detector = ModelBasedRegimeDetector(
        window=int(regime_params['window']),
        hysteresis_threshold=regime_params['hysteresis_threshold']
    )
    
    results = []
    confidence_history_buy = []
    confidence_history_sell = []
    
    # RL/Strategy parameters
    buy_confidence_threshold = rl_params['buy_confidence_threshold']
    sell_confidence_threshold = rl_params['sell_confidence_threshold']
    RL_BLEND_FACTOR = rl_params['rl_blend_factor']
    bull_sell_vol_mult = rl_params['bull_sell_vol_mult'] # New: Volatility based multiplier
    bull_sell_fraction = rl_params['bull_sell_fraction'] # New: Optimized sell ratio
    
    for i, row in signals_df.iterrows():
        actual_price = row['actual_price']
        high_price = row['high_price']
        low_price = row['low_price']
        close_price = row['close_price']
        predicted_mean_buy = row['predicted_mean_buy']
        predicted_mean_sell = row['predicted_mean_sell']
        confidence_buy = row['confidence_buy']
        confidence_sell = row['confidence_sell']
        lower_mean = row['lower_mean']
        upper_mean = row['upper_mean']
        garch_mean_average = row['garch_mean_average']
        garch_vol = row['garch_volatility']
        cl_momentum = row['CL=F_momentum']
        zb_momentum = row['ZB=F_momentum']
        ect_alpha = row['ECT_Alpha']
        
        # Dynamic Regime Detection
        current_regime = detector.detect_regime(
            vecm_pred=predicted_mean_buy,
            garch_vol=garch_vol
        )
        
        confidence_history_buy.append(confidence_buy)
        confidence_history_sell.append(confidence_sell)
        
        abs_threshold_buy = np.percentile(confidence_history_buy[-60:], 10) if len(confidence_history_buy) >= 2 else 0.35
        
        def get_fraction(conf, hist, reg_min, reg_max, abs_thresh):
            if conf < abs_thresh: return reg_min
            if len(hist) < 2: return (reg_min + reg_max) / 2
            window = hist[-60:]
            c_min, c_max = min(window), max(window)
            if c_max == c_min: return (reg_min + reg_max) / 2
            norm = (conf - c_min) / (c_max - c_min)
            return reg_min + (reg_max - reg_min) * norm

        regime_config = {
            'bull': {'min_fraction': 0.60, 'max_fraction': 0.99},
            'bear': {'min_fraction': 0.20, 'max_fraction': 0.60},
            'sideways': {'min_fraction': 0.20, 'max_fraction': 0.80},
            'high_vol': {'min_fraction': 0.25, 'max_fraction': 0.65}
        }
        
        reg_cfg = regime_config.get(current_regime, {'min_fraction': 0.2, 'max_fraction': 0.8})
        base_fraction_buy = get_fraction(confidence_buy, confidence_history_buy, reg_cfg['min_fraction'], reg_cfg['max_fraction'], abs_threshold_buy)
        
        fraction_buy = base_fraction_buy
        
        if i >= 10:
            pos_ratio = (total_shares * actual_price) / (capital + total_shares * actual_price) if (capital + total_shares * actual_price) > 0 else 0
            cl_penalty = max(0.0, float(cl_momentum)) * penalty_config['cl_multiplier']
            zb_penalty = max(0.0, -float(zb_momentum)) * penalty_config['zb_multiplier']
            macro_penalty_ratio = cl_penalty + zb_penalty
            
            if cl_momentum > penalty_config['stag_cl_threshold'] and zb_momentum < penalty_config['stag_zb_threshold']:
                macro_penalty_ratio = max(macro_penalty_ratio, penalty_config['stag_penalty_floor'])
                
            ect_bonus = 0.0
            if ect_alpha > 0:
                macro_penalty_ratio += float(ect_alpha) * penalty_config['ect_penalty_multiplier']
            elif ect_alpha < 0:
                ect_bonus = -float(ect_alpha) * penalty_config['ect_bonus_multiplier']
                
            macro_penalty_ratio = min(penalty_config['max_penalty_cap'], macro_penalty_ratio)
            penalized_confidence = min(1.0, max(0.0, confidence_buy * (1.0 - macro_penalty_ratio) + ect_bonus))
            
            if confidence_buy > buy_confidence_threshold and pos_ratio < 0.95:
                position_size = min(0.95, penalized_confidence)
                rl_fraction_buy = np.clip(position_size, reg_cfg['min_fraction'], reg_cfg['max_fraction'])
                fraction_buy = RL_BLEND_FACTOR * rl_fraction_buy + (1 - RL_BLEND_FACTOR) * base_fraction_buy
                
        hybrid_yhat_buy = predicted_mean_buy + garch_mean_average
        
        # Execution
        if hybrid_yhat_buy > actual_price and capital > 0 and low_price < lower_mean:
            position = 'long'
            shares = int((capital * fraction_buy) / lower_mean)
            if shares >= 1 and shares * lower_mean <= capital:
                total_shares += shares
                average_price = (average_price * (total_shares - shares) + lower_mean * shares) / total_shares
                capital -= (shares * lower_mean * (1 + commission_rate))
        
        if position == 'long' and total_shares >= 1:
            is_bull = (current_regime == 'bull')
            should_sell = (high_price > upper_mean)
            
            # ADAPTIVE BULL SELL FILTER
            if is_bull:
                # Equilibrium point: Is the predicted reversal > sensitivity * current volatility?
                expected_sell_ret = (predicted_mean_sell + garch_mean_average - actual_price) / actual_price
                adaptive_reversal_threshold = -(garch_vol * bull_sell_vol_mult)
                if expected_sell_ret >= adaptive_reversal_threshold:
                    should_sell = False
            
            if should_sell:
                ratio = bull_sell_fraction if is_bull else 0.5 # Simplified sell logic
                shares = int(total_shares * ratio)
                if shares >= 1:
                    capital += (shares * upper_mean * (1 - commission_rate))
                    total_shares -= shares
                    if total_shares <= 0: position, average_price = None, 0

        val = capital + total_shares * close_price
        results.append(val)
            
    return results

def objective(trial):
    penalty_config = {
        'cl_multiplier': trial.suggest_float('cl_multiplier', 5.0, 20.0),
        'zb_multiplier': trial.suggest_float('zb_multiplier', 1.0, 15.0),
        'ect_penalty_multiplier': trial.suggest_float('ect_penalty_multiplier', 1.0, 15.0),
        'ect_bonus_multiplier': trial.suggest_float('ect_bonus_multiplier', 1.0, 10.0),
        'max_penalty_cap': trial.suggest_float('max_penalty_cap', 0.50, 0.95),
        'stag_cl_threshold': trial.suggest_float('stag_cl_threshold', 0.05, 0.15),
        'stag_zb_threshold': trial.suggest_float('stag_zb_threshold', -0.10, -0.02),
        'stag_penalty_floor': trial.suggest_float('stag_penalty_floor', 0.30, 0.95)
    }
    
    regime_params = {
        'window': trial.suggest_int('window', 50, 100),
        'hysteresis_threshold': trial.suggest_float('hysteresis_threshold', 0.03, 0.12)
    }
    
    rl_params = {
        'buy_confidence_threshold': trial.suggest_float('buy_confidence_threshold', 0.45, 0.65),
        'sell_confidence_threshold': trial.suggest_float('sell_confidence_threshold', 0.45, 0.65),
        'rl_blend_factor': trial.suggest_float('rl_blend_factor', 0.6, 0.95),
        'bull_sell_vol_mult': trial.suggest_float('bull_sell_vol_mult', 0.5, 3.0),
        'bull_sell_fraction': trial.suggest_float('bull_sell_fraction', 0.1, 0.5)
    }
    
    try:
        signals_df = pd.read_csv('precomputed_signals.csv')
        results = run_trading_simulation(signals_df, penalty_config, regime_params, rl_params)
        
        if not results: return -1e9
        
        returns = np.diff(results) / results[:-1]
        final_return = (results[-1] - 10000) / 10000
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # MDD calculation
        peak = results[0]
        max_dd = 0
        for r in results:
            if r > peak: peak = r
            dd = (peak - r) / peak
            if dd > max_dd: max_dd = dd
        safe_dd = max_dd if max_dd > 0 else 0.01
        
        # Objective: (Sharpe * Return^1.5) / MDD^1.2
        score = (sharpe * (max(0.1, final_return) ** 1.5)) / (safe_dd ** 1.2)
        return score
    except:
        return -1e9

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("Starting Equilibrium Optimization study (150 trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=150, show_progress_bar=True)
    
    print("\n" + "="*50)
    print("NEW GOLDEN RATIO (EQUILIBRIUM FOUND)")
    print("="*50)
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.4f}")
    
    # Final Result Simulation
    best_results = run_trading_simulation(pd.read_csv('precomputed_signals.csv'), 
                                         study.best_params, study.best_params, study.best_params)
    print(f"\nFinal Expected Capital: ${best_results[-1]:.2f}")
    print("="*50)

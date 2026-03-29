import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions_ import get_vecm_confidence, get_alpha_value
from statsmodels.tsa.vector_ar.vecm import VECM

def precompute_signals(data_path, target_index='TQQQ'):
    """
    Precomputes VECM predictions, confidence levels, and macro indicators
    to allow for hyper-fast Bayesian optimization of RL parameters.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Required columns check
    req_cols = [target_index, 'CL=F', 'ZB=F']
    if not all(col in df.columns for col in req_cols):
        raise ValueError(f"Data must contain {req_cols}")

    results_list = []
    
    # We need a rolling window to simulate the VECM fitting process accurately
    # Start after sufficient data for initial fit (e.g., 252 days)
    start_idx = 252 
    window_size = 252
    
    print(f"Starting precomputation for {len(df) - start_idx} steps...")
    
    for t in range(start_idx, len(df)):
        history = df.iloc[t-window_size:t]
        current_row = df.iloc[t]
        
        # Fit VECM (Simplified for precompute speed, fixed parameters)
        try:
            model = VECM(history, k_ar_diff=1, coint_rank=1, deterministic="colo")
            model_fitted = model.fit()
            
            # Forecasts
            forecast_buy, low_buy, high_buy = model_fitted.predict(steps=5, alpha=0.5)
            pred_buy = forecast_buy.mean(axis=0)[history.columns.get_loc(target_index)]
            low_mean_buy = low_buy.mean(axis=0)[history.columns.get_loc(target_index)]
            high_mean_buy = high_buy.mean(axis=0)[history.columns.get_loc(target_index)]
            
            forecast_sell, low_sell, high_sell = model_fitted.predict(steps=2, alpha=0.5)
            pred_sell = forecast_sell.mean(axis=0)[history.columns.get_loc(target_index)]
            low_mean_sell = low_sell.mean(axis=0)[history.columns.get_loc(target_index)]
            high_mean_sell = high_sell.mean(axis=0)[history.columns.get_loc(target_index)]
            
            # Confidence
            conf_buy = get_vecm_confidence(model_fitted, target_index, history, low_mean_buy, high_mean_buy, pred_buy)
            conf_sell = get_vecm_confidence(model_fitted, target_index, history, low_mean_sell, high_mean_sell, pred_sell)
            
            # ECT Alpha
            ect_alpha = get_alpha_value(model_fitted, target_index, history)
            
            # Macro Momentum (1-month)
            cl_p = history['CL=F'].iloc[-21]
            cl_mom = (history['CL=F'].iloc[-1] - cl_p) / cl_p
            
            zb_p = history['ZB=F'].iloc[-21]
            zb_mom = (history['ZB=F'].iloc[-1] - zb_p) / zb_p
            
            # Regime (Simplified for precompute)
            # In a real scenario, this would use the ModelBasedRegimeDetector
            # Here we just record daily volatility for the optimizer to decide
            vol = history[target_index].pct_change().std() * np.sqrt(252)
            
            results_list.append({
                'date': df.index[t],
                'actual_price': current_row[target_index],
                'lower_price': current_row[target_index] * 0.99, # Proxy
                'upper_price': current_row[target_index] * 1.01, # Proxy
                'close_price': current_row[target_index],
                'predicted_mean_buy': pred_buy,
                'predicted_mean_sell': pred_sell,
                'confidence_buy': conf_buy,
                'confidence_sell': conf_sell,
                'lower_mean': low_mean_buy,
                'upper_mean': high_mean_sell,
                'garch_mean_average': 0.0, # Proxy if not running GARCH
                'regime': 'sideways', # Proxy
                'CL=F_momentum': cl_mom,
                'ZB=F_momentum': zb_mom,
                'ECT_Alpha': ect_alpha,
                'annualized_vol': vol
            })
            
            if t % 100 == 0:
                print(f"  Processed {t}/{len(df)}...")
                
        except Exception as e:
            print(f"  Skipping t={t} due to error: {e}")
            continue
            
    output_df = pd.DataFrame(results_list)
    output_path = os.path.join(os.path.dirname(__file__), 'precomputed_signals.csv')
    output_df.to_csv(output_path, index=False)
    print(f"Precomputed signals saved to: {output_path}")

if __name__ == "__main__":
    # Placeholder for usage
    # precompute_signals('../../data/TQQQ_data.csv')
    print("This script is a research utility to generate signals for the Bayesian optimizer.")
    print("Usage: import precompute_signals and run with your price data path.")

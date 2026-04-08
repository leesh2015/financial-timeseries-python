import pandas as pd
import numpy as np
import os
import sys
import yfinance as yf
from datetime import datetime, timedelta
from arch import arch_model
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions_ import get_vecm_confidence, get_alpha_value

def precompute_signals(tickers=None, target_index='TQQQ', years=10):
    """
    Precomputes full high-fidelity signals for Bayesian optimization.
    Matches the logic of dynamic_simulation_rl.py exactly.
    """
    output_path = os.path.join(os.path.dirname(__file__), 'precomputed_signals.csv')
    
    if tickers is None:
        tickers = ['TQQQ', 'NQ=F', 'ZB=F', 'UNG', 'RB=F', 'BZ=F', 'CL=F']

    print(f"Downloading data for {tickers}...")
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365*years)).strftime('%Y-%m-%d')
    
    df = yf.download(tickers, start=start_date, end=end_date, interval='1d', auto_adjust=True, progress=False)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.asfreq('D').dropna()
    
    split_index = int(len(df) * 0.7)
    train_data = df['Open'].iloc[:split_index]
    test_data = df['Open'].iloc[split_index:]
    ohlc_test = df.iloc[split_index:]
    
    # Check for existing progress
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        last_date = pd.to_datetime(existing_df['date'].iloc[-1])
        start_t = len(existing_df)
        print(f"Found existing progress. Resuming from {last_date} (Index: {start_t})")
        results_list = existing_df.to_dict('records')
    else:
        print("Starting fresh precomputation...")
        start_t = 0
        results_list = []

    print("Estimating optimal VECM parameters from training data...")
    lag_order = select_order(train_data, maxlags=15, deterministic="cili")
    k_ar_diff_opt = lag_order.aic
    coint_rank_test = select_coint_rank(train_data, det_order=1, k_ar_diff=k_ar_diff_opt, method='trace')
    coint_rank_opt = coint_rank_test.rank
    
    # GARCH parameters (baseline)
    p_opt, o_opt, q_opt = 1, 0, 1
    
    print(f"Starting/Resuming high-fidelity precomputation for remaining {len(test_data) - start_t} days...")
    
    for t in range(start_t, len(test_data)):
        current_date = test_data.index[t]
        history = pd.concat([train_data, test_data.iloc[:t+1]]) # Efficient history construction
        
        try:
            # 1. Fit VECM
            model = VECM(history, k_ar_diff=k_ar_diff_opt, coint_rank=coint_rank_opt, deterministic="cili")
            model_fitted = model.fit()
            
            # 2. VECM Forecasts
            output_buy, lower_bound_buy, upper_bound_buy = model_fitted.predict(steps=4, alpha=0.5)
            lower_mean = lower_bound_buy.mean(axis=0)[history.columns.get_loc(target_index)]
            predicted_mean_buy = output_buy.mean(axis=0)[history.columns.get_loc(target_index)]
            upper_mean_buy = upper_bound_buy.mean(axis=0)[history.columns.get_loc(target_index)]
            
            output_sell, lower_bound_sell, upper_bound_sell = model_fitted.predict(steps=7, alpha=0.5)
            upper_mean = upper_bound_sell.mean(axis=0)[history.columns.get_loc(target_index)]
            predicted_mean_sell = output_sell.mean(axis=0)[history.columns.get_loc(target_index)]
            lower_mean_sell = lower_bound_sell.mean(axis=0)[history.columns.get_loc(target_index)]
            
            # 3. Confidence & ECT Alpha
            conf_buy = get_vecm_confidence(model_fitted, target_index, history, lower_mean, upper_mean_buy, predicted_mean_buy)
            conf_sell = get_vecm_confidence(model_fitted, target_index, history, lower_mean_sell, upper_mean, predicted_mean_sell)
            ect_alpha = get_alpha_value(model_fitted, target_index, history)
            
            # 4. GARCH Forecasting
            target_resids = model_fitted.resid[:, history.columns.get_loc(target_index)]
            garch_model = arch_model(target_resids, vol='EGARCH', p=p_opt, o=o_opt, q=q_opt, rescale=True)
            garch_fit = garch_model.fit(disp='off')
            garch_forecast = garch_fit.forecast(horizon=1, start=len(target_resids)-p_opt)
            
            garch_mean_vals = garch_forecast.mean.values
            if np.isnan(garch_mean_vals).any():
                garch_mean_avg = np.mean(garch_mean_vals[~np.isnan(garch_mean_vals)]) if len(garch_mean_vals[~np.isnan(garch_mean_vals)]) > 0 else 0.0
            else:
                garch_mean_avg = np.mean(garch_mean_vals)
                
            garch_vol = np.sqrt(garch_forecast.variance.values[-1, 0])
            
            # 5. Macro Momentum
            cl_p = history['CL=F'].iloc[-21] if len(history) > 20 else history['CL=F'].iloc[0]
            cl_mom = (history['CL=F'].iloc[-1] - cl_p) / cl_p
            
            zb_p = history['ZB=F'].iloc[-21] if len(history) > 20 else history['ZB=F'].iloc[0]
            zb_mom = (history['ZB=F'].iloc[-1] - zb_p) / zb_p
            
            # 6. Store
            results_list.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'actual_price': test_data.iloc[t][target_index],
                'high_price': ohlc_test.iloc[t]['High'][target_index],
                'low_price': ohlc_test.iloc[t]['Low'][target_index],
                'close_price': ohlc_test.iloc[t]['Close'][target_index],
                'predicted_mean_buy': predicted_mean_buy,
                'predicted_mean_sell': predicted_mean_sell,
                'confidence_buy': conf_buy,
                'confidence_sell': conf_sell,
                'lower_mean': lower_mean,
                'upper_mean': upper_mean,
                'garch_mean_average': garch_mean_avg,
                'garch_volatility': garch_vol,
                'CL=F_momentum': cl_mom,
                'ZB=F_momentum': zb_mom,
                'ECT_Alpha': ect_alpha
            })
            
            if (t + 1) % 10 == 0:
                print(f"  Processed {t + 1}/{len(test_data)} days... (Checkpointing)")
                pd.DataFrame(results_list).to_csv(output_path, index=False)
                
        except Exception as e:
            print(f"  Error on {current_date}: {e}")
            continue

    pd.DataFrame(results_list).to_csv(output_path, index=False)
    print(f"\nSuccess! High-fidelity signals saved to: {output_path}")

if __name__ == "__main__":
    precompute_signals()

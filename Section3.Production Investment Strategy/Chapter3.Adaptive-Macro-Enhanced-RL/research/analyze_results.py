import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.api import VECM
from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

def get_alpha_value(model_fitted, target_index, history):
    try:
        alpha = model_fitted.alpha
        target_idx = history.columns.get_loc(target_index)
        if alpha.ndim == 2:
            if target_idx < alpha.shape[0] and alpha.shape[1] > 0:
                target_alphas = alpha[target_idx, :]
                negative_alphas = target_alphas[target_alphas < 0]
                if len(negative_alphas) > 0:
                    abs_negative = np.abs(negative_alphas)
                    weights = abs_negative / np.sum(abs_negative)
                    return float(np.sum(negative_alphas * weights))
                else:
                    return float(np.mean(target_alphas))
        elif alpha.ndim == 1:
            if target_idx < len(alpha):
                return float(alpha[target_idx])
        return 0.0
    except:
        return 0.0

def run_analysis():
    print("Downloading data...")
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    tickers = ['TQQQ', 'NQ=F', 'ZB=F', 'UNG', 'RB=F', 'BZ=F', 'CL=F']
    target_index = 'TQQQ'
    
    df = yf.download(tickers, start=start_date, end=end_date, interval='1d', auto_adjust=True, progress=False)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.asfreq('D')
    df = df.dropna()
    df = df[(df >= 0).all(axis=1)]
    
    split_index = int(len(df) * 0.7)
    train_data = df['Open'].iloc[:split_index]
    test_data = df['Open'].iloc[split_index:]
    
    # Calculate initial lags on train data to save time (rolling Johansen lags is expensive)
    lag_order = select_order(train_data, maxlags=15, deterministic="colo")
    time_lag = lag_order.aic
    print(f"Using fixed time_lag: {time_lag} for performance")
    
    history = train_data.copy()
    results = []
    
    total_steps = len(test_data)
    print(f"Total test steps to analyze: {total_steps}")
    
    for t in range(total_steps - 7):  # Save last 7 days for forward return
        date = test_data.index[t]
        current_price = test_data[target_index].iloc[t]
        
        # Forward 5-day return
        forward_price = test_data[target_index].iloc[t+5]
        forward_return = (forward_price - current_price) / current_price
        
        # Johansen Cointegration Rank on rolling history
        coint_test = select_coint_rank(history, det_order=1, k_ar_diff=time_lag, method='trace')
        rank = coint_test.rank
        
        # Fit VECM to get ECT alpha
        coint_rank_v = rank if rank > 0 else 1 # fallback if 0
        model = VECM(history, k_ar_diff=time_lag, coint_rank=coint_rank_v, deterministic="colo")
        fitted = model.fit()
        alpha = get_alpha_value(fitted, target_index, history)
        
        # Current Macro Data (Normalized roughly as pct change from 20 days ago)
        macro_state = {}
        for ticker in tickers:
            if ticker != target_index:
                past_val = history[ticker].iloc[-20]
                curr_val = history[ticker].iloc[-1]
                macro_state[f'{ticker}_momentum'] = (curr_val - past_val) / past_val
        
        results.append({
            'date': date,
            'coint_rank': rank,
            'alpha': alpha,
            'forward_return_5d': forward_return,
            **macro_state
        })
        
        history = pd.concat([history, test_data.iloc[[t]]])
        
        if (t+1) % 50 == 0:
            print(f"Processed {t+1}/{total_steps}")
            
    results_df = pd.DataFrame(results)
    
    print("\n--- 1. False Positive Analysis ---")
    # False Positive: Cointegrated (rank >= 1) BUT return is below -5%
    false_positives = results_df[(results_df['coint_rank'] > 0) & (results_df['forward_return_5d'] <= -0.05)]
    print(f"Found {len(false_positives)} False Positive instances.")
    if len(false_positives) > 0:
        print("Macro characteristics during False Positives (average momentum):")
        for col in [c for c in false_positives.columns if '_momentum' in c]:
            print(f"  {col}: {false_positives[col].mean():.4f}")
            
    print("\n--- 2. Structural Break Analysis ---")
    # Structural Break: Alpha > 0 (Diverging) BUT return is strong positive (> 5%)
    struct_breaks = results_df[(results_df['alpha'] > 0) & (results_df['forward_return_5d'] >= 0.05)]
    print(f"Found {len(struct_breaks)} Structural Break instances.")
    
    # Calculate Opportunity Cost
    # Assuming the engine reduced fraction to minimum (e.g. 0.2) instead of fully invested (0.8)
    missed_profit_total = 0
    test_dates = test_data.index
    for idx, row in struct_breaks.iterrows():
        # missed profit for this 5-day window is roughly 0.6 * forward_return
        # because the engine only took 20% participation instead of 80%
        missed = 0.6 * row['forward_return_5d']
        missed_profit_total += missed
        
    print(f"Estimated aggregate Opportunity Cost (missed return) during Structural Breaks: {missed_profit_total*100:.2f}%")

if __name__ == "__main__":
    run_analysis()

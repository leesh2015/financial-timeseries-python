import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

def analyze_momentum_windows():
    print("Downloading data for Window Analysis...")
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    tickers = ['TQQQ', 'CL=F']
    target_index = 'TQQQ'
    
    df = yf.download(tickers, start=start_date, end=end_date, interval='1d', auto_adjust=True, progress=False)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.asfreq('D')
    df = df.dropna()
    df = df[(df >= 0).all(axis=1)]
    
    split_index = int(len(df) * 0.7)
    test_data = df.iloc[split_index:]
    
    windows = [5, 10, 15, 20, 25, 30, 40, 60]
    results = []
    
    # Pre-calculate 5-day forward returns for TQQQ
    forward_returns = []
    for t in range(len(test_data) - 7):
        current_price = test_data['Open'][target_index].iloc[t]
        forward_price = test_data['Open'][target_index].iloc[t+5]
        ret = (forward_price - current_price) / current_price
        forward_returns.append(ret)
        
    crashes = [i for i, r in enumerate(forward_returns) if r <= -0.05]
    rallies = [i for i, r in enumerate(forward_returns) if r >= 0.05]
    total_crashes = len(crashes)
    total_rallies = len(rallies)
    
    print(f"Total Test Days Evaluated: {len(forward_returns)}")
    print(f"Total Crashes (<-5%): {total_crashes}")
    print(f"Total Rallies (>+5%): {total_rallies}\n")
    
    # We want a fair comparison. For each window W, 
    # we find the top 15% momentum threshold in the historical data up to that point.
    # To keep it simple, we just use the test data distribution.
    
    for w in windows:
        # Calculate momentum for window w on entire df to get historical perspective
        all_mom = df['Open']['CL=F'].pct_change(periods=w).dropna()
        # Find the 85th percentile threshold for this window (Top 15% inflation spikes)
        threshold = all_mom.quantile(0.85)
        
        blocked_crashes = 0
        blocked_rallies = 0
        
        for t in range(len(test_data) - 7):
            # Calculate momentum for this specific day
            past_val = df['Open']['CL=F'].iloc[split_index + t - w]
            curr_val = df['Open']['CL=F'].iloc[split_index + t]
            mom = (curr_val - past_val) / past_val if past_val > 0 else 0
            
            # If momentum exceeds the threshold, this trade is "blocked"
            if mom > threshold:
                if t in crashes:
                    blocked_crashes += 1
                if t in rallies:
                    blocked_rallies += 1
        
        crash_saved_pct = (blocked_crashes / total_crashes) * 100 if total_crashes > 0 else 0
        rally_missed_pct = (blocked_rallies / total_rallies) * 100 if total_rallies > 0 else 0
        efficiency = (crash_saved_pct / rally_missed_pct) if rally_missed_pct > 0 else float('inf')
        
        results.append({
            'Window (Days)': w,
            'Threshold (Top 15%)': f"+{threshold*100:.1f}%",
            'Crashes Saved': f"{blocked_crashes}/{total_crashes} ({crash_saved_pct:.1f}%)",
            'Rallies Missed': f"{blocked_rallies}/{total_rallies} ({rally_missed_pct:.1f}%)",
            'Efficiency Ratio': efficiency
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    analyze_momentum_windows()

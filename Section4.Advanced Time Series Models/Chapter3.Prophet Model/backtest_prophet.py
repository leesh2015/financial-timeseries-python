"""
Backtest for Chapter 3: Prophet Model
Forecast-based trading strategy
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import load_nasdaq_tqqq_data
from utils.backtest import BacktestEngine, walk_forward_backtest

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


def generate_trading_signals_prophet(prices, lookback=252, forecast_horizon=5,
                                     price_threshold=0.02, retrain_interval=5):
    """
    Generate trading signals based on Prophet forecast
    
    Strategy:
    - Buy when forecast > current price * (1 + threshold)
    - Sell when forecast < current price * (1 - threshold)
    - Hold otherwise
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
    lookback : int
        Minimum training period
    forecast_horizon : int
        Days ahead to forecast
    price_threshold : float
        Minimum price change threshold for signal
    
    Returns:
    --------
    pd.Series
        Trading signals
    """
    if not HAS_PROPHET:
        raise ImportError("Prophet is not installed. Install with: pip install prophet")
    
    signals = pd.Series('hold', index=prices.index, dtype=object)
    cached_forecasts = {}
    last_trained_idx = None
    
    for i in range(lookback, len(prices)):
        current_date = prices.index[i]
        history = prices.iloc[:i]  # exclude current day to avoid look-ahead
        if len(history) < lookback:
            continue
        
        need_retrain = (last_trained_idx is None) or ((i - last_trained_idx) >= retrain_interval)
        if need_retrain:
            try:
                df = pd.DataFrame({'ds': history.index, 'y': history.values})
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05
                )
                model.fit(df)
                future = model.make_future_dataframe(periods=forecast_horizon)
                forecast = model.predict(future)
                forecast_tail = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_horizon)
                cached_forecasts = {
                    pd.Timestamp(row['ds']): {
                        'yhat': row['yhat'],
                        'lower': row['yhat_lower'],
                        'upper': row['yhat_upper']
                    }
                    for _, row in forecast_tail.iterrows()
                }
                last_trained_idx = i
            except Exception:
                cached_forecasts = {}
        
        forecast_info = cached_forecasts.get(current_date)
        if not forecast_info:
            signals.loc[current_date] = 'hold'
            continue
        
        last_known_price = history.iloc[-1]
        price_change = (forecast_info['yhat'] - last_known_price) / last_known_price
        if price_change > price_threshold and forecast_info['lower'] > last_known_price:
            signals.loc[current_date] = 'buy'
        elif price_change < -price_threshold and forecast_info['upper'] < last_known_price:
            signals.loc[current_date] = 'sell'
        else:
            signals.loc[current_date] = 'hold'
    
    # Shift by one day so trades execute with next day's price (no same-day trading)
    signals = signals.shift(1).fillna('hold')
    return signals


def run_backtest_prophet():
    """
    Run backtest for Prophet model
    
    백테스트 원리:
    ------------
    1. Walk-Forward 방식:
       - Train/Test Split (70/30): 전체 데이터의 70%는 학습, 30%는 테스트
       - Test 기간에서만 실제 거래 실행
       - Look-ahead bias 방지: 각 시점에서 과거 데이터만 사용
    
    2. 신호 생성 (Prophet 기반):
       - 각 시점에서 과거 데이터로 Prophet 모델 학습
       - 5일 후 가격 예측 (forecast_horizon=5)
       - 예측 가격이 현재 가격보다 2% 이상 높으면 매수
       - 예측 가격이 현재 가격보다 2% 이상 낮으면 매도
       - 신뢰구간도 고려 (forecast_lower/upper)
    
    3. 거래 실행:
       - 매수: 포지션이 없을 때만 가능 (분할매수 없음)
       - 매도: 포지션이 있을 때만 가능 (전량 매도)
       - 수수료: 0.02% (매수/매도 각각)
    
    4. 승률 계산:
       - Win Rate = (이익 거래 수 / 전체 SELL 거래 수) * 100
       - SELL 거래만 카운트 (매수→매도 완료된 거래)
       - pnl > 0이면 이익 거래, pnl <= 0이면 손실 거래
       - 100% = 모든 매도 거래에서 이익 발생
    """
    print("="*60)
    print("Chapter 3: Prophet Model - Backtest")
    print("="*60)
    
    if not HAS_PROPHET:
        print("\nError: Prophet is not installed.")
        print("Install with: pip install prophet")
        return
    
    # Load data
    data = load_nasdaq_tqqq_data(start_date='2020-01-01')
    tqqq = data['tqqq']
    
    # Generate signals
    print("\n1. Generating trading signals using Prophet forecast...")
    signals = generate_trading_signals_prophet(
        tqqq['Close'], lookback=252, forecast_horizon=5, price_threshold=0.02
    )
    
    # Run backtest
    print("2. Running backtest...")
    results = walk_forward_backtest(tqqq['Close'], signals, train_ratio=0.7)
    
    # Print results
    print(f"\n{'='*60}")
    print("Prophet Backtest Results")
    print(f"{'='*60}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Benchmark Return: {results['benchmark_return']:.2f}%")
    print(f"Annualized Return: {results['annualized_return']:.2f}%")
    print(f"Volatility: {results['volatility']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    
    # Explain win rate calculation
    if results['total_trades'] > 0:
        trades_df = pd.DataFrame(results['trade_history'])
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        if len(sell_trades) > 0:
            print(f"\n   Win Rate Explanation:")
            print(f"   - Total SELL trades: {len(sell_trades)}")
            print(f"   - Winning trades (pnl > 0): {(sell_trades['pnl'] > 0).sum()}")
            print(f"   - Losing trades (pnl <= 0): {(sell_trades['pnl'] <= 0).sum()}")
            if len(sell_trades) > 0:
                avg_pnl = sell_trades['pnl_pct'].mean()
                print(f"   - Average P&L per trade: {avg_pnl:.2f}%")
                print(f"   - Win Rate = (Winning trades / Total SELL trades) * 100")
                print(f"   - Note: Only completed trades (BUY → SELL) are counted")
    
    # Visualize
    visualize_backtest_results(results, tqqq['Close'])
    
    return results


def visualize_backtest_results(results, prices):
    """Visualize backtest results"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    equity_curve = results['equity_curve']
    test_prices = prices.loc[equity_curve.index]
    
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    ax1.plot(test_prices.index, test_prices.values, 
            label='TQQQ Price', linewidth=2, color='blue', alpha=0.7)
    ax1_twin.plot(equity_curve.index, equity_curve['total_value'].values,
                  label='Portfolio Value', linewidth=2, color='red')
    
    ax1.set_ylabel('Price ($)', fontsize=12, color='blue')
    ax1_twin.set_ylabel('Portfolio Value ($)', fontsize=12, color='red')
    ax1.set_title('Equity Curve vs Price', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    portfolio_values = equity_curve['total_value']
    portfolio_returns = portfolio_values.pct_change().dropna()
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative / running_max - 1) * 100
    
    axes[1].fill_between(drawdown.index, drawdown.values, 0, 
                        alpha=0.3, color='red', label='Drawdown')
    axes[1].plot(drawdown.index, drawdown.values, linewidth=2, color='red')
    axes[1].set_ylabel('Drawdown (%)', fontsize=12)
    axes[1].set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    trades_df = pd.DataFrame(results['trade_history'])
    if len(trades_df) > 0:
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        axes[2].scatter(buy_trades['date'], buy_trades['price'], 
                       marker='^', color='green', s=100, label='Buy', zorder=5)
        axes[2].scatter(sell_trades['date'], sell_trades['price'], 
                       marker='v', color='red', s=100, label='Sell', zorder=5)
        axes[2].plot(test_prices.index, test_prices.values, 
                    linewidth=1, color='gray', alpha=0.5, label='Price')
        axes[2].set_ylabel('Price ($)', fontsize=12)
        axes[2].set_xlabel('Date', fontsize=12)
        axes[2].set_title('Trade History', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_backtest_prophet()


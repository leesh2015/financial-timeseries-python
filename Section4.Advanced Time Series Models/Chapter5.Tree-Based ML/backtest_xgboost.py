"""
Backtest for Chapter 5: Tree-Based ML (XGBoost)
XGBoost log return prediction-based trading strategy
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import json
import pickle

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import load_nasdaq_tqqq_data
from utils.backtest import BacktestEngine, walk_forward_backtest
from xgboost_forecast import create_features

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def generate_trading_signals_xgboost(returns, prices, model_name='tqqq', lookback=20,
                                     prediction_threshold=None, use_adaptive_threshold=True):
    """
    Generate trading signals based on pre-trained XGBoost model
    
    Strategy:
    - Buy when predicted log return > threshold
    - Sell when predicted log return < -threshold
    - Hold otherwise
    - Uses pre-trained model from xgboost_forecast.py
    - For TQQQ (high volatility), uses adaptive threshold based on prediction distribution
    
    Parameters:
    -----------
    returns : pd.Series
        Log return series
    prices : pd.Series
        Price series
    model_name : str
        Model name ('nasdaq' or 'tqqq')
    lookback : int
        Feature lookback period (must match trained model)
    prediction_threshold : float, optional
        Minimum predicted log return for signal. If None, uses adaptive threshold.
    use_adaptive_threshold : bool
        If True, adjusts threshold based on prediction distribution (for high volatility assets)
    
    Returns:
    --------
    pd.Series
        Trading signals
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
    
    # Load model and metadata
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_path = os.path.join(models_dir, f'xgboost_{model_name}.pkl')
    metadata_path = os.path.join(models_dir, f'xgboost_{model_name}_metadata.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please run xgboost_forecast.py first to train and save the model."
        )
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}\n"
            f"Please run xgboost_forecast.py first to train and save the model."
        )
    
    # Load model
    print(f"   Loading pre-trained model: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    model_lookback = metadata.get('lookback', lookback)
    model_test_rmse = metadata.get('test_rmse', None)
    model_test_r2 = metadata.get('test_r2', None)
    
    if model_lookback != lookback:
        print(f"   Warning: Model lookback ({model_lookback}) != specified lookback ({lookback})")
        print(f"   Using model lookback: {model_lookback}")
        lookback = model_lookback
    
    print(f"   Model loaded successfully!")
    print(f"   - Model lookback: {lookback}")
    if model_test_rmse is not None:
        print(f"   - Model test RMSE: {model_test_rmse:.6f}")
    if model_test_r2 is not None:
        print(f"   - Model test R²: {model_test_r2:.4f}")
    
    feature_matrix = create_features(returns, prices, lookback)
    feature_matrix = feature_matrix.shift(1).dropna()  # only use info up to t-1
    feature_matrix = feature_matrix.reindex(prices.index).dropna()
    
    if feature_matrix.empty:
        print("   ⚠️  Warning: Not enough feature data to generate signals.")
        return pd.Series('hold', index=prices.index)
    
    preds = pd.Series(model.predict(feature_matrix.values), index=feature_matrix.index)
    
    abs_preds = np.abs(preds.values)
    if prediction_threshold is None or use_adaptive_threshold:
        if model_name == 'tqqq':
            adaptive_threshold = np.percentile(abs_preds, 75)
        else:
            adaptive_threshold = np.percentile(abs_preds, 70)
        if prediction_threshold is None:
            prediction_threshold = adaptive_threshold
            print(f"   Using adaptive threshold: {prediction_threshold:.6f}")
        elif use_adaptive_threshold:
            original_threshold = prediction_threshold
            prediction_threshold = max(prediction_threshold, adaptive_threshold)
            print(f"   Adjusted threshold from {original_threshold:.6f} to {prediction_threshold:.6f}")
    else:
        print(f"   Using fixed threshold: {prediction_threshold:.6f}")
    
    signals = pd.Series('hold', index=prices.index, dtype=object)
    buy_dates = preds.index[preds > prediction_threshold]
    sell_dates = preds.index[preds < -prediction_threshold]
    signals.loc[buy_dates] = 'buy'
    signals.loc[sell_dates] = 'sell'
    signals = signals.shift(1).fillna('hold')  # execute next day
    
    print(f"\n   Prediction Statistics:")
    print(f"   - Total predictions: {len(preds)}")
    print(f"   - Min predicted return: {preds.min():.6f}")
    print(f"   - Max predicted return: {preds.max():.6f}")
    print(f"   - Mean predicted return: {preds.mean():.6f}")
    print(f"   - Std predicted return: {preds.std():.6f}")
    
    total_signals = (signals != 'hold').sum()
    print(f"\n   Signal Statistics:")
    print(f"   - Total signals (non-hold): {total_signals}")
    print(f"   - Buy signals: {(signals == 'buy').sum()}")
    print(f"   - Sell signals: {(signals == 'sell').sum()}")
    print(f"   - Hold signals: {(signals == 'hold').sum()}")
    
    return signals


def run_backtest_xgboost():
    """Run backtest for XGBoost model"""
    print("="*60)
    print("Chapter 5: Tree-Based ML (XGBoost) - Backtest")
    print("Log Return Prediction Strategy")
    print("="*60)
    
    if not HAS_XGBOOST:
        print("\nError: XGBoost is not installed.")
        print("Install with: pip install xgboost")
        return
    
    # Load data
    data = load_nasdaq_tqqq_data(start_date='2020-01-01')
    tqqq = data['tqqq']
    
    # Generate signals using pre-trained model
    print("\n1. Generating trading signals using pre-trained XGBoost model...")
    print("   Strategy: Buy when predicted log return > threshold, Sell when < -threshold")
    print("   Note: Uses adaptive threshold for TQQQ (high volatility asset)")
    print("   Note: Model must be trained first using xgboost_forecast.py")
    try:
        signals = generate_trading_signals_xgboost(
            tqqq['Log_Returns'].dropna(), tqqq['Close'],
            model_name='tqqq', lookback=20, 
            prediction_threshold=None,  # Use adaptive threshold
            use_adaptive_threshold=True
        )
    except FileNotFoundError as e:
        print(f"\n   Error: {e}")
        print("\n   Please run xgboost_forecast.py first to train and save the model.")
        return None
    
    # Run backtest
    print("2. Running backtest...")
    results = walk_forward_backtest(tqqq['Close'], signals, train_ratio=0.7)
    
    # Print results
    print(f"\n{'='*60}")
    print("Backtest Results")
    print(f"{'='*60}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Benchmark Return: {results['benchmark_return']:.2f}%")
    print(f"Annualized Return: {results['annualized_return']:.2f}%")
    print(f"Volatility: {results['volatility']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    
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
    run_backtest_xgboost()


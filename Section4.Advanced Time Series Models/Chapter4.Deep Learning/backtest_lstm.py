"""
Backtest for Chapter 4: Deep Learning Models (LSTM)
LSTM direction prediction-based trading strategy
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

from utils.data_loader import load_nasdaq_tqqq_data, prepare_sequences
from utils.backtest import BacktestEngine, walk_forward_backtest

warnings.filterwarnings("ignore")

try:
    from tensorflow.keras.models import load_model  # type: ignore
    from sklearn.preprocessing import MinMaxScaler
    import json
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


def generate_trading_signals_lstm(returns, prices, model_name='tqqq', lookback=60):
    """
    Generate trading signals based on pre-trained LSTM direction prediction model
    
    Strategy:
    - Buy when predicted direction is Up (probability > optimal threshold)
    - Sell when predicted direction is Down (probability < optimal threshold)
    - Uses pre-trained model from lstm_forecast.py
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    prices : pd.Series
        Price series
    model_name : str
        Model name ('nasdaq' or 'tqqq')
    lookback : int
        LSTM lookback period (must match trained model)
    
    Returns:
    --------
    pd.Series
        Trading signals
    """
    if not HAS_TENSORFLOW:
        raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")
    
    # Load model and metadata
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_path = os.path.join(models_dir, f'lstm_{model_name}_direction.h5')
    metadata_path = os.path.join(models_dir, f'lstm_{model_name}_metadata.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please run lstm_forecast.py first to train and save the model."
        )
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}\n"
            f"Please run lstm_forecast.py first to train and save the model."
        )
    
    # Load model
    print(f"   Loading pre-trained model: {model_path}")
    model = load_model(model_path)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    optimal_threshold = metadata.get('optimal_threshold', 0.5)
    model_lookback = metadata.get('lookback', lookback)
    model_test_accuracy = metadata.get('test_accuracy', None)
    model_test_roc_auc = metadata.get('test_roc_auc', None)
    
    if model_lookback != lookback:
        print(f"   Warning: Model lookback ({model_lookback}) != specified lookback ({lookback})")
        print(f"   Using model lookback: {model_lookback}")
        lookback = model_lookback
    
    print(f"   Model loaded successfully!")
    print(f"   - Model lookback: {lookback}")
    print(f"   - Optimal threshold: {optimal_threshold:.4f}")
    if model_test_accuracy is not None:
        print(f"   - Model test accuracy: {model_test_accuracy:.4f} ({model_test_accuracy*100:.2f}%)")
    if model_test_roc_auc is not None:
        print(f"   - Model test ROC-AUC: {model_test_roc_auc:.4f}")
    
    # Verify model architecture
    print(f"   - Model input shape: {model.input_shape}")
    print(f"   - Model output shape: {model.output_shape}")
    
    # Prepare scaler using TRAIN period only (prevents leakage)
    train_ratio = metadata.get('train_ratio', 0.8)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_cutoff = max(lookback, int(len(returns) * train_ratio))
    scaler_fit_values = returns.iloc[:scaler_cutoff].values.reshape(-1, 1)
    scaler.fit(scaler_fit_values)
    scaled_returns = pd.Series(
        scaler.transform(returns.values.reshape(-1, 1)).flatten(),
        index=returns.index
    )
    
    signals = pd.Series('hold', index=prices.index, dtype=object)
    pred_probas = []  # Store predictions for debugging
    
    # Generate signals
    print(f"   Generating signals for {len(returns) - lookback} days...")
    for i in range(lookback, len(returns)):
        current_date = returns.index[i]
        
        if current_date not in prices.index:
            continue
        
        try:
            # IMPORTANT: Use data up to i-1 for prediction (not including current)
            # This ensures we don't use current price to predict current price
            recent_data = scaled_returns.iloc[i-lookback:i].values  # already scaled using train stats
            
            if len(recent_data) < lookback:
                continue
                    
            # Reshape for prediction
            X_pred = recent_data.reshape(1, lookback, 1)
            
            # Predict direction probability
            pred_proba = model.predict(X_pred, verbose=0)[0, 0]
            pred_probas.append(pred_proba)
            
            # Generate signal based on optimal threshold
            if pred_proba >= optimal_threshold:
                signals.loc[current_date] = 'buy'
            else:
                signals.loc[current_date] = 'sell'
        except Exception as e:
            signals.loc[current_date] = 'hold'
    
    # Debug: Print prediction statistics
    if len(pred_probas) > 0:
        pred_array = np.array(pred_probas)
        print(f"\n   Prediction Statistics:")
        print(f"   - Total predictions: {len(pred_probas)}")
        print(f"   - Min probability: {pred_array.min():.4f}")
        print(f"   - Max probability: {pred_array.max():.4f}")
        print(f"   - Mean probability: {pred_array.mean():.4f}")
        print(f"   - Median probability: {np.median(pred_array):.4f}")
        print(f"   - Predictions >= threshold ({optimal_threshold:.4f}): {(pred_array >= optimal_threshold).sum()} ({(pred_array >= optimal_threshold).sum()/len(pred_array)*100:.1f}%)")
        print(f"   - Predictions < threshold: {(pred_array < optimal_threshold).sum()} ({(pred_array < optimal_threshold).sum()/len(pred_array)*100:.1f}%)")
    
    # Debug: Print signal statistics
    total_signals = len(signals[signals != 'hold'])
    buy_signals = len(signals[signals == 'buy'])
    sell_signals = len(signals[signals == 'sell'])
    print(f"\n   Signal Statistics:")
    print(f"   - Total signals (non-hold): {total_signals}")
    print(f"   - Buy signals: {buy_signals}")
    print(f"   - Sell signals: {sell_signals}")
    print(f"   - Hold signals: {len(signals) - total_signals}")
    
    return signals


def run_backtest_lstm():
    """Run backtest for LSTM model"""
    print("="*60)
    print("Chapter 4: Deep Learning (LSTM) - Backtest")
    print("Direction Prediction Strategy")
    print("="*60)
    
    if not HAS_TENSORFLOW:
        print("\nError: TensorFlow is not installed.")
        print("Install with: pip install tensorflow")
        return
    
    # Load data
    data = load_nasdaq_tqqq_data(start_date='2020-01-01')
    tqqq = data['tqqq']
    
    # Generate signals using pre-trained model
    print("\n1. Generating trading signals using pre-trained LSTM model...")
    print("   Strategy: Buy when predicted direction is Up, Sell when Down")
    print("   Note: Model must be trained first using lstm_forecast.py")
    try:
        signals = generate_trading_signals_lstm(
            tqqq['Log_Returns'].dropna(), tqqq['Close'],
            model_name='tqqq', lookback=60
        )
    except FileNotFoundError as e:
        print(f"\n   Error: {e}")
        print("\n   Please run lstm_forecast.py first to train and save the model.")
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
    run_backtest_lstm()


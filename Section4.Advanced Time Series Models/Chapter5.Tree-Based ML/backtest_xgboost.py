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
                                     prediction_threshold=None, use_adaptive_threshold=False,
                                     train_ratio=0.7):
    """
    Generate trading signals based on pre-trained XGBoost binary classification model
    
    Strategy:
    - Buy when predicted probability of up > threshold (default: optimal threshold from training)
    - Sell when predicted probability of up < (1 - threshold)
    - Hold otherwise
    - Uses pre-trained binary classification model from xgboost_forecast.py
    - Model predicts direction (up/down) not return value
    
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
        Minimum predicted probability for buy signal. If None, uses optimal threshold from model metadata.
    use_adaptive_threshold : bool
        If True, adjusts threshold based on prediction distribution (currently not used for binary classification)
    train_ratio : float
        Training period ratio (for compatibility, not used for threshold calculation in binary classification)
    
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
    model_type = metadata.get('model_type', 'binary_classification')
    model_test_accuracy = metadata.get('test_accuracy', None)
    model_test_f1 = metadata.get('test_f1', None)
    model_test_roc_auc = metadata.get('test_roc_auc', None)
    optimal_threshold = metadata.get('optimal_threshold', 0.5)
    
    if model_lookback != lookback:
        print(f"   Warning: Model lookback ({model_lookback}) != specified lookback ({lookback})")
        print(f"   Using model lookback: {model_lookback}")
        lookback = model_lookback
    
    print(f"   Model loaded successfully!")
    print(f"   - Model type: {model_type}")
    print(f"   - Model lookback: {lookback}")
    if model_test_accuracy is not None:
        print(f"   - Model test accuracy: {model_test_accuracy:.4f} ({model_test_accuracy*100:.2f}%)")
    if model_test_f1 is not None:
        print(f"   - Model test F1: {model_test_f1:.4f}")
    if model_test_roc_auc is not None:
        print(f"   - Model test ROC-AUC: {model_test_roc_auc:.4f}")
    
    # Create features (same as training)
    feature_matrix = create_features(returns, prices, lookback)
    feature_matrix = feature_matrix.shift(1).dropna()  # only use info up to t-1
    feature_matrix = feature_matrix.reindex(prices.index).dropna()
    
    if feature_matrix.empty:
        print("   ⚠️  Warning: Not enough feature data to generate signals.")
        return pd.Series('hold', index=prices.index)
    
    # CRITICAL: Use only features that were used during training
    # This ensures consistency between training and backtesting
    feature_names = metadata.get('feature_names', None)
    if feature_names is not None:
        # Check if all training features are present
        missing_features = set(feature_names) - set(feature_matrix.columns)
        if missing_features:
            print(f"   ⚠️  Warning: Missing features from training: {missing_features}")
            # Use only available features
            available_features = [f for f in feature_names if f in feature_matrix.columns]
            if len(available_features) < len(feature_names):
                print(f"   ⚠️  Warning: Using {len(available_features)}/{len(feature_names)} training features")
        else:
            available_features = feature_names
        
        # Select features in the same order as training
        feature_matrix = feature_matrix[available_features]
        print(f"   Using {len(available_features)} features (same as training)")
    else:
        print(f"   ⚠️  Warning: feature_names not found in metadata, using all {len(feature_matrix.columns)} features")
    
    # Get prediction probabilities (probability of up day)
    pred_proba = model.predict_proba(feature_matrix.values)[:, 1]
    preds = pd.Series(pred_proba, index=feature_matrix.index)
    
    # Use optimal threshold from model metadata if not specified
    if prediction_threshold is None:
        prediction_threshold = optimal_threshold
        print(f"   Using optimal threshold from model: {prediction_threshold:.4f}")
    else:
        print(f"   Using specified threshold: {prediction_threshold:.4f}")
    
    # Generate signals based on probability
    # Buy when probability of up > threshold
    # Sell when probability of up < (1 - threshold) or we can use a lower threshold
    sell_threshold = 1 - prediction_threshold  # Symmetric: if buy threshold is 0.6, sell threshold is 0.4
    
    signals = pd.Series('hold', index=prices.index, dtype=object)
    buy_dates = preds.index[preds > prediction_threshold]
    sell_dates = preds.index[preds < sell_threshold]
    signals.loc[buy_dates] = 'buy'
    signals.loc[sell_dates] = 'sell'
    signals = signals.shift(1).fillna('hold')  # execute next day
    
    print(f"\n   Prediction Statistics:")
    print(f"   - Total predictions: {len(preds)}")
    print(f"   - Min predicted probability: {preds.min():.4f}")
    print(f"   - Max predicted probability: {preds.max():.4f}")
    print(f"   - Mean predicted probability: {preds.mean():.4f}")
    print(f"   - Std predicted probability: {preds.std():.4f}")
    print(f"   - Buy threshold: {prediction_threshold:.4f}")
    print(f"   - Sell threshold: {sell_threshold:.4f}")
    
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
    print("\n1. Generating trading signals using pre-trained XGBoost binary classification model...")
    print("   Strategy: Buy when predicted probability of up > threshold, Sell when < (1 - threshold)")
    print("   Note: Model predicts direction (up/down) not return value")
    print("   Note: Model must be trained first using xgboost_forecast.py")
    train_ratio = 0.7  # Must match walk_forward_backtest train_ratio
    try:
        signals = generate_trading_signals_xgboost(
            tqqq['Log_Returns'].dropna(), tqqq['Close'],
            model_name='tqqq', lookback=20, 
            prediction_threshold=None,  # Use optimal threshold from model metadata
            use_adaptive_threshold=False,  # Not used for binary classification
            train_ratio=train_ratio  # Use same train_ratio as backtest
        )
    except FileNotFoundError as e:
        print(f"\n   Error: {e}")
        print("\n   Please run xgboost_forecast.py first to train and save the model.")
        return None
    
    # Run backtest
    print("2. Running backtest...")
    results = walk_forward_backtest(tqqq['Close'], signals, train_ratio=train_ratio)
    
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


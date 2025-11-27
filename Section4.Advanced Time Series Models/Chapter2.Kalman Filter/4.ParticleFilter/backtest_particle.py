"""
Backtest for Particle Filter-based dynamic beta estimates.

The particle filter captures non-linearities / non-Gaussian noise.
We convert the posterior mean beta into trading signals and run the
shared walk-forward backtest.
"""

import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.data_loader import load_nasdaq_tqqq_data  # noqa: E402
from utils.backtest import walk_forward_backtest  # noqa: E402

from particle_filter_demo import particle_filter_beta  # noqa: E402


def _generate_signals(beta_series: pd.Series, low: float, high: float, index: pd.Index) -> pd.Series:
    beta_shifted = beta_series.shift(1).reindex(index)
    signals = pd.Series("hold", index=index, dtype=object)
    signals.loc[beta_shifted < low] = "buy"
    signals.loc[beta_shifted > high] = "sell"
    return signals


def _describe_signals(signals: pd.Series, low: float, high: float) -> None:
    print("\nSignal Summary")
    print("-" * 40)
    print(f"Thresholds -> low: {low:.2f}, high: {high:.2f}")
    print(f"Total signals: {(signals != 'hold').sum()}")
    print(f"  Buy : {(signals == 'buy').sum()}")
    print(f"  Sell: {(signals == 'sell').sum()}")
    print(f"  Hold: {(signals == 'hold').sum()}")


def run_backtest_particle(
    n_particles: int = 2000,
    process_std: np.ndarray = np.array([0.0007, 0.04]),
    obs_std: float = 0.025,
    train_ratio: float = 0.7,
) -> Tuple[pd.Series, dict]:
    print("=" * 60)
    print("Chapter 2: Particle Filter Backtest")
    print("=" * 60)

    data = load_nasdaq_tqqq_data(start_date="2020-01-01")
    nasdaq_returns = data["nasdaq"]["Returns"]
    tqqq_returns = data["tqqq"]["Returns"]
    tqqq_close = data["tqqq"]["Close"]

    results = particle_filter_beta(
        nasdaq_returns,
        tqqq_returns,
        n_particles=n_particles,
        process_std=process_std,
        obs_std=obs_std,
    )

    beta = results["beta"]
    # Particle filter betas are the noisiest (non-linear + non-Gaussian),
    # so we default to wide bands (10/90) to avoid trading every micro spike.
    low = beta.quantile(0.1)
    high = beta.quantile(0.9)

    signals = _generate_signals(beta, low, high, tqqq_close.index)
    _describe_signals(signals, low, high)

    aligned_signals = signals.loc[tqqq_close.index]

    print("\nRunning walk-forward backtest...")
    bt_results = walk_forward_backtest(tqqq_close, aligned_signals, train_ratio=train_ratio)

    print("\nBacktest Results")
    print("-" * 60)
    for key, value in bt_results.items():
        if isinstance(value, (int, str)):
            print(f"{key}: {value}")
        elif isinstance(value, float):
            print(f"{key}: {value:.2f}")

    split_idx = int(len(tqqq_close) * train_ratio)
    test_prices = tqqq_close.iloc[split_idx:]
    visualize_backtest(test_prices, bt_results)

    return aligned_signals, bt_results


def visualize_backtest(test_prices: pd.Series, bt_results: dict) -> None:
    equity_curve = bt_results["equity_curve"]
    trade_history = pd.DataFrame(bt_results["trade_history"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].plot(test_prices.index, test_prices.values, color="tab:gray", alpha=0.6, label="TQQQ Price")
    if len(trade_history) > 0:
        buys = trade_history[trade_history["action"] == "BUY"]
        sells = trade_history[trade_history["action"] == "SELL"]
        axes[0].scatter(buys["date"], buys["price"], marker="^", color="green", s=90, label="Buy", zorder=5)
        axes[0].scatter(sells["date"], sells["price"], marker="v", color="red", s=90, label="Sell", zorder=5)

    axes[0].set_title("Particle Filter Strategy Trades")
    axes[0].set_ylabel("Price ($)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    strategy_curve = equity_curve["total_value"]
    strategy_cum = strategy_curve / strategy_curve.iloc[0] - 1
    benchmark_cum = test_prices / test_prices.iloc[0] - 1

    axes[1].plot(strategy_cum.index, strategy_cum.values * 100, label="Strategy Return (%)", linewidth=2)
    axes[1].plot(benchmark_cum.index, benchmark_cum.values * 100, label="Benchmark Return (%)", linewidth=2, linestyle="--")
    axes[1].set_title("Strategy vs Benchmark Cumulative Returns")
    axes[1].set_ylabel("Return (%)")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_backtest_particle()


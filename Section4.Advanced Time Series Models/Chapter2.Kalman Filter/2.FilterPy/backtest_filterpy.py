"""
Backtest for Chapter 2: FilterPy Kalman (Dynamic Beta)
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.data_loader import load_nasdaq_tqqq_data  # noqa: E402
from utils.backtest import walk_forward_backtest  # noqa: E402
from kalman_filter_filterpy import estimate_alpha_beta_filterpy, HAS_FILTERPY  # noqa: E402

warnings.filterwarnings("ignore")


def generate_beta_signals(
    beta_series: pd.Series,
    price_index: pd.Index,
    lower_quantile: float = 0.5,
    upper_quantile: float = 0.5,
) -> pd.Series:
    # The adaptive FilterPy Kalman can broaden/narrow beta distributions as Q/R shift.
    # Mid-range quantiles (default 50/50 here) keep the strategy from over-trading extreme noise bursts.
    beta_low = beta_series.quantile(lower_quantile)
    beta_high = beta_series.quantile(upper_quantile)

    beta_shifted = beta_series.shift(1).reindex(price_index)
    signals = pd.Series("hold", index=price_index, dtype=object)
    signals.loc[beta_shifted < beta_low] = "buy"
    signals.loc[beta_shifted > beta_high] = "sell"

    print("\nSignal Summary (FilterPy)")
    print("-" * 50)
    print(f"Thresholds -> low: {beta_low:.2f}, high: {beta_high:.2f}")
    print(f"Total non-hold signals: {(signals != 'hold').sum()}")
    print(f"  Buy : {(signals == 'buy').sum()}")
    print(f"  Sell: {(signals == 'sell').sum()}")
    print(f"  Hold: {(signals == 'hold').sum()}")

    return signals


def run_backtest_filterpy():
    if not HAS_FILTERPY:
        print("filterpy is required. Install with: pip install filterpy")
        return

    print("=" * 60)
    print("Chapter 2: FilterPy Kalman Backtest (Dynamic Beta)")
    print("=" * 60)

    data = load_nasdaq_tqqq_data(start_date="2020-01-01")
    results_alpha_beta = estimate_alpha_beta_filterpy(
        data["nasdaq"]["Returns"],
        data["tqqq"]["Returns"],
        use_adaptive=True,
    )

    signals = generate_beta_signals(results_alpha_beta["beta"], data["tqqq"]["Close"].index)

    print("\nRunning walk-forward backtest...")
    bt_results = walk_forward_backtest(data["tqqq"]["Close"], signals, train_ratio=0.7)

    print("\nBacktest Results (FilterPy)")
    print("-" * 60)
    for key, value in bt_results.items():
        if isinstance(value, (int, str)):
            print(f"{key}: {value}")
        elif isinstance(value, float):
            print(f"{key}: {value:.2f}")

    visualize_backtest_results(bt_results, data["tqqq"]["Close"])


def visualize_backtest_results(results, prices):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    equity_curve = results["equity_curve"]
    test_prices = prices.loc[equity_curve.index]

    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    ax1.plot(test_prices.index, test_prices.values, label="TQQQ Price", linewidth=2, color="blue", alpha=0.7)
    ax1_twin.plot(equity_curve.index, equity_curve["total_value"].values, label="Portfolio Value", linewidth=2, color="red")

    ax1.set_ylabel("Price ($)", fontsize=12, color="blue")
    ax1_twin.set_ylabel("Portfolio Value ($)", fontsize=12, color="red")
    ax1.set_title("Equity Curve vs Price (FilterPy)", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1_twin.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    portfolio_values = equity_curve["total_value"]
    portfolio_returns = portfolio_values.pct_change().dropna()
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative / running_max - 1) * 100

    axes[1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color="red", label="Drawdown")
    axes[1].plot(drawdown.index, drawdown.values, linewidth=2, color="red")
    axes[1].set_ylabel("Drawdown (%)", fontsize=12)
    axes[1].set_title("Portfolio Drawdown", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    trades_df = pd.DataFrame(results["trade_history"])
    if len(trades_df) > 0:
        buy_trades = trades_df[trades_df["action"] == "BUY"]
        sell_trades = trades_df[trades_df["action"] == "SELL"]

        axes[2].scatter(buy_trades["date"], buy_trades["price"], marker="^", color="green", s=100, label="Buy", zorder=5)
        axes[2].scatter(sell_trades["date"], sell_trades["price"], marker="v", color="red", s=100, label="Sell", zorder=5)
        axes[2].plot(test_prices.index, test_prices.values, linewidth=1, color="gray", alpha=0.5, label="Price")
        axes[2].set_ylabel("Price ($)", fontsize=12)
        axes[2].set_xlabel("Date", fontsize=12)
        axes[2].set_title("Trade History", fontsize=14, fontweight="bold")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return results


if __name__ == "__main__":
    run_backtest_filterpy()

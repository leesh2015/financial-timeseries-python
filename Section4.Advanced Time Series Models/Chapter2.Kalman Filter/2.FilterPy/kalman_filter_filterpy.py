"""
Chapter 2: FilterPy Kalman Filter (Dynamic Alpha/Beta)

FilterPy implementation of a linear-Gaussian alpha/beta model with
optional adaptive noise updates.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.data_loader import load_nasdaq_tqqq_data, align_data  # noqa: E402

warnings.filterwarnings("ignore")

try:
    from filterpy.kalman import KalmanFilter as FilterPyKalman
    HAS_FILTERPY = True
except ImportError:
    HAS_FILTERPY = False
    print("Warning: filterpy not available. Install with: pip install filterpy")


def estimate_alpha_beta_filterpy(
    nasdaq_returns: pd.Series,
    tqqq_returns: pd.Series,
    use_adaptive: bool = True,
    forget_factor: float = 0.97,
) -> dict:
    if not HAS_FILTERPY:
        raise ImportError("filterpy is required for this module.")

    nasdaq_ret, tqqq_ret = align_data(nasdaq_returns, tqqq_returns)

    kf = FilterPyKalman(dim_x=2, dim_z=1)
    kf.x = np.array([[0.0], [3.0]])
    kf.F = np.eye(2)
    base_q = np.array([[1e-6, 0.0], [0.0, 1e-3]])
    kf.Q = base_q.copy()
    base_r = max(tqqq_ret.var(), 1e-5)
    kf.R = np.array([[base_r]])
    kf.P = np.eye(2) * 1000.0

    alpha_vals, beta_vals = [], []
    innovation_hist = []

    for r_nasdaq, r_tqqq in zip(nasdaq_ret.values, tqqq_ret.values):
        kf.H = np.array([[1.0, r_nasdaq]])
        kf.predict()
        kf.update(np.array([[r_tqqq]]))

        if use_adaptive:
            innovation = r_tqqq - (kf.H @ kf.x)[0, 0]
            innovation_hist.append(innovation)
            if len(innovation_hist) > 30:
                recent = np.array(innovation_hist[-60:])
                var_innov = max(np.var(recent), base_r * 0.5)
                kf.R[0, 0] = np.clip(
                    forget_factor * kf.R[0, 0] + (1 - forget_factor) * var_innov,
                    base_r * 0.5,
                    base_r * 2.0,
                )
                q_update = np.clip(var_innov * 0.05, base_q[0, 0] * 0.5, base_q[0, 0] * 2.0)
                kf.Q = np.array([[q_update, 0.0], [0.0, q_update]])

        alpha_vals.append(kf.x[0, 0])
        beta_vals.append(kf.x[1, 0])

    alpha_series = pd.Series(alpha_vals, index=nasdaq_ret.index, name="alpha")
    beta_series = pd.Series(beta_vals, index=nasdaq_ret.index, name="beta")
    predicted = alpha_series + beta_series * nasdaq_ret
    residual = tqqq_ret - predicted

    return {
        "alpha": alpha_series,
        "beta": beta_series,
        "predicted_returns": predicted,
        "actual_returns": tqqq_ret,
        "nasdaq_returns": nasdaq_ret,
        "residual": residual,
    }


def visualize_alpha_beta(results: dict):
    alpha = results["alpha"]
    beta = results["beta"]
    residual = results["residual"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    axes[0].plot(beta.index, beta.values, label="FilterPy Beta", color="tab:green")
    axes[0].axhline(3.0, color="tab:red", linestyle="--", label="3x Theoretical")
    axes[0].set_title("Dynamic Beta (FilterPy)")
    axes[0].set_ylabel("Beta")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(alpha.index, alpha.values, label="Alpha", color="tab:purple")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Alpha (Excess Return Component)")
    axes[1].set_ylabel("Alpha")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(residual.index, residual.values, label="Residual", color="tab:gray")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("Residual (Actual - Predicted Returns)")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Residual")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nFilterPy Summary")
    print("-" * 60)
    print(f"Beta mean: {beta.mean():.4f}, std: {beta.std():.4f}, min: {beta.min():.4f}, max: {beta.max():.4f}")
    print(f"Alpha mean: {alpha.mean():.6f}, std: {alpha.std():.6f}")
    print(f"Residual std: {residual.std():.6f}")


def main():
    if not HAS_FILTERPY:
        print("filterpy is required for this demo.")
        return

    print("=" * 60)
    print("Chapter 2: FilterPy Kalman (Dynamic Beta)")
    print("=" * 60)

    data = load_nasdaq_tqqq_data(start_date="2020-01-01")
    results = estimate_alpha_beta_filterpy(data["nasdaq"]["Returns"], data["tqqq"]["Returns"])
    visualize_alpha_beta(results)
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

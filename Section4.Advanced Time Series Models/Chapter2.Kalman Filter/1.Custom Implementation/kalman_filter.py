"""
Chapter 2: Custom Kalman Filter (Dynamic Alpha/Beta)

Reuses a lightweight Kalman implementation to estimate the hidden
alpha/beta relationship between NASDAQ and TQQQ returns.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.data_loader import load_nasdaq_tqqq_data, align_data  # noqa: E402

warnings.filterwarnings("ignore")


class KalmanFilter:
    """Minimal Kalman filter used for instructional purposes."""

    def __init__(self, dim_x: int, dim_z: int):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x) * 1000.0
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P
        return self.x.copy()


def estimate_alpha_beta_custom(
    nasdaq_returns: pd.Series,
    tqqq_returns: pd.Series,
    process_var_alpha: float = 1e-6,
    process_var_beta: float = 1e-3,
) -> dict:
    """
    Estimate dynamic alpha/beta for TQQQ vs NASDAQ using a custom Kalman filter.
    """
    nasdaq_ret, tqqq_ret = align_data(nasdaq_returns, tqqq_returns)

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.0], [3.0]])
    kf.F = np.eye(2)
    kf.Q = np.array([[process_var_alpha, 0.0], [0.0, process_var_beta]])
    kf.R = np.array([[max(tqqq_ret.var(), 1e-5)]])

    alpha_vals, beta_vals = [], []
    for r_nasdaq, r_tqqq in zip(nasdaq_ret.values, tqqq_ret.values):
        kf.H = np.array([[1.0, r_nasdaq]])
        kf.predict()
        kf.update(np.array([[r_tqqq]]))
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

    axes[0].plot(beta.index, beta.values, label="Custom Kalman Beta", color="tab:blue")
    axes[0].axhline(3.0, color="tab:red", linestyle="--", label="3x Theoretical")
    axes[0].set_title("Dynamic Beta (Custom Kalman)")
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

    print("\nCustom Kalman Summary")
    print("-" * 60)
    print(f"Beta mean: {beta.mean():.4f}, std: {beta.std():.4f}, min: {beta.min():.4f}, max: {beta.max():.4f}")
    print(f"Alpha mean: {alpha.mean():.6f}, std: {alpha.std():.6f}")
    print(f"Residual std: {residual.std():.6f}")


def main():
    print("=" * 60)
    print("Chapter 2: Custom Kalman Filter (Dynamic Beta)")
    print("=" * 60)

    data = load_nasdaq_tqqq_data(start_date="2020-01-01")
    results = estimate_alpha_beta_custom(data["nasdaq"]["Returns"], data["tqqq"]["Returns"])
    visualize_alpha_beta(results)
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
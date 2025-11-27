"""
Chapter 2 Extension: PyKalman EM Demonstration

This script shows how to use PyKalman to:
1. Run EM to learn transition/observation covariances
2. Smooth a dynamic alpha/beta relationship between NASDAQ and TQQQ
3. Visualize posterior credible intervals and compare to theoretical beta
"""

import os
import sys
import warnings
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from pykalman import KalmanFilter as PyKalmanFilter
except ImportError as exc:
    raise ImportError(
        "PyKalman is required for this demo. Install with `pip install pykalman`."
    ) from exc

# Add project root to path for shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.data_loader import load_nasdaq_tqqq_data, align_data  # noqa: E402

warnings.filterwarnings("ignore")


def run_pykalman_em(em_iter: int = 30) -> Dict[str, Any]:
    """
    Fit a linear-Gaussian state-space model with PyKalman + EM.

    State:  [alpha_t, beta_t]
    Obs:    r_tqqq = alpha_t + beta_t * r_nasdaq + v_t
    """
    data = load_nasdaq_tqqq_data(start_date="2020-01-01")
    nasdaq_ret, tqqq_ret = align_data(data["nasdaq"]["Returns"], data["tqqq"]["Returns"])

    observations = tqqq_ret.values.reshape(-1, 1)
    obs_idx = tqqq_ret.index

    # Build observation matrices of shape (n_timesteps, dim_z, dim_x)
    design = np.column_stack([np.ones_like(nasdaq_ret.values), nasdaq_ret.values])
    observation_matrices = design.reshape(-1, 1, 2)

    kf = PyKalmanFilter(
        transition_matrices=np.eye(2),
        observation_matrices=observation_matrices,
        transition_covariance=np.diag([1e-6, 1e-3]),
        observation_covariance=np.array([[1e-3]]),
        initial_state_mean=np.array([0.0, 3.0]),
        initial_state_covariance=np.eye(2),
    )

    print(f"Running EM for {em_iter} iterations...")
    kf = kf.em(observations, n_iter=em_iter)

    filtered_means, filtered_covs = kf.filter(observations)
    smoothed_means, smoothed_covs = kf.smooth(observations)

    alpha = pd.Series(smoothed_means[:, 0], index=obs_idx, name="alpha")
    beta = pd.Series(smoothed_means[:, 1], index=obs_idx, name="beta")
    beta_std = np.sqrt(smoothed_covs[:, 1, 1])
    beta_ci_upper = pd.Series(beta.values + 1.96 * beta_std, index=obs_idx)
    beta_ci_lower = pd.Series(beta.values - 1.96 * beta_std, index=obs_idx)

    predicted = alpha + beta * nasdaq_ret
    residual = tqqq_ret - predicted

    return {
        "alpha": alpha,
        "beta": beta,
        "beta_ci_upper": beta_ci_upper,
        "beta_ci_lower": beta_ci_lower,
        "predicted_returns": predicted,
        "actual_returns": tqqq_ret,
        "nasdaq_returns": nasdaq_ret,
        "residual": residual,
        "filtered_means": filtered_means,
        "filtered_covs": filtered_covs,
        "smoothed_covs": smoothed_covs,
        "em_meta": {
            "transition_covariance": kf.transition_covariance,
            "observation_covariance": kf.observation_covariance,
            "iterations": em_iter,
            "log_likelihood": float(kf.loglikelihood(observations)),
        },
    }


def visualize_results(results: Dict[str, Any]) -> None:
    beta = results["beta"]
    alpha = results["alpha"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    axes[0].plot(beta.index, beta.values, label="Smoothed Beta (PyKalman)", color="tab:blue")
    axes[0].fill_between(
        beta.index,
        results["beta_ci_lower"].values,
        results["beta_ci_upper"].values,
        alpha=0.2,
        color="tab:blue",
        label="95% Credible Interval",
    )
    axes[0].axhline(y=3.0, color="tab:red", linestyle="--", label="Theoretical Beta 3x")
    axes[0].set_title("Dynamic Beta via PyKalman EM")
    axes[0].set_ylabel("Beta")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(alpha.index, alpha.values, color="tab:purple", label="Alpha")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Alpha (Excess Return Component)")
    axes[1].set_ylabel("Alpha")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        results["actual_returns"].index,
        results["actual_returns"].values,
        label="Actual TQQQ Returns",
        alpha=0.6,
    )
    axes[2].plot(
        results["predicted_returns"].index,
        results["predicted_returns"].values,
        label="Predicted Returns (PyKalman)",
        alpha=0.8,
        linestyle="--",
    )
    axes[2].set_title("Actual vs Predicted Returns")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Return")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nEM Summary")
    print("-" * 60)
    for k, v in results["em_meta"].items():
        print(f"{k}: {v}")
    print("\nResidual Statistics")
    print("-" * 60)
    print(results["residual"].describe())


def main():
    results = run_pykalman_em(em_iter=30)
    visualize_results(results)


if __name__ == "__main__":
    main()


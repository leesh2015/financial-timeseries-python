"""
Chapter 2 Extension: Particle Filter Demonstration

This script builds a simple bootstrap particle filter to track
the latent alpha/beta relationship between NASDAQ and TQQQ.
"""

import os
import sys
import warnings
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root so we can reuse the shared data loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.data_loader import load_nasdaq_tqqq_data, align_data  # noqa: E402

warnings.filterwarnings("ignore")


def _systematic_resample(weights: np.ndarray) -> np.ndarray:
    """Return systematic-resampled particle indices."""
    n = len(weights)
    positions = (np.random.rand() + np.arange(n)) / n
    cumulative_sum = np.cumsum(weights)
    indexes = np.zeros(n, dtype=int)
    i, j = 0, 0
    while i < n:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def particle_filter_beta(
    nasdaq_returns: pd.Series,
    tqqq_returns: pd.Series,
    n_particles: int = 1500,
    process_std: np.ndarray = np.array([0.0005, 0.03]),
    obs_std: float = 0.02,
) -> Dict[str, Any]:
    """
    Bootstrap particle filter for dynamic alpha/beta.

    State transition: x_t = x_{t-1} + w_t,  w_t ~ N(0, Q)
    Observation: r_tqqq = alpha_t + beta_t * r_nasdaq + v_t
    """
    nasdaq_ret, tqqq_ret = align_data(nasdaq_returns, tqqq_returns)
    idx = nasdaq_ret.index

    particles = np.zeros((n_particles, 2))
    particles[:, 1] = np.random.normal(3.0, 0.1, size=n_particles)  # beta prior
    weights = np.ones(n_particles) / n_particles

    alpha_est, beta_est = [], []
    alpha_ci, beta_ci = [], []

    for i, (r_nasdaq, r_tqqq) in enumerate(zip(nasdaq_ret.values, tqqq_ret.values)):
        # Propagate particles with process noise
        particles[:, 0] += np.random.normal(0.0, process_std[0], size=n_particles)
        particles[:, 1] += np.random.normal(0.0, process_std[1], size=n_particles)

        # Compute likelihood
        predicted = particles[:, 0] + particles[:, 1] * r_nasdaq
        residual = r_tqqq - predicted
        likelihood = (
            (1.0 / (np.sqrt(2 * np.pi) * obs_std))
            * np.exp(-0.5 * (residual / obs_std) ** 2)
        )
        weights *= likelihood + 1e-300
        weights /= np.sum(weights)

        # Resample if degeneracy
        neff = 1.0 / np.sum(weights**2)
        if neff < n_particles / 2:
            indexes = _systematic_resample(weights)
            particles = particles[indexes]
            weights.fill(1.0 / n_particles)

        alpha_est.append(np.average(particles[:, 0], weights=weights))
        beta_est.append(np.average(particles[:, 1], weights=weights))
        alpha_ci.append(
            (
                np.percentile(particles[:, 0], 5),
                np.percentile(particles[:, 0], 95),
            )
        )
        beta_ci.append(
            (
                np.percentile(particles[:, 1], 5),
                np.percentile(particles[:, 1], 95),
            )
        )

    alpha_series = pd.Series(alpha_est, index=idx, name="alpha_pf")
    beta_series = pd.Series(beta_est, index=idx, name="beta_pf")
    alpha_bounds = pd.DataFrame(alpha_ci, index=idx, columns=["alpha_p5", "alpha_p95"])
    beta_bounds = pd.DataFrame(beta_ci, index=idx, columns=["beta_p5", "beta_p95"])

    predicted_returns = alpha_series + beta_series * nasdaq_ret
    residual = tqqq_ret - predicted_returns

    return {
        "alpha": alpha_series,
        "beta": beta_series,
        "alpha_bounds": alpha_bounds,
        "beta_bounds": beta_bounds,
        "predicted_returns": predicted_returns,
        "actual_returns": tqqq_ret,
        "nasdaq_returns": nasdaq_ret,
        "residual": residual,
    }


def visualize_particle_results(results: Dict[str, Any]) -> None:
    beta = results["beta"]
    beta_bounds = results["beta_bounds"]
    alpha = results["alpha"]
    alpha_bounds = results["alpha_bounds"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    axes[0].plot(beta.index, beta.values, color="tab:green", label="Particle Filter Beta")
    axes[0].fill_between(
        beta.index,
        beta_bounds["beta_p5"].values,
        beta_bounds["beta_p95"].values,
        color="tab:green",
        alpha=0.15,
        label="5/95 Percentile",
    )
    axes[0].axhline(3.0, color="tab:red", linestyle="--", label="Theory 3x")
    axes[0].set_title("Particle Filter: Dynamic Beta")
    axes[0].set_ylabel("Beta")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(alpha.index, alpha.values, color="tab:purple", label="Particle Filter Alpha")
    axes[1].fill_between(
        alpha.index,
        alpha_bounds["alpha_p5"].values,
        alpha_bounds["alpha_p95"].values,
        color="tab:purple",
        alpha=0.12,
        label="5/95 Percentile",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Particle Filter: Alpha")
    axes[1].set_ylabel("Alpha")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        results["actual_returns"].index,
        results["actual_returns"].values,
        label="Actual TQQQ",
        alpha=0.6,
    )
    axes[2].plot(
        results["predicted_returns"].index,
        results["predicted_returns"].values,
        label="PF Predicted",
        linestyle="--",
        alpha=0.8,
    )
    axes[2].set_title("Actual vs Particle-Filter Predicted Returns")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Return")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nResidual Statistics")
    print("-" * 60)
    print(results["residual"].describe())


def main():
    data = load_nasdaq_tqqq_data(start_date="2020-01-01")
    results = particle_filter_beta(
        data["nasdaq"]["Returns"],
        data["tqqq"]["Returns"],
        n_particles=2000,
        process_std=np.array([0.0007, 0.04]),
        obs_std=0.025,
    )
    visualize_particle_results(results)


if __name__ == "__main__":
    main()


"""
Chapter 4: Bayesian Statistics & Filtering - Understanding Kalman Filter

Core Analogy: "Navigation"
- Combine GPS signal (observation) and car speed (model) to estimate 'true position'
- Bayesian update: Prior probability → Observation → Posterior probability

This example demonstrates:
1. Principles of Bayesian update
2. Mathematical structure of Kalman Filter
3. Dynamic beta estimation (financial application)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from filterpy.kalman import KalmanFilter
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def demonstrate_bayesian_update():
    """
    1. Basic Principles of Bayesian Update
    """
    print("=" * 60)
    print("1. Bayesian Update")
    print("=" * 60)
    
    # Example: Coin flip
    # Prior probability: Probability that coin is fair = 0.5
    prior_prob_fair = 0.5
    prior_prob_unfair = 0.5
    
    # Observation: 8 heads out of 10 tosses
    n_tosses = 10
    n_heads = 8
    
    # Likelihood: Probability of 8 heads from a fair coin
    from scipy.stats import binom
    likelihood_fair = binom.pmf(n_heads, n_tosses, 0.5)
    likelihood_unfair = binom.pmf(n_heads, n_tosses, 0.7)  # Unfair coin (70% heads)
    
    # Posterior probability: Bayes' theorem
    # P(fair|data) = P(data|fair) × P(fair) / P(data)
    posterior_prob_fair = (likelihood_fair * prior_prob_fair) / \
                          (likelihood_fair * prior_prob_fair + likelihood_unfair * prior_prob_unfair)
    posterior_prob_unfair = 1 - posterior_prob_fair
    
    print(f"\nCoin flip example:")
    print(f"  Observation: {n_heads} heads out of {n_tosses} tosses")
    print(f"\nPrior probability:")
    print(f"  Fair coin: {prior_prob_fair:.4f}")
    print(f"  Unfair coin: {prior_prob_unfair:.4f}")
    print(f"\nLikelihood:")
    print(f"  This result from fair coin: {likelihood_fair:.6f}")
    print(f"  This result from unfair coin: {likelihood_unfair:.6f}")
    print(f"\nPosterior probability (Bayes' theorem):")
    print(f"  Fair coin: {posterior_prob_fair:.4f}")
    print(f"  Unfair coin: {posterior_prob_unfair:.4f}")
    print(f"\n  → After seeing data, probability of unfair coin increased!")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Fair', 'Unfair']
    prior = [prior_prob_fair, prior_prob_unfair]
    posterior = [posterior_prob_fair, posterior_prob_unfair]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, prior, width, label='Prior Probability', alpha=0.7, color='blue')
    ax.bar(x + width/2, posterior, width, label='Posterior Probability', alpha=0.7, color='red')
    
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Bayesian Update: Prior → Posterior', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'bayesian_update.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def simple_kalman_filter_demo():
    """
    2. Simple Kalman Filter Example
    """
    print("\n" + "=" * 60)
    print("2. Kalman Filter Basic Principles")
    print("=" * 60)
    
    # Simulation: True position and noisy observations
    np.random.seed(42)
    n_steps = 50
    true_position = np.linspace(0, 10, n_steps)  # True position (linear movement)
    observation_noise = np.random.randn(n_steps) * 0.5  # Observation noise
    observations = true_position + observation_noise  # Observations
    
    # Kalman filter setup
    kf = KalmanFilter(dim_x=2, dim_z=1)  # State: [position, velocity], Observation: [position]
    
    # Initial state
    kf.x = np.array([[0.], [1.]])  # Initial position=0, velocity=1
    
    # State transition matrix (F): Update position and velocity
    kf.F = np.array([[1., 1.],  # position = position + velocity
                     [0., 1.]])  # velocity = velocity
    
    # Observation matrix (H): Only observe position
    kf.H = np.array([[1., 0.]])
    
    # Covariance matrices
    kf.P = np.eye(2) * 1000  # Initial uncertainty (large value)
    kf.R = np.array([[0.5]])  # Observation noise
    kf.Q = np.eye(2) * 0.1  # Process noise
    
    # Filtering
    estimated_positions = []
    estimated_velocities = []
    
    for obs in observations:
        kf.predict()  # Prediction step
        kf.update(obs)  # Update step
        estimated_positions.append(kf.x[0, 0])
        estimated_velocities.append(kf.x[1, 0])
    
    estimated_positions = np.array(estimated_positions)
    estimated_velocities = np.array(estimated_velocities)
    
    print(f"\nKalman Filter Results:")
    print(f"  True final position: {true_position[-1]:.4f}")
    print(f"  Observed final position: {observations[-1]:.4f}")
    print(f"  Estimated final position: {estimated_positions[-1]:.4f}")
    print(f"  Estimated velocity: {estimated_velocities[-1]:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Position comparison
    axes[0].plot(true_position, 'g-', linewidth=2, label='True Position', alpha=0.7)
    axes[0].scatter(range(n_steps), observations, color='red', s=20, alpha=0.5, label='Observations (with noise)')
    axes[0].plot(estimated_positions, 'b-', linewidth=2, label='Kalman Filter Estimate')
    axes[0].set_xlabel('Time', fontsize=12)
    axes[0].set_ylabel('Position', fontsize=12)
    axes[0].set_title('Kalman Filter: Position Estimation', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Velocity estimation
    true_velocity = np.ones(n_steps)  # True velocity = 1
    axes[1].plot(true_velocity, 'g-', linewidth=2, label='True Velocity', alpha=0.7)
    axes[1].plot(estimated_velocities, 'b-', linewidth=2, label='Kalman Filter Estimated Velocity')
    axes[1].set_xlabel('Time', fontsize=12)
    axes[1].set_ylabel('Velocity', fontsize=12)
    axes[1].set_title('Kalman Filter: Velocity Estimation', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'kalman_filter_demo.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return estimated_positions, estimated_velocities


def kalman_filter_beta_estimation(ticker='TQQQ', benchmark='^IXIC', 
                                   start_date='2020-01-01', end_date='2024-01-01'):
    """
    3. Dynamic Beta Estimation using Kalman Filter (Financial Application)
    """
    print("\n" + "=" * 60)
    print("3. Dynamic Beta Estimation using Kalman Filter")
    print("=" * 60)
    
    # Download data
    print(f"\nDownloading {ticker} and {benchmark} data...")
    data = yf.download([ticker, benchmark], start=start_date, end=end_date)['Close']
    data = data.dropna()
    
    returns_tqqq = data[ticker].pct_change().dropna()
    returns_nasdaq = data[benchmark].pct_change().dropna()
    
    # Align data
    common_dates = returns_tqqq.index.intersection(returns_nasdaq.index)
    returns_tqqq = returns_tqqq.loc[common_dates]
    returns_nasdaq = returns_nasdaq.loc[common_dates]
    
    print(f"Data period: {returns_tqqq.index[0].date()} ~ {returns_tqqq.index[-1].date()}")
    print(f"Number of observations: {len(returns_tqqq)}")
    
    # Kalman filter setup
    # State: [alpha, beta]
    # Observation: TQQQ returns
    # Model: r_tqqq = alpha + beta * r_nasdaq + noise
    kf = KalmanFilter(dim_x=2, dim_z=1)
    
    # Initial state
    kf.x = np.array([[0.], [3.]])  # Initial alpha=0, beta=3 (TQQQ is 3x leveraged)
    
    # State transition: Assume beta follows random walk
    kf.F = np.eye(2)
    
    # Observation matrix: r_tqqq = alpha + beta * r_nasdaq
    # H is dynamically updated (depends on r_nasdaq)
    kf.H = np.array([[1., 0.]])  # Initial value (will be updated)
    
    # Covariance
    kf.P = np.eye(2) * 10
    kf.R = np.array([[0.01]])  # Observation noise
    kf.Q = np.eye(2) * 0.001  # Process noise
    
    # Filtering
    estimated_betas = []
    estimated_alphas = []
    
    for i in range(len(returns_nasdaq)):
        r_nasdaq = returns_nasdaq.iloc[i]
        r_tqqq = returns_tqqq.iloc[i]
        
        # Update observation matrix: H = [1, r_nasdaq]
        kf.H = np.array([[1., r_nasdaq]])
        
        # Predict and update
        kf.predict()
        kf.update(r_tqqq)
        
        estimated_alphas.append(kf.x[0, 0])
        estimated_betas.append(kf.x[1, 0])
    
    estimated_betas = np.array(estimated_betas)
    estimated_alphas = np.array(estimated_alphas)
    
    # Rolling beta (for comparison)
    window = 60
    rolling_betas = []
    for i in range(window, len(returns_nasdaq)):
        y = returns_tqqq.iloc[i-window:i]
        x = returns_nasdaq.iloc[i-window:i]
        beta_ols = np.cov(y, x)[0, 1] / np.var(x)
        rolling_betas.append(beta_ols)
    
    rolling_betas = np.array(rolling_betas)
    rolling_dates = returns_nasdaq.index[window:]
    
    print(f"\nBeta Estimation Results:")
    print(f"  Initial beta: {estimated_betas[0]:.4f}")
    print(f"  Final beta: {estimated_betas[-1]:.4f}")
    print(f"  Mean beta: {estimated_betas.mean():.4f}")
    print(f"  Beta std dev: {estimated_betas.std():.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Beta estimation
    axes[0].plot(returns_nasdaq.index, estimated_betas, 'b-', linewidth=2, 
                label='Kalman Filter Beta', alpha=0.8)
    axes[0].plot(rolling_dates, rolling_betas, 'r--', linewidth=2, 
                label=f'{window}-day Rolling Beta', alpha=0.7)
    axes[0].axhline(y=3.0, color='g', linestyle='--', alpha=0.7, label='Theoretical Beta (3.0)')
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Beta', fontsize=12)
    axes[0].set_title(f'{ticker} Dynamic Beta Estimation (Kalman Filter)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Alpha estimation
    axes[1].plot(returns_nasdaq.index, estimated_alphas, 'purple', linewidth=2, 
                label='Kalman Filter Alpha', alpha=0.8)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Alpha', fontsize=12)
    axes[1].set_title(f'{ticker} Dynamic Alpha Estimation', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f'kalman_beta_{ticker}.png')
    plt.savefig(output_path, 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return estimated_betas, estimated_alphas


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("Chapter 4: Bayesian Statistics & Filtering - Understanding Kalman Filter")
    print("=" * 60)
    
    # 1. Bayesian update
    demonstrate_bayesian_update()
    
    # 2. Simple Kalman Filter
    estimated_pos, estimated_vel = simple_kalman_filter_demo()
    
    # 3. Financial application: Dynamic beta estimation
    betas, alphas = kalman_filter_beta_estimation('TQQQ', '^IXIC')
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Bayesian update: Prior probability → Observation → Posterior probability")
    print("2. Kalman Filter: Iterate prediction step + update step")
    print("3. Prediction: Predict next state using model")
    print("4. Update: Correct prediction with observation")
    print("5. Financial application: Dynamic beta/alpha estimation, state-space models")


if __name__ == "__main__":
    main()


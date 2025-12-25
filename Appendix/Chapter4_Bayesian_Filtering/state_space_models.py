"""
Chapter 4: Bayesian Statistics & Filtering - State Space Models

Core Analogy: "Inferring hidden state from observations"
- State equation: Evolution of hidden state (e.g., true market beta)
- Observation equation: Observable data (e.g., stock returns)
- Kalman Filter: Combine state and observation for optimal estimation

This example demonstrates:
1. Understanding the structure of state space models
2. Dynamic beta estimation
3. Tracking relationship changes over time
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


def explain_state_space():
    """
    1. Explain State Space Model Structure
    """
    print("=" * 60)
    print("1. State Space Model")
    print("=" * 60)
    
    print("\n[State Space Model Structure]")
    print("  State equation: x_t = F × x_{t-1} + w_t")
    print("  Observation equation: y_t = H × x_t + v_t")
    print("\n  Where:")
    print("    x_t: Hidden state (e.g., true beta)")
    print("    y_t: Observable data (e.g., stock returns)")
    print("    F: State transition matrix")
    print("    H: Observation matrix")
    print("    w_t: Process noise")
    print("    v_t: Observation noise")
    
    print("\n[Financial Application Example]")
    print("  State: True market beta (varies over time)")
    print("  Observation: Stock price and market returns")
    print("  Goal: Track changes in beta")
    
    print("\n[Role of Kalman Filter]")
    print("  1. Prediction: Predict next state using model")
    print("  2. Update: Correct prediction with observation")
    print("  3. Iterate: Track state over time")


def dynamic_beta_estimation(ticker='TQQQ', benchmark='^IXIC', 
                            start_date='2020-01-01', end_date='2024-01-01'):
    """
    2. Dynamic Beta Estimation (State Space Model)
    """
    print("\n" + "=" * 60)
    print("2. Dynamic Beta Estimation (State Space Model)")
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
    
    # State space model setup
    # State: [alpha, beta]
    # Observation: TQQQ returns
    # Model: r_tqqq = alpha + beta * r_nasdaq + noise
    
    kf = KalmanFilter(dim_x=2, dim_z=1)
    
    # Initial state
    kf.x = np.array([[0.], [3.]])  # Initial alpha=0, beta=3
    
    # State transition: beta follows random walk
    kf.F = np.eye(2)
    
    # Observation matrix (dynamically updated)
    kf.H = np.array([[1., 0.]])
    
    # Covariance
    kf.P = np.eye(2) * 10  # Initial uncertainty
    kf.R = np.array([[0.01]])  # Observation noise
    kf.Q = np.eye(2) * 0.001  # Process noise
    
    # Filtering
    estimated_betas = []
    estimated_alphas = []
    beta_uncertainty = []
    
    for i in range(len(returns_nasdaq)):
        r_nasdaq = returns_nasdaq.iloc[i]
        r_tqqq = returns_tqqq.iloc[i]
        
        # Update observation matrix
        kf.H = np.array([[1., r_nasdaq]])
        
        # Predict and update
        kf.predict()
        kf.update(r_tqqq)
        
        estimated_alphas.append(kf.x[0, 0])
        estimated_betas.append(kf.x[1, 0])
        beta_uncertainty.append(np.sqrt(kf.P[1, 1]))  # Beta uncertainty
    
    estimated_betas = np.array(estimated_betas)
    estimated_alphas = np.array(estimated_alphas)
    beta_uncertainty = np.array(beta_uncertainty)
    
    # Rolling beta (for comparison)
    window = 60
    rolling_betas = []
    rolling_dates = []
    for i in range(window, len(returns_nasdaq)):
        y = returns_tqqq.iloc[i-window:i]
        x = returns_nasdaq.iloc[i-window:i]
        beta_ols = np.cov(y, x)[0, 1] / np.var(x)
        rolling_betas.append(beta_ols)
        rolling_dates.append(returns_nasdaq.index[i])
    
    rolling_betas = np.array(rolling_betas)
    
    print(f"\n[Beta Estimation Results]")
    print(f"  Initial beta: {estimated_betas[0]:.4f}")
    print(f"  Final beta: {estimated_betas[-1]:.4f}")
    print(f"  Mean beta: {estimated_betas.mean():.4f}")
    print(f"  Beta std dev: {estimated_betas.std():.4f}")
    print(f"  Mean uncertainty: {beta_uncertainty.mean():.4f}")
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Beta estimation
    axes[0].plot(returns_nasdaq.index, estimated_betas, 'b-', linewidth=2, 
                label='Kalman Filter Beta', alpha=0.8)
    axes[0].fill_between(returns_nasdaq.index, 
                        estimated_betas - 2*beta_uncertainty,
                        estimated_betas + 2*beta_uncertainty,
                        alpha=0.2, color='blue', label='95% Confidence Interval')
    axes[0].plot(rolling_dates, rolling_betas, 'r--', linewidth=2, 
                label=f'{window}-day Rolling Beta', alpha=0.7)
    axes[0].axhline(y=3.0, color='g', linestyle='--', alpha=0.7, label='Theoretical Beta (3.0)')
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Beta', fontsize=12)
    axes[0].set_title(f'{ticker} Dynamic Beta Estimation (State Space Model)', 
                     fontsize=14, fontweight='bold')
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
    
    # Beta uncertainty
    axes[2].plot(returns_nasdaq.index, beta_uncertainty, 'orange', linewidth=2, 
                label='Beta Uncertainty (Std Dev)', alpha=0.8)
    axes[2].set_xlabel('Date', fontsize=12)
    axes[2].set_ylabel('Uncertainty', fontsize=12)
    axes[2].set_title('Beta Estimation Uncertainty', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f'state_space_beta_{ticker}.png')
    plt.savefig(output_path, 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return estimated_betas, estimated_alphas, beta_uncertainty


def time_varying_relationship():
    """
    3. Track Relationship Changes Over Time
    """
    print("\n" + "=" * 60)
    print("3. Track Relationship Changes Over Time")
    print("=" * 60)
    
    print("\n[Advantages of State Space Models]")
    print("  1. Track relationships that change over time")
    print("  2. Quantify uncertainty")
    print("  3. Real-time updates possible")
    print("  4. Detect structural changes")
    
    print("\n[Financial Application Examples]")
    print("  - Temporal changes in beta (leveraged ETFs)")
    print("  - Changes in correlation (by market regime)")
    print("  - Dynamic volatility estimation")
    print("  - Risk factor exposure tracking")


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("Chapter 4: Bayesian Statistics & Filtering - State Space Models")
    print("=" * 60)
    
    # 1. Explain state space models
    explain_state_space()
    
    # 2. Dynamic beta estimation
    betas, alphas, uncertainty = dynamic_beta_estimation('TQQQ', '^IXIC')
    
    # 3. Relationship changes over time
    time_varying_relationship()
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. State space models: Infer hidden state from observations")
    print("2. Kalman Filter: Iterate prediction + update")
    print("3. Can track relationships that change over time")
    print("4. Quantify uncertainty (confidence intervals)")
    print("5. Foundation of State-Space Models, Kalman Filter")


if __name__ == "__main__":
    main()


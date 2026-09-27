"""
Chapter 2: Calculus & Analysis - Bayesian Optimization

Core Analogy: "Finding optimal value through intelligent search"
- Grid Search: Try all points (inefficient)
- Random Search: Try randomly (inefficient)
- Bayesian Optimization: Select next trial point considering uncertainty
- Acquisition Function: Balance between Exploration vs Exploitation

This example demonstrates:
1. Principles of Bayesian Optimization
2. Function estimation using Gaussian Process
3. Acquisition Function (EI, UCB)
4. Application to hyperparameter tuning
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
try:
    from skopt import gp_minimize
    from skopt.acquisition import gaussian_ei
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    print("Warning: scikit-optimize not available. Install with: pip install scikit-optimize")

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def explain_bayesian_optimization():
    """
    1. Principles of Bayesian Optimization
    """
    print("=" * 60)
    print("1. Principles of Bayesian Optimization")
    print("=" * 60)
    
    print("\n[Problem Setup]")
    print("  Goal: x* = argmin f(x)  (or argmax)")
    print("  Constraint: Evaluating f(x) is expensive")
    print("  → Find optimal value with as few evaluations as possible")
    
    print("\n[Bayesian Optimization Process]")
    print("  1. Initial observation: Evaluate f(x) at a few points")
    print("  2. Function estimation: Estimate distribution of f(x) using Gaussian Process")
    print("  3. Next point selection: Select optimal next point using Acquisition Function")
    print("  4. Evaluation and update: Evaluate at selected point and update GP")
    print("  5. Iterate: Repeat until optimal value converges")
    
    print("\n[Acquisition Function]")
    print("  - Expected Improvement (EI): Expected improvement")
    print("  - Upper Confidence Bound (UCB): Consider uncertainty")
    print("  - Balance between exploration and exploitation")
    
    print("\n[Financial Applications]")
    print("  - Hyperparameter tuning: Optimize model training parameters")
    print("  - Strategy parameter optimization: Find trading strategy parameters")
    print("  - Portfolio optimization: Optimize under constraints")


def simple_bayesian_optimization():
    """
    2. Simple Bayesian Optimization Example
    """
    print("\n" + "=" * 60)
    print("2. Bayesian Optimization Example")
    print("=" * 60)
    
    # Objective function: f(x) = (x - 2)² + 0.1*sin(10*x) + noise
    # Note: scikit-optimize passes x as a list, so we extract the first element
    def objective_function(x):
        x_val = x[0] if isinstance(x, (list, np.ndarray)) else x
        return (x_val - 2)**2 + 0.1 * np.sin(10 * x_val) + np.random.randn() * 0.1
    
    # True optimal value (approximately x = 2)
    x_true_opt = 2.0
    
    if not HAS_SKOPT:
        print("\n⚠ scikit-optimize package is required.")
        print("Performing simple simulation.")
        
        # Simple simulation
        n_iterations = 20
        x_bounds = [(-5.0, 5.0)]
        
        # Initial points
        x_observed = np.random.uniform(-5, 5, 5)
        y_observed = [objective_function(x) for x in x_observed]
        
        # Simple search (actually uses GP)
        for i in range(n_iterations - 5):
            # Search in high uncertainty regions (simplified)
            x_candidates = np.random.uniform(-5, 5, 10)
            y_candidates = [objective_function(x) for x in x_candidates]
            
            # Select best point
            best_idx = np.argmin(y_candidates)
            x_observed = np.append(x_observed, x_candidates[best_idx])
            y_observed.append(y_candidates[best_idx])
        
        best_x = x_observed[np.argmin(y_observed)]
        best_y = min(y_observed)
        
        print(f"\n[Optimization Results]")
        print(f"  Optimal x: {best_x:.4f}")
        print(f"  Optimal y: {best_y:.4f}")
        print(f"  True optimal x: {x_true_opt:.4f}")
        print(f"  Error: {abs(best_x - x_true_opt):.4f}")
        
        # Visualization
        x_range = np.linspace(-5, 5, 1000)
        y_range = [(x - 2)**2 + 0.1 * np.sin(10 * x) for x in x_range]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(x_range, y_range, 'b-', linewidth=2, label='Objective Function', alpha=0.7)
        ax.scatter(x_observed, y_observed, color='red', s=50, alpha=0.6, label='Observed Points')
        ax.scatter([best_x], [best_y], color='green', s=200, marker='*', 
                  zorder=5, label=f'Optimal Point ({best_x:.2f})')
        ax.axvline(x=x_true_opt, color='orange', linestyle='--', 
                  linewidth=2, label=f'True Optimal Point ({x_true_opt})')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('f(x)', fontsize=12)
        ax.set_title('Bayesian Optimization Results', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, 'bayesian_optimization_simple.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return best_x, best_y
    
    # Using scikit-optimize
    result = gp_minimize(
        objective_function,
        dimensions=[(-5.0, 5.0)],
        n_calls=20,
        random_state=42,
        acq_func='EI'  # Expected Improvement
    )
    
    print(f"\n[Optimization Results]")
    print(f"  Optimal x: {result.x[0]:.4f}")
    print(f"  Optimal y: {result.fun:.4f}")
    print(f"  True optimal x: {x_true_opt:.4f}")
    print(f"  Error: {abs(result.x[0] - x_true_opt):.4f}")
    print(f"  Number of evaluations: {len(result.func_vals)}")
    
    # Visualization
    x_range = np.linspace(-5, 5, 1000)
    y_range = [(x - 2)**2 + 0.1 * np.sin(10 * x) for x in x_range]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Objective function and observed points
    axes[0].plot(x_range, y_range, 'b-', linewidth=2, label='Objective Function', alpha=0.7)
    axes[0].scatter(result.x_iters, result.func_vals, color='red', s=50, 
                   alpha=0.6, label='Observed Points')
    axes[0].scatter([result.x[0]], [result.fun], color='green', s=200, 
                   marker='*', zorder=5, label=f'Optimal Point ({result.x[0]:.2f})')
    axes[0].axvline(x=x_true_opt, color='orange', linestyle='--', 
                   linewidth=2, label=f'True Optimal Point ({x_true_opt})')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('f(x)', fontsize=12)
    axes[0].set_title('Bayesian Optimization Results', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Convergence curve
    best_so_far = np.minimum.accumulate(result.func_vals)
    axes[1].plot(range(1, len(best_so_far) + 1), best_so_far, 'o-', 
                linewidth=2, markersize=6)
    axes[1].set_xlabel('Iteration Number', fontsize=12)
    axes[1].set_ylabel('Best Value (So Far)', fontsize=12)
    axes[1].set_title('Bayesian Optimization Convergence Curve', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'bayesian_optimization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return result.x[0], result.fun


def hyperparameter_tuning_example():
    """
    3. Hyperparameter Tuning Example
    """
    print("\n" + "=" * 60)
    print("3. Hyperparameter Tuning Example")
    print("=" * 60)
    
    print("\n[Hyperparameter Tuning]")
    print("  Goal: Find hyperparameters that maximize model performance")
    print("  Examples:")
    print("    - Learning rate")
    print("    - Regularization parameters")
    print("    - Network structure (hidden units, layers)")
    
    print("\n[Advantages of Bayesian Optimization]")
    print("  1. Find optimal value with few evaluations")
    print("  2. Consider uncertainty")
    print("  3. Balance exploration vs exploitation")
    print("  4. Suitable for expensive evaluations")
    
    print("\n[Financial Applications]")
    print("  - LSTM hyperparameters: hidden units, learning rate")
    print("  - XGBoost parameters: max_depth, learning_rate")
    print("  - Strategy parameters: moving average period, threshold")
    print("  - Portfolio optimization: risk penalty coefficient")


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("Chapter 2: Calculus & Analysis - Bayesian Optimization")
    print("=" * 60)
    
    # 1. Explain Bayesian Optimization principles
    explain_bayesian_optimization()
    
    # 2. Simple example
    best_x, best_y = simple_bayesian_optimization()
    
    # 3. Hyperparameter tuning example
    hyperparameter_tuning_example()
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Bayesian Optimization: Intelligent search considering uncertainty")
    print("2. Gaussian Process: Estimate distribution of function")
    print("3. Acquisition Function: Select next trial point")
    print("4. Find optimal value with few evaluations (cost-effective)")
    print("5. Useful for hyperparameter tuning")


if __name__ == "__main__":
    main()


"""
Chapter 2: Calculus & Analysis - Gradient Descent Visualization

Core Analogy: "Descending a mountain in fog"
- Derivative: Speedometer (measuring rate of change)
- Gradient Descent: Feeling the slope to descend (finding minimum)
- Learning Rate: Step size per iteration

This example demonstrates:
1. Understanding the concept of derivative (rate of change)
2. Principles of gradient descent
3. Impact of learning rate
4. Application to financial optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def demonstrate_derivative():
    """
    1. Basic concept of derivative
    """
    print("=" * 60)
    print("1. Basic Concept of Derivative")
    print("=" * 60)
    
    # Simple function: f(x) = x²
    def f(x):
        return x**2
    
    # Derivative: f'(x) = 2x
    def df(x):
        return 2*x
    
    # Numerical derivative (approximation)
    def numerical_derivative(f, x, h=1e-5):
        return (f(x + h) - f(x)) / h
    
    x = np.linspace(-3, 3, 100)
    y = f(x)
    dy_analytical = df(x)
    dy_numerical = numerical_derivative(f, np.linspace(-3, 3, 100))
    
    print("\nFunction: f(x) = x²")
    print("Derivative: f'(x) = 2x")
    print(f"\nWhen x = 1:")
    print(f"  Function value: f(1) = {f(1):.4f}")
    print(f"  Derivative (analytical): f'(1) = {df(1):.4f}")
    print(f"  Derivative (numerical): {numerical_derivative(f, 1):.4f}")
    print("  → If derivative is negative, function decreases; if positive, increases")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b-', linewidth=2, label='f(x) = x²')
    ax.plot(x, dy_analytical, 'r--', linewidth=2, label="f'(x) = 2x")
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.scatter([1], [f(1)], color='green', s=100, zorder=5)
    ax.annotate(f'x=1\nf(1)={f(1):.2f}\nf\'(1)={df(1):.2f}', 
                xy=(1, f(1)), xytext=(1.5, 3),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Concept of Derivative: f(x) = x²', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'derivative_concept.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return f, df


def gradient_descent_1d(f, df, x0, learning_rate=0.1, n_iterations=50):
    """
    2. 1D Gradient Descent
    """
    print("\n" + "=" * 60)
    print("2. Gradient Descent (1D)")
    print("=" * 60)
    
    x = x0
    history = [x]
    
    print(f"\nInitial value: x₀ = {x0:.4f}")
    print(f"Learning rate: α = {learning_rate:.4f}")
    print(f"Number of iterations: {n_iterations}")
    
    for i in range(n_iterations):
        # Gradient descent update: x_new = x_old - α × f'(x_old)
        gradient = df(x)
        x = x - learning_rate * gradient
        history.append(x)
        
        if i < 5 or i % 10 == 0:
            print(f"  Iteration {i+1}: x = {x:.6f}, f(x) = {f(x):.6f}, f'(x) = {gradient:.6f}")
    
    print(f"\nFinal result: x* = {x:.6f}, f(x*) = {f(x):.6f}")
    print(f"Theoretical minimum: x = 0, f(0) = 0")
    
    return np.array(history)


def visualize_gradient_descent_1d(f, df, history, learning_rate):
    """
    3. 1D Gradient Descent Visualization
    """
    x_range = np.linspace(-3, 3, 100)
    y_range = f(x_range)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Function graph
    ax.plot(x_range, y_range, 'b-', linewidth=2, label='f(x) = x²', zorder=1)
    
    # Gradient descent path
    for i in range(len(history) - 1):
        x_curr = history[i]
        x_next = history[i + 1]
        y_curr = f(x_curr)
        y_next = f(x_next)
        
        # Arrow
        ax.annotate('', xy=(x_next, y_next), xytext=(x_curr, y_curr),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.6))
        
        # Points
        if i == 0:
            ax.scatter(x_curr, y_curr, color='green', s=150, zorder=5, 
                      label='Start point', marker='o')
        elif i == len(history) - 2:
            ax.scatter(x_next, y_next, color='red', s=150, zorder=5, 
                      label='End point', marker='*')
        else:
            ax.scatter(x_curr, y_curr, color='orange', s=50, zorder=4, alpha=0.6)
    
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title(f'Gradient Descent Visualization (Learning Rate = {learning_rate})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f'gradient_descent_1d_lr{learning_rate}.png')
    plt.savefig(output_path, 
                dpi=300, bbox_inches='tight')
    plt.show()


def gradient_descent_2d(f, df, x0, learning_rate=0.1, n_iterations=50):
    """
    4. 2D Gradient Descent (Portfolio Optimization Example)
    """
    print("\n" + "=" * 60)
    print("4. Gradient Descent (2D) - Portfolio Optimization Example")
    print("=" * 60)
    
    # Example: Minimize portfolio variance
    # Objective function: f(w1, w2) = w1²σ1² + w2²σ2² + 2w1w2σ12
    # Constraint: w1 + w2 = 1 → w2 = 1 - w1
    # Therefore: f(w1) = w1²σ1² + (1-w1)²σ2² + 2w1(1-w1)σ12
    
    sigma1 = 0.2  # Volatility of asset 1
    sigma2 = 0.15  # Volatility of asset 2
    sigma12 = 0.05  # Covariance
    
    def portfolio_variance(w1):
        w2 = 1 - w1
        return w1**2 * sigma1**2 + w2**2 * sigma2**2 + 2 * w1 * w2 * sigma12
    
    def d_portfolio_variance(w1):
        w2 = 1 - w1
        return 2 * w1 * sigma1**2 - 2 * w2 * sigma2**2 + 2 * (1 - 2*w1) * sigma12
    
    x = x0
    history = [x]
    
    print(f"\nInitial weights: w1 = {x0:.4f}, w2 = {1-x0:.4f}")
    print(f"Initial portfolio variance: {portfolio_variance(x0):.6f}")
    
    for i in range(n_iterations):
        gradient = d_portfolio_variance(x)
        x = x - learning_rate * gradient
        
        # Constraint: 0 <= w1 <= 1
        x = np.clip(x, 0, 1)
        history.append(x)
        
        if i < 5 or i % 10 == 0:
            print(f"  Iteration {i+1}: w1 = {x:.4f}, w2 = {1-x:.4f}, "
                  f"variance = {portfolio_variance(x):.6f}")
    
    print(f"\nOptimal weights: w1* = {x:.4f}, w2* = {1-x:.4f}")
    print(f"Minimum variance: {portfolio_variance(x):.6f}")
    
    # Visualization
    w1_range = np.linspace(0, 1, 100)
    var_range = [portfolio_variance(w) for w in w1_range]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(w1_range, var_range, 'b-', linewidth=2, label='Portfolio Variance')
    
    # Gradient descent path
    for i in range(len(history) - 1):
        w1_curr = history[i]
        w1_next = history[i + 1]
        var_curr = portfolio_variance(w1_curr)
        var_next = portfolio_variance(w1_next)
        
        if i == 0:
            ax.scatter(w1_curr, var_curr, color='green', s=150, zorder=5, 
                      label='Start point', marker='o')
        elif i == len(history) - 2:
            ax.scatter(w1_next, var_next, color='red', s=150, zorder=5, 
                      label='Optimal point', marker='*')
        else:
            ax.scatter(w1_curr, var_curr, color='orange', s=50, zorder=4, alpha=0.6)
    
    ax.set_xlabel('w1 (Asset 1 Weight)', fontsize=12)
    ax.set_ylabel('Portfolio Variance', fontsize=12)
    ax.set_title('Portfolio Optimization with Gradient Descent', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'gradient_descent_portfolio.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return np.array(history)


def compare_learning_rates():
    """
    5. Learning Rate Comparison
    """
    print("\n" + "=" * 60)
    print("5. Impact of Learning Rate")
    print("=" * 60)
    
    def f(x):
        return x**2
    
    def df(x):
        return 2*x
    
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    x0 = 2.5
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, lr in enumerate(learning_rates):
        ax = axes[idx]
        history = gradient_descent_1d(f, df, x0, learning_rate=lr, n_iterations=50)
        
        x_range = np.linspace(-3, 3, 100)
        y_range = f(x_range)
        
        ax.plot(x_range, y_range, 'b-', linewidth=2, label='f(x) = x²', zorder=1)
        
        # Gradient descent path
        for i in range(len(history) - 1):
            x_curr = history[i]
            x_next = history[i + 1]
            y_curr = f(x_curr)
            y_next = f(x_next)
            
            if i == 0:
                ax.scatter(x_curr, y_curr, color='green', s=100, zorder=5, marker='o')
            elif i == len(history) - 2:
                ax.scatter(x_next, y_next, color='red', s=100, zorder=5, marker='*')
            else:
                ax.scatter(x_curr, y_curr, color='orange', s=30, zorder=4, alpha=0.5)
            
            if i < 10:  # Show arrows for first 10 steps only
                ax.annotate('', xy=(x_next, y_next), xytext=(x_curr, y_curr),
                           arrowprops=dict(arrowstyle='->', color='red', lw=1, alpha=0.5))
        
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('f(x)', fontsize=10)
        ax.set_title(f'Learning Rate = {lr}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Convergence indicator
        final_x = history[-1]
        if abs(final_x) < 0.01:
            ax.text(0.5, 0.95, '✓ Converged', transform=ax.transAxes, 
                   ha='center', va='top', fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        else:
            ax.text(0.5, 0.95, '✗ Diverged', transform=ax.transAxes, 
                   ha='center', va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'learning_rate_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nImpact of Learning Rate:")
    print("  - Too small: Slow convergence")
    print("  - Appropriate: Fast convergence")
    print("  - Too large: Divergence or oscillation")


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("Chapter 2: Calculus & Analysis - Gradient Descent Visualization")
    print("=" * 60)
    
    # 1. Basic concept of derivative
    f, df = demonstrate_derivative()
    
    # 2. 1D gradient descent
    history = gradient_descent_1d(f, df, x0=2.5, learning_rate=0.1, n_iterations=30)
    
    # 3. Visualization
    visualize_gradient_descent_1d(f, df, history, learning_rate=0.1)
    
    # 4. 2D example (portfolio optimization)
    history_2d = gradient_descent_2d(None, None, x0=0.8, learning_rate=0.05, n_iterations=30)
    
    # 5. Learning rate comparison
    compare_learning_rates()
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Derivative measures the rate of change of a function (speedometer)")
    print("2. Gradient descent: Move in the opposite direction of the derivative (gradient) to find minimum")
    print("3. Update formula: x_new = x_old - α × f'(x_old)")
    print("4. Learning rate α determines convergence speed and stability")
    print("5. Same principles apply to financial optimization (portfolio optimization, etc.)")


if __name__ == "__main__":
    main()


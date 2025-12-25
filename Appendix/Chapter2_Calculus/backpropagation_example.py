"""
Chapter 2: Calculus & Analysis - Understanding Backpropagation Algorithm

Core Analogy: "Propagating error backward to assign responsibility"
- Forward Pass: Input → Output (prediction)
- Backward Pass: Output → Input (propagate error backward to calculate each weight's responsibility)
- Chain Rule: Decompose complex function derivatives into simple function derivatives

This example demonstrates:
1. Understanding Chain Rule
2. Backpropagation in simple neural network
3. Weight update process
4. Mathematical principles of deep learning training
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def demonstrate_chain_rule():
    """
    1. Basic concept of Chain Rule
    """
    print("=" * 60)
    print("1. Chain Rule")
    print("=" * 60)
    
    # Example: f(x) = (2x + 1)²
    # Decompose into g(u) = u², u(x) = 2x + 1
    # f'(x) = g'(u) × u'(x) = 2u × 2 = 4(2x + 1)
    
    def u(x):
        return 2*x + 1
    
    def g(u):
        return u**2
    
    def f(x):
        return g(u(x))
    
    def df_chain_rule(x):
        u_val = u(x)
        dg_du = 2 * u_val  # g'(u) = 2u
        du_dx = 2  # u'(x) = 2
        return dg_du * du_dx  # Chain rule
    
    def df_direct(x):
        return 4 * (2*x + 1)  # Direct differentiation
    
    x = 2.0
    print(f"\nFunction: f(x) = (2x + 1)²")
    print(f"When x = {x}:")
    print(f"  f(x) = {f(x):.4f}")
    print(f"  f'(x) (Chain Rule): {df_chain_rule(x):.4f}")
    print(f"  f'(x) (Direct): {df_direct(x):.4f}")
    print("  → Both results are the same!")
    
    # Visualization
    x_range = np.linspace(-2, 3, 100)
    y_range = [f(x) for x in x_range]
    dy_range = [df_chain_rule(x) for x in x_range]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_range, y_range, 'b-', linewidth=2, label='f(x) = (2x + 1)²')
    ax.plot(x_range, dy_range, 'r--', linewidth=2, label="f'(x) = 4(2x + 1)")
    ax.scatter([x], [f(x)], color='green', s=100, zorder=5)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Chain Rule Example', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'chain_rule_example.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


class SimpleNeuralNetwork:
    """
    2. Simple Neural Network Class (Backpropagation Implementation)
    """
    def __init__(self, input_size=1, hidden_size=2, output_size=1):
        # Initialize weights (small random values)
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        
        # Training history
        self.loss_history = []
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Sigmoid derivative"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        """Forward pass: Input → Output"""
        # First layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Second layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2  # Output layer is linear (regression problem)
        
        return self.a2
    
    def backward(self, X, y, output, learning_rate=0.1):
        """Backward pass: Propagate error backward to update weights"""
        m = X.shape[0]  # Number of samples
        
        # Output layer error
        dz2 = output - y  # Linear activation, so derivative is 1
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer error (apply chain rule)
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.sigmoid_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Weight update (gradient descent)
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
        return dW1, db1, dW2, db2
    
    def compute_loss(self, y_true, y_pred):
        """Mean Squared Error (MSE)"""
        return np.mean((y_true - y_pred)**2)
    
    def train(self, X, y, epochs=1000, learning_rate=0.1, verbose=True):
        """Training function"""
        print("\n" + "=" * 60)
        print("2. Neural Network Training (Backpropagation)")
        print("=" * 60)
        
        print(f"\nNeural Network Structure:")
        print(f"  Input: {X.shape[1]} units")
        print(f"  Hidden layer: {self.W1.shape[1]} neurons")
        print(f"  Output: {self.W2.shape[1]} units")
        print(f"\nTraining Parameters:")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {learning_rate}")
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Loss calculation
            loss = self.compute_loss(y, output)
            self.loss_history.append(loss)
            
            # Backward pass and weight update
            self.backward(X, y, output, learning_rate)
            
            if verbose and (epoch == 0 or epoch % 200 == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch+1}/{epochs}: Loss = {loss:.6f}")
        
        print(f"\nFinal Loss: {self.loss_history[-1]:.6f}")


def demonstrate_backpropagation():
    """
    3. Backpropagation Demonstration
    """
    # Simple regression problem: y = 2x + 1 (with some noise added)
    np.random.seed(42)
    X = np.random.rand(20, 1) * 5
    y_true = 2 * X + 1
    y = y_true + np.random.randn(20, 1) * 0.3  # Add noise
    
    # Create and train neural network
    nn = SimpleNeuralNetwork(input_size=1, hidden_size=3, output_size=1)
    nn.train(X, y, epochs=1000, learning_rate=0.1)
    
    # Prediction
    predictions = nn.forward(X)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Training curve
    ax1 = axes[0]
    ax1.plot(nn.loss_history, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Training Curve (Loss Decrease)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Prediction results
    ax2 = axes[1]
    X_sorted = np.sort(X, axis=0)
    idx = np.argsort(X.flatten())
    y_sorted = y[idx]
    pred_sorted = predictions[idx]
    
    ax2.scatter(X, y, color='blue', alpha=0.6, s=50, label='Actual Data')
    ax2.plot(X_sorted, pred_sorted, 'r-', linewidth=2, label='Neural Network Prediction')
    ax2.plot(X_sorted, 2*X_sorted + 1, 'g--', linewidth=2, label='True Function (y=2x+1)')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('Neural Network Prediction Results', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'backpropagation_result.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nBackpropagation Process:")
    print("1. Forward pass: Input → Output (calculate prediction)")
    print("2. Error calculation: Prediction - Actual")
    print("3. Backward pass: Propagate error backward to calculate each weight's contribution")
    print("4. Weight update: Modify weights using gradient descent")
    print("5. Iterate: Repeat until loss decreases")


def explain_gradient_flow():
    """
    4. Gradient Flow Explanation
    """
    print("\n" + "=" * 60)
    print("4. Gradient Flow")
    print("=" * 60)
    
    print("\nHow gradients flow in backpropagation:")
    print("\n[Forward Pass]")
    print("  Input x → Hidden layer h → Output y")
    print("  h = σ(W1·x + b1)")
    print("  y = W2·h + b2")
    
    print("\n[Backward Pass]")
    print("  Loss L → Output layer → Hidden layer → Input")
    print("  ∂L/∂y = 2(y - y_true)  # Derivative of loss w.r.t. output")
    print("  ∂L/∂W2 = (∂L/∂y) × h   # Chain rule")
    print("  ∂L/∂h = (∂L/∂y) × W2   # Chain rule")
    print("  ∂L/∂W1 = (∂L/∂h) × σ'(z1) × x  # Chain rule")
    
    print("\nKey Points:")
    print("  - Each layer's gradient depends on the next layer's gradient")
    print("  - Chain rule decomposes complex derivatives into products of simple derivatives")
    print("  - As error propagates backward, each weight's 'responsibility' is calculated")


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("Chapter 2: Calculus & Analysis - Understanding Backpropagation Algorithm")
    print("=" * 60)
    
    # 1. Chain Rule
    demonstrate_chain_rule()
    
    # 2. Backpropagation demonstration
    demonstrate_backpropagation()
    
    # 3. Gradient flow explanation
    explain_gradient_flow()
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Chain Rule: Decompose complex function derivatives into simple function derivatives")
    print("2. Backpropagation: Propagate error backward to calculate each weight's contribution")
    print("3. Weight update: Modify weights using gradient descent to reduce loss")
    print("4. Core of deep learning: Iterate forward pass → backward pass → update")
    print("5. Financial prediction models (LSTM, etc.) also learn using the same principles")


if __name__ == "__main__":
    main()


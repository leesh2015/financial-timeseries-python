"""
Cointegrated Linear Indicators (CILI) - Sample Data

This script generates sample data for two economic indicators that move
linearly over time, demonstrating potential cointegration relationships.

Cointegration:
- Two or more non-stationary series share a common stochastic trend
- Linear combination of the series is stationary
- Long-term equilibrium relationship exists

This example shows indicators that:
- Both trend upward over time
- May be cointegrated (share common trend)
- Can be tested using Johansen cointegration test
"""

import numpy as np
import matplotlib.pyplot as plt

# Setting the random seed for reproducibility
np.random.seed(42)

# Number of data points
n = 100

# Generate time index
time = np.arange(n)

# Generate two linearly increasing economic indicators with noise
indicator1 = 0.1 * time + np.random.normal(scale=2.0, size=n)
indicator2 = 0.05 * time + np.random.normal(scale=1.5, size=n)

# Plotting the data
plt.figure(figsize=(12, 6))
plt.plot(time, indicator1, label='Economic Indicator 1', color='blue', linestyle='-', marker='o')
plt.plot(time, indicator2, label='Economic Indicator 2', color='red', linestyle='-', marker='x')
plt.title('Two Economic Indicators Increasing Linearly Over Time')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

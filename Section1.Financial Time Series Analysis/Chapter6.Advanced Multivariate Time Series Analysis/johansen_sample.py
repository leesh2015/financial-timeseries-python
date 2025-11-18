"""
Johansen Cointegration Test - Simple Example

This script demonstrates the Johansen cointegration test using synthetic data.

Johansen Test:
Tests for cointegration rank (number of long-term equilibrium relationships)
between non-stationary time series.

For two variables X and Y:
- If cointegrated: They share a common stochastic trend
- Cointegrating vector β: Y - βX is stationary
- Rank = 1: One cointegrating relationship exists
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Set random seed
np.random.seed(0)

# Generate sample data
n = 100
time = np.arange(n)
X = np.cumsum(np.random.normal(size=n))  # Create a random walk using cumulative sum
Y = 2 * X + np.random.normal(size=n)  # Create Y influenced by X

data = pd.DataFrame({'X': X, 'Y': Y})

# Visualize the data
plt.figure(figsize=(12, 6))
plt.plot(data['X'], label='X')
plt.plot(data['Y'], label='Y')
plt.legend()
plt.title('Johansen Cointegration Test between X and Y')
plt.show()

# Perform Johansen cointegration test
def johansen_test(data, det_order=-1, k_ar_diff=1):
    result = coint_johansen(data, det_order, k_ar_diff)
    return result

# Print cointegration test results
johansen_result = johansen_test(data)
print("Eigenvalues:", johansen_result.eig)
print("Trace Statistic:", johansen_result.lr1)
print("Critical Values (90%, 95%, 99%):", johansen_result.cvt)

# Determine the presence of cointegration relationships
for i in range(len(johansen_result.lr1)):
    if johansen_result.lr1[i] > johansen_result.cvt[i, 1]:  # Using 95% confidence level
        print(f"Cointegration relationship exists (Rank {i+1})")
    else:
        print(f"No cointegration relationship (Rank {i+1})")

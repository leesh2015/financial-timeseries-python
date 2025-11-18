"""
Granger Causality Test - Simple Example

This script demonstrates Granger causality testing using synthetic data.

Granger Causality:
Tests if past values of X help predict Y beyond what past values of Y alone predict.

Test statistic: F-test comparing restricted vs unrestricted models
- Restricted: Y_t = f(Y_{t-1}, Y_{t-2}, ...)
- Unrestricted: Y_t = f(Y_{t-1}, Y_{t-2}, ..., X_{t-1}, X_{t-2}, ...)

If p-value < 0.05, we conclude X Granger-causes Y.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

# Set random seed
np.random.seed(0)

# Generate data
n = 100
time = np.arange(n)
A = np.sin(time / 5) + np.random.normal(scale=0.5, size=n)  # Variable A
B = np.sin(time / 5) + np.random.normal(scale=0.5, size=n) + 2 * A  # Variable B (influenced by A)

data = pd.DataFrame({'A': A, 'B': B})

# Data visualization
plt.figure(figsize=(12, 6))
plt.plot(data['A'], label='A')
plt.plot(data['B'], label='B')
plt.legend()
plt.title('Granger Causality Tests between A and B')
plt.show()

# Granger Causality Test
max_lag = 2
granger_test_result = grangercausalitytests(data[['A', 'B']], max_lag, verbose=True)

# Interpretation of results
for lag, test_result in granger_test_result.items():
    f_test_p_value = test_result[0]['ssr_ftest'][1]
    print(f"Lag {lag}: p-value = {f_test_p_value}")
    if f_test_p_value < 0.05:
        print(f"Variable A Granger-causes Variable B (lag {lag}).")
    else:
        print(f"Variable A does not Granger-cause Variable B (lag {lag}).")

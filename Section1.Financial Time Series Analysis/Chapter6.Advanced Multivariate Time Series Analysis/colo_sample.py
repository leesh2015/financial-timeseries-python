"""
Constant and Linear Trend (COLO) - Sample Data

This script generates sample data with a constant term and linear trend,
demonstrating deterministic components in time series models.

COLO (Constant + Linear trend):
- Constant term: c (intercept)
- Linear trend: t (time trend)
- Model: Y_t = c + βt + ε_t

This pattern is common in economic time series with:
- Long-term growth trends
- Deterministic time trends
- Used in VECM models with deterministic="colo"
"""

import numpy as np
import matplotlib.pyplot as plt

# Setting the random seed for reproducibility
np.random.seed(42)

# Number of data points
n = 100

# Generate linearly increasing mean over time
time = np.arange(n)
mean = 0.05 * time

# Generate fluctuating data around the mean
data = mean + np.random.normal(scale=1.0, size=n)

# Plotting the data
plt.figure(figsize=(12, 6))
plt.plot(time, data, label='Fluctuating Data', color='b', linestyle='-', marker='o')
plt.plot(time, mean, label='Linear Mean', color='r', linestyle='--')
plt.title('Data Fluctuating Linearly Around a Mean Over Time')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

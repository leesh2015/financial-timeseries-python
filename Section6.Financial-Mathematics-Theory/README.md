# Section 6 - Financial Mathematics Theory and Practical Examples

This directory contains a comprehensive guide to all financial mathematics theory used in quant trading, implemented with **easy-to-understand example code**.

## 📚 Structure

```
Section6.Financial-Mathematics-Theory/
├── README.md                           # This file
├── Chapter1_Linear_Algebra/           # Linear Algebra: Portfolio & Factors
│   ├── __init__.py
│   ├── portfolio_optimization.py      # Portfolio optimization example
│   ├── pca_factor_analysis.py         # Factor analysis using PCA
│   ├── factor_regression.py           # Multi-factor regression (Fama-French)
│   └── var_vecm_models.py             # Matrix operations in VAR & VECM models
├── Chapter2_Calculus/                  # Analysis & Calculus: Options & Optimization
│   ├── __init__.py
│   ├── gradient_descent_demo.py       # Gradient descent visualization
│   ├── backpropagation_example.py     # Understanding backpropagation algorithm
│   ├── garch_volatility.py            # Calculus principles in GARCH models
│   ├── wavelet_transform.py           # Wavelet Transform
│   ├── ito_lemma.py                   # Ito's Lemma
│   └── bayesian_optimization.py       # Bayesian Optimization
├── Chapter3_Probability_Statistics/    # Probability & Time Series Statistics
│   ├── __init__.py
│   ├── stationarity_analysis.py       # Stationarity testing and understanding
│   ├── arima_modeling.py              # Probabilistic foundations of ARIMA models
│   ├── cointegration_pairs.py         # Cointegration and pair trading
│   ├── copula_dependence.py            # Dependence analysis using Copula
│   └── monte_carlo_simulation.py       # Monte Carlo simulation
├── Chapter4_Bayesian_Filtering/       # Bayesian Statistics & Filtering
│   ├── __init__.py
│   ├── bayesian_inference.py          # Bayesian inference examples
│   ├── kalman_filter_demo.py          # Understanding Kalman Filter
│   └── state_space_models.py           # State-space models
└── utils/                              # Utility functions
    └── __init__.py
```

## 🎯 Learning Objectives

Each Chapter aims to:

1. **Understand fundamental mathematical structures**: Intuitively explain core concepts of each mathematical field
2. **Financial mathematics specialization**: Real examples applied to financial data
3. **Learn through code**: Implement formulas in code to clearly understand concepts
4. **Analogies and explanations**: Explain complex mathematics through everyday analogies

## 🚀 Quick Start

### 1. Package Installation

```bash
# Install from project root
pip install -r requirements.txt
```

### 2. Running Examples

Each Chapter's examples can be run independently:

```bash
# Chapter 1: Linear Algebra
python Chapter1_Linear_Algebra/portfolio_optimization.py

# Chapter 2: Calculus
python Chapter2_Calculus/gradient_descent_demo.py

# Chapter 3: Probability & Statistics
python Chapter3_Probability_Statistics/stationarity_analysis.py

# Chapter 4: Bayesian
python Chapter4_Bayesian_Filtering/kalman_filter_demo.py
```

## 📊 Overview of Modern Mathematics and Financial Mathematics

### 1️⃣ Classification of Modern Mathematics

Modern mathematics is broadly divided into **Pure Mathematics** and **Applied Mathematics**.

| Category | Subcategory | Main Research Areas | Financial Relevance | Notes |
| :---: | :--- | :--- | :---: | :--- |
| **Pure Math** | **Algebra** | Groups, rings, fields, equations | ⭐⭐⭐ | Linear algebra is essential |
| | **Analysis** | Limits, continuity, derivatives, integrals | ⭐⭐⭐⭐⭐ | Core of financial mathematics |
| | **Geometry** | Shapes, space, distance | ⭐⭐ | Used in data visualization |
| | **Topology** | Continuity, connectivity | ⭐ | Theoretical research level |
| | **Number Theory** | Integers, primes, congruence | ⭐ | Cryptography (blockchain) |
| | **Logic** | Proofs, set theory | ⭐ | Foundation of algorithm design |
| **Applied Math** | **Probability & Statistics** | Uncertainty, distributions, estimation | ⭐⭐⭐⭐⭐ | Language of finance |
| | **Numerical Analysis** | Approximation, optimization | ⭐⭐⭐⭐ | Essential in practice |
| | **Differential Equations** | Dynamics, SDE | ⭐⭐⭐⭐ | Option pricing, GARCH |
| | **Optimization Theory** | Constraints, objective functions | ⭐⭐⭐⭐⭐ | Core of portfolio theory |
| | **Information Theory** | Entropy, information content | ⭐⭐⭐ | Model selection (AIC/BIC) |
| | **Graph Theory** | Networks, connectivity | ⭐⭐ | System risk |
| **Computational** | **Machine Learning Math** | Gradient descent, backpropagation | ⭐⭐⭐⭐⭐ | Core of AI trading |

**Relevance Legend:**
- ⭐⭐⭐⭐⭐ Absolutely essential (pillar of financial mathematics)
- ⭐⭐⭐⭐ Very important (frequently used in practice)
- ⭐⭐⭐ Important (essential in specific areas)
- ⭐⭐ Optional (advanced applications)
- ⭐ Indirect (theoretical background or special fields)

### 2️⃣ Financial Mathematics Learning Path

```
[Essential Foundations]          [Core Applications]              [Advanced Applications]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Linear Algebra          →  Portfolio Optimization        →  PCA, Factor Models
2. Calculus                →  Option Pricing, Gradient Descent →  Ito Calculus, SDE
3. Probability & Statistics →  Time Series, Risk Management →  Bayesian, Copula
4. Numerical Analysis      →  Optimization, Simulation      →  Monte Carlo
```

## 📊 Project-Specific Mathematical Matrix

This matrix organizes the mathematical theories **actually implemented** in this project by field.

| Mathematical Field | Core Concepts | Financial Applications | Key Techniques/Algorithms |
| :--- | :--- | :--- | :--- |
| **Linear Algebra** | Vectors, matrices, eigenvalues | Portfolio optimization, factor models | Covariance matrix, regression matrix, PCA |
| **Analysis** | Derivatives, integrals, limits | Deep learning, volatility models | Gradient descent, backpropagation, GARCH |
| **Probability & Statistics** | Distributions, estimation, testing | Time series, risk management | ARIMA, Kalman, VaR, Copula |
| **Numerical Analysis** | Optimization, approximation | Portfolio, simulation | scipy.optimize, Monte Carlo |
| **Geometry** | Space, distance, dimensions | Dimensionality reduction, visualization | PCA, t-SNE, distance metrics |
| **Information Theory** | Entropy, information content | Model selection, feature selection | AIC/BIC, mutual information |
| **Graph Theory** | Networks, connectivity | Asset relationships, risk | Correlation networks, MST |

### 📝 Detailed Mathematical Techniques

#### 1. Linear Algebra
- **Covariance Matrix (Σ)**: Measures correlation and volatility between assets
- **Regression Matrix**: β = (X'X)⁻¹X'y (least squares method)
- **VAR/VECM**: Multivariate time series matrix operations
- **PCA**: Dimensionality reduction through eigenvalue decomposition

#### 2. Analysis
- **Gradient Descent**: Search for minimum following ∇f(x)
- **Backpropagation**: Calculate gradients using chain rule
- **GARCH**: σ²ₜ = α₀ + α₁ε²ₜ₋₁ + β₁σ²ₜ₋₁
- **Ito's Lemma**: Core of stochastic differential equations

#### 3. Probability & Statistics
- **ARIMA**: Stationarity testing + autoregressive models
- **Kalman Filter**: State estimation through Bayesian updates
- **VaR/CVaR**: Extreme loss risk measurement
- **Copula**: Modeling dependence independent of marginal distributions

#### 4. Numerical Analysis
- **scipy.optimize**: SLSQP, L-BFGS optimization
- **Monte Carlo**: Probabilistic simulation
- **Bayesian Optimization**: Hyperparameter tuning

## 📖 Chapter Details

### Chapter 1: Linear Algebra

**Core Analogy**: "Cocktail Recipe"
- Vector: Amount of each ingredient [Gin 30ml, Tonic 90ml, Lime 10ml]
- Matrix: Compatibility chart between ingredients (covariance matrix)
- Eigenvalue decomposition: Extract core flavors (PCA)

**Key Topics**:
- Portfolio variance calculation: σ² = wᵀΣw
  - This formula calculates **"how risky is this cocktail?"**
  - Foundation of MVO (Mean-Variance Optimization) and factor models
- **PCA (Principal Component Analysis)**:
  - Compresses movements of hundreds of stocks into a few key factors like 'market', 'interest rates', 'oil prices'
- Meaning and calculation of covariance matrices
- Matrix operations in Fama-French factor models
- Matrix representation and estimation of VAR & VECM models

### Chapter 2: Analysis & Calculus

**Core Analogy**: "Descending a mountain in fog"
- Derivative: Speedometer (measuring rate of change) - "How fast is the price/error changing at this moment?"
- Gradient descent: Feeling the slope with your feet to descend to the lowest valley (minimum error)
- Backpropagation: Propagating errors backward to assign responsibility

**Key Topics**:
- Optimization through gradient descent
- **Deep Learning (Backpropagation)**:
  - Principle of how LSTM learns. Differentiates prediction errors to assign responsibility and adjust weights
- Calculus principles in GARCH models
- **Ito's Lemma**:
  - Formula for calculating the rate of change of option prices when stock prices jump randomly (Brownian motion)

### Chapter 3: Probability & Time Series Statistics

**Core Analogy**: "Predicting the future from past weather"
- Stationarity: Spring (property of returning to original position) - fundamental premise of time series analysis
- Cointegration: Owner and dog (connected by a leash) - seem to move independently but ultimately move together due to long-term equilibrium

**Key Topics**:
- Stationarity testing (ADF Test)
- Probabilistic foundations of ARIMA models
- **ARIMA / GARCH**:
  - Statistically estimates patterns of past data (AR), errors (MA), and volatility (GARCH)
- Cointegration and pair trading
- **Copula**:
  - **"Panic Room Effect"**: Models tail dependence where assets that normally move independently all crash together during crises
- Probability distributions in GARCH models

### Chapter 4: Bayesian Statistics & Filtering

**Core Analogy**: "Narrowing down suspects with new clues"
- Bayesian inference: Detective investigation - starts with many suspects (prior probability), updates probability of the real culprit (posterior probability) as evidence (data) emerges
- Kalman Filter: Navigation (combining GPS + speed) - combines noisy GPS signals (observations) with car speed (model) to estimate 'true position'

**Key Topics**:
- Bayesian update (prior → posterior probability)
- State estimation with Kalman Filter
- **State-Space Models**:
  - Tracks unobserved 'true market beta'
- **Prophet**:
  - Flexibly decomposes trends and seasonality using Bayesian methods to predict the future
- Bayesian structure of Prophet model

## 🔗 Connection to Project

| Mathematical Field | Core Concepts | Main Applications | Analogy |
| :--- | :--- | :--- | :--- |
| **Linear Algebra** | Matrix operations, eigenvalues | Portfolio optimization, factor models | Cocktail recipe |
| **Calculus** | Gradient descent, backpropagation | Deep learning, optimization | Descending in fog |
| **Time Series Statistics** | Stationarity, cointegration | Pair trading | Owner and dog |
| **Bayesian** | Kalman Filter, posterior probability | State-space models, Prophet | Detective investigation |

This guide serves as a map that mathematically supports **"why the code works that way"**, rather than simply listing formulas.

## 📝 Usage

Each example file can be run independently and follows this structure:

1. **Theory explanation**: Mathematical concepts explained with analogies
2. **Basic examples**: Simple mathematical examples
3. **Financial application**: Applied to real financial data
4. **Visualization**: Results expressed as graphs

## 📚 Code Structure

Each example file is self-contained and includes:
- **Theory explanation**: Mathematical concepts explained with intuitive analogies
- **Code implementation**: Practical Python code demonstrating the concepts
- **Financial applications**: Real-world examples using financial data
- **Visualizations**: Graphs and plots to illustrate results

The code itself serves as documentation, with comments explaining the mathematical concepts through everyday analogies.

## ⚠️ Notes

- All examples are for **educational purposes**
- Sufficient verification is required before using in actual investments
- Data is automatically downloaded via yfinance

## 🤝 Contributing

This guide aims to explain all financial mathematics in quant trading in an easy-to-understand way.
Please suggest improvements or additional examples!

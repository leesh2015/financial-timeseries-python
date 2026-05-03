# Financial Time Series Analysis with Python

[English](README.md) | [한국어](README_KR.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Udemy Course](https://img.shields.io/badge/Udemy-Course-orange)](https://www.udemy.com/course/mastering-financial-time-series-analysis-with-python/?referralCode=BA6CA9A3E5406E41359E&couponCode=KRLETSLEARNNOW)

A comprehensive open-source project for mastering financial time series analysis, algorithmic trading strategies, and production-level automated trading systems using Python. This project combines theoretical knowledge with practical implementation, featuring a **live production trading system** that is currently operating with real capital.

## 🎓 Udemy Course

**Mastering Financial Time Series Analysis with Python**

👉 [Enroll in the Course](https://www.udemy.com/course/mastering-financial-time-series-analysis-with-python/?referralCode=BA6CA9A3E5406E41359E&couponCode=KRLETSLEARNNOW)

- **All Sections**: Course materials are fully uploaded and available
- **Section 3**: Live production system with real-time trading records

## 📊 Live Trading Records

**Currently in Live Production**

This project includes a **production trading system** that is actively trading with real capital. Daily trading records are automatically uploaded to:

🌐 **[Trading History Dashboard](https://leenaissance.com/competition)**

The system uses broker API integration for automated trading, and all trades are transparently displayed on the website.

## 📚 Project Structure

### Section 1: Financial Time Series Analysis

**Status**: ✅ Course Available

Comprehensive coverage of time series fundamentals and advanced techniques:

- **Chapter 1**: Fundamentals of Time Series Data Analysis
  - Stationarity and non-stationarity
  - Differencing and transformation
  - Seasonal decomposition

- **Chapter 2**: Advanced Time Series Analysis
  - ADF (Augmented Dickey-Fuller) test
  - AR (Autoregressive) models
  - PACF (Partial Autocorrelation Function) analysis
  - Random walk theory

- **Chapter 3**: Univariate Time Series Analysis
  - AR, MA, ARMA models
  - AIC vs BIC model selection
  - Auto-ARIMA
  - Ljung-Box test for residual analysis

- **Chapter 4**: Advanced Volatility Modeling and Forecasting
  - ARCH models
  - GARCH models
  - ARIMA-GARCH hybrid models
  - Backtesting strategies

- **Chapter 5**: Multivariate Time Series Analysis
  - VAR (Vector Autoregression) models
  - VARMA models
  - Granger causality analysis

- **Chapter 6**: Advanced Multivariate Time Series Analysis
  - VECM (Vector Error Correction Model)
  - Johansen cointegration test
  - VAR IRF (Impulse Response Function)
  - VAR FEVD (Forecast Error Variance Decomposition)
  - VECM-APARCH hybrid models

### Section 2: Advanced Investment Strategy Design

**Status**: ✅ Course Available

Practical implementation of trading strategies:

- **Chapter 1**: Dynamic Time Series Simulations
  - VECM-EGARCH hybrid model
  - Dynamic re-optimization
  - Long/short position management

- **Chapter 2**: Applying Strategies to Bitcoin Trading
  - Bitcoin-specific optimizations
  - Commission fee considerations
  - Volatility-based re-optimization

- **Chapter 3**: AI Trading Using Binance
  - Binance API integration
  - Real-time signal generation
  - Automated order execution

### Section 3: Production Investment Strategy

**Status**: ✅ Course Available 🚀 **Live Production System**

A production-level trading system currently operating with real capital:

- **Chapter 1: VECM-EGARCH Hybrid Model**
  - Vector Error Correction Model for cointegration relationships
  - Exponential GARCH for volatility modeling
  - Dynamic position sizing based on model confidence
  - Key Features:
    - ✅ Information leakage prevention (walking forward validation)
    - ✅ Dynamic re-optimization based on ECT alpha
    - ✅ Confidence-based position sizing (0.2~0.8 fraction range)
    - ✅ Separate forecast horizons for buy (4 days) and sell (7 days)

- **Chapter 2: Reinforcement Learning (RL) Strategy**
  - VECM-GARCH Hybrid combined with RL Agent
  - Market Regime Detection (Bull, Bear, Sideways, High Vol)
  - Simple Policy RL agent for dynamic position blending
  - Adaptive confidence thresholds

- **Mathematical Models**:

  **VECM Model**:
  ```
  ΔY_t = αβ'Y_{t-1} + Γ₁ΔY_{t-1} + ... + Γ_{p-1}ΔY_{t-p+1} + ε_t
  ```
  - α: adjustment coefficients (speed of adjustment to equilibrium)
  - β: cointegration vectors (long-run relationships)
  - Γᵢ: short-run dynamics coefficients

  **EGARCH Model**:
  ```
  log(σ²_t) = ω + Σᵢ₌₁ᵖ (αᵢ|z_{t-i}| + γᵢz_{t-i}) + Σⱼ₌₁ᵠ βⱼlog(σ²_{t-j})
  ```
  - Captures asymmetric volatility effects
  - Ensures positive variance through log transformation

  **Hybrid Forecast**:
  ```
  Ŷ_{t+1} = VECM_forecast + EGARCH_mean_adjustment
  ```

- **Trading Strategy**:
  - Long Entry: `hybrid_yhat_buy > actual_price AND lower_price < lower_bound_mean`
  - Long Exit: `upper_price > upper_bound_mean`
  - Dynamic Re-optimization: When ECT alpha changes from negative to positive

### Section 4: Advanced Time Series Models

**Status**: ✅ Course Available

[📖 Detailed Documentation](Section4.Advanced%20Time%20Series%20Models/README.md)

Modern statistical and ML techniques for challenging temporal structures:

- **Chapter 1**: State-Space Models – time-varying beta tracking and error-correction diagnostics
- **Chapter 2**: Kalman Filter suite (Custom, FilterPy, PyKalman EM, Particle) – dynamic beta tracking with quantile-based switching
- **Chapter 3**: Prophet Model – seasonality-aware forecasting with rolling re-training
- **Chapter 4**: Deep Learning (LSTM) – direction classification with imbalance-aware training
- **Chapter 5**: Tree-Based ML (XGBoost) – binary classification for direction prediction with rich technical features and ROC-optimized thresholds
- **Chapter 6**: Wavelet Transform – multi-resolution feature engineering for volatility regimes
- **Chapter 7**: Copula Models – dependence modeling and tail-risk simulation

### Section 5: Factor-Based Asset Pricing Models

**Status**: ✅ Course Available

[📖 Detailed Documentation](Section5.Factor-Based%20Asset%20Pricing%20Models/README.md)

Theoretical foundations and practical applications of factor-based asset pricing models:

- **Chapter 1**: CAPM Limitations and Fama-French Model Origins
  - Empirical testing of CAPM
  - Identifying market anomalies (Size, Value)
  - Visualizing model limitations

- **Chapter 2**: Fama-French 3-Factor Model
  - Implementing the 3-factor model
  - Calculating factor exposures (Betas)
  - Comparing multi-factor models vs CAPM

- **Chapter 3**: Fama-French 5-Factor and Extended Models
  - Profitability (RMW) and Investment (CMA) factors
  - Momentum factor integration (6-Factor model)
  - Model selection and comparison

- **Chapter 4**: Practical Application and Backtesting
  - Factor-based portfolio construction
  - Walk-forward validation
  - Transaction cost analysis
  - Performance evaluation (Sharpe, Alpha, etc.)
- **Chapter 5**: Transaction Cost Analysis & Execution Optimization

### Section 6: Quantum Market State Engine

**Status**: ✅ Course Available 🚀 **[v2.3] Symmetric Parity & Zero-Lag Upgrade**

[📖 Detailed Documentation](Section6.Quantum-Market-State-Engine/README.md)

A paradigm shift in quantitative trading using **Quantum Fluid Dynamics** and event-driven causality:

- **Chapter 1**: The Death of Moving Averages (Event-Time vs Clock-Time)
- **Chapter 2**: Reconstructing the Market as a Hamiltonian System
- **Chapter 3**: Matrix Mechanics & Dynamic Dimension Scaling (5x5 to 10x10)
- **Chapter 4**: The Probability Dial (Customizable Win-Rate Engineering)
- **Chapter 5**: [NEW] **Symmetric Parity**: Eliminating dimensional bias and ensuring mathematical integrity
- **Chapter 6**: [NEW] **Zero-Lag Architecture**: Ultra-precise predictor removing horizon ($N$) derivation delay
- **Verified Proof**: Overwhelming win-rate on TQQQ (3x ETF) and BTC/USDT (Crypto) empirical data.

### Appendix: Financial Mathematics Theory and Practical Examples

**Status**: ✅ Available

[📖 Detailed Documentation](Appendix/README.md)

A comprehensive guide to all financial mathematics theory used in quant trading, implemented with **easy-to-understand example code**:

- **Chapter 1: Linear Algebra**
  - Portfolio optimization
  - PCA-based factor analysis
  - Multi-factor regression (Fama-French)
  - Matrix operations in VAR & VECM models

- **Chapter 2: Analysis & Calculus**
  - Gradient descent visualization
  - Understanding backpropagation algorithm
  - Calculus principles in GARCH models
  - Wavelet Transform
  - Ito's Lemma
  - Bayesian Optimization

- **Chapter 3: Probability & Time Series Statistics**
  - Stationarity testing and understanding
  - Probabilistic foundations of ARIMA models
  - Cointegration and pair trading
  - Dependence analysis using Copula
  - Monte Carlo simulation

- **Chapter 4: Bayesian Statistics & Filtering**
  - Bayesian inference examples
  - Understanding Kalman Filter
  - State-space models

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/leesh2015/financial-timeseries-python.git
cd financial-timeseries-python
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Examples

**Section 1 - Time Series Analysis:**
```bash
cd "Section1.Financial Time Series Analysis/Chapter1.Fundamentals of Time Series Data Analysis"
python stable_data.py
```

**Section 2 - Strategy Design:**
```bash
cd "Section2.Advanced Investment Strategy Design/Chapter1.Dynamic Time Series Simulations"
python dynamic_simulation.py
```

**Section 3 - Production Simulation:**
```bash
# Chapter 1: VECM-EGARCH Hybrid
cd "Section3.Production Investment Strategy/Chapter1.VECM-EGARCH Hybrid"
python production_simulation_.py

# Chapter 2: Reinforcement Learning
cd "Section3.Production Investment Strategy/Chapter2.Reinforcement Learning"
python dynamic_simulation_rl.py
```

**Section 4 - Advanced Time Series Models:**
```bash
cd "Section4.Advanced Time Series Models/Chapter1.State-Space Models"
python state_space_model.py
```

**Section 5 - Factor Models:**
```bash
cd "Section5.Factor-Based Asset Pricing Models/Chapter4.Practical Application and Backtesting"
python factor_portfolio_backtest.py

**Section 6 - Quantum Market State Engine:**
```bash
# Terminal 1: Start Mock Collector
cd "Section6.Quantum-Market-State-Engine/scripts"
python mock_collector.py

# Terminal 2: Run Live Predictor UI (Money Maker Dashboard)
python quantum_predictor.py --threshold 0.83
```
```

**Appendix - Financial Mathematics:**
```bash
# Install dependencies from project root (skip if already installed)
pip install -r requirements.txt

cd Appendix

# Chapter 1: Linear Algebra
python Chapter1_Linear_Algebra/portfolio_optimization.py

# Chapter 2: Calculus
python Chapter2_Calculus/gradient_descent_demo.py

# Chapter 3: Probability & Statistics
python Chapter3_Probability_Statistics/stationarity_analysis.py

# Chapter 4: Bayesian
python Chapter4_Bayesian_Filtering/kalman_filter_demo.py
```

Results will be saved in the `results/` folder within each section.

## 📦 Dependencies

Core dependencies (see `requirements.txt` for full list):

- **Data Science**: `numpy`, `pandas`, `scipy`
- **Time Series**: `statsmodels`, `arch`, `pmdarima`
- **Data Collection**: `yfinance`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`
- **Cryptocurrency**: `ccxt` (for Binance integration)
- **Excel Support**: `openpyxl`

## 🎯 Key Features

### Information Leakage Prevention
- Uses data up to time t-1 to predict price at time t
- Models are re-trained at each step using only historical data (walking forward)
- No future data is used in any prediction or optimization step

### Dynamic Model Adaptation
- Automatic re-optimization when market conditions change
- ECT alpha monitoring for cointegration relationship health
- Volatility-based model adjustments

### Confidence-Based Position Sizing
- Dynamic position sizing based on VECM model confidence
- Fraction range: 0.2 (low confidence) to 0.8 (high confidence)
- Adaptive threshold calculation using rolling window

### Production-Ready
- Real-time broker API integration
- Automated trade execution
- Daily performance tracking and reporting
- Transparent trade history on web dashboard

## 📈 Performance Metrics

The production system tracks comprehensive performance metrics:

- Total P&L
- Win Rate
- Sharpe Ratio
- Maximum Drawdown
- Annualized Returns
- Buy-and-Hold Comparison

View live metrics at: [Trading History Dashboard](https://leenaissance.com/competition)

## 🔬 Research & Methodology

This project implements state-of-the-art financial econometrics techniques:

- **Cointegration Analysis**: Identifying long-run equilibrium relationships
- **Error Correction Models**: Capturing short-term deviations from equilibrium
- **GARCH Family Models**: Modeling volatility clustering and asymmetry
- **Hybrid Forecasting**: Combining multiple models for improved accuracy
- **Dynamic Optimization**: Adapting to changing market regimes

## 📖 Course-Code Mapping

| Course Section | Repository Section | Status |
|---------------|-------------------|--------|
| Section 1: Time Series Fundamentals | `Section1.Financial Time Series Analysis/` | ✅ Available |
| Section 2: Strategy Design | `Section2.Advanced Investment Strategy Design/` | ✅ Available |
| Section 3: Production System | `Section3.Production Investment Strategy/` | ✅ Available 🚀 |
| Section 4: Advanced Time Series Models | `Section4.Advanced Time Series Models/` | ✅ Available |
| Section 5: Factor Models | `Section5.Factor-Based Asset Pricing Models/` | ✅ Available |
| Section 6: Quantum Engine | `Section6.Quantum-Market-State-Engine/` | ✅ Available 🚀 |
| Appendix: Financial Mathematics | `Appendix/` | ✅ Available |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

**Join the Discussion**: Have questions, ideas, or want to share your results? Join our [GitHub Discussions](https://github.com/leesh2015/financial-timeseries-python/discussions) to connect with the community!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**Important**: This project is for educational and research purposes. The production trading system is provided as a demonstration of the concepts taught in the course. 

- Past performance does not guarantee future results
- Trading involves risk of financial loss
- Always conduct thorough backtesting before deploying any trading strategy
- The authors are not responsible for any financial losses incurred from using this code

## 🔗 Links

- **Udemy Course**: [Mastering Financial Time Series Analysis with Python](https://www.udemy.com/course/mastering-financial-time-series-analysis-with-python/?referralCode=BA6CA9A3E5406E41359E&couponCode=KRLETSLEARNNOW)
- **Trading Dashboard**: [leenaissance.com/competition](https://leenaissance.com/competition)
- **GitHub Discussions**: [Join the Community](https://github.com/leesh2015/financial-timeseries-python/discussions)

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub or contact through the Udemy course platform.

---

**Made with ❤️ for the algorithmic trading community**

*This project demonstrates the complete journey from theoretical time series analysis to production-level automated trading systems.*


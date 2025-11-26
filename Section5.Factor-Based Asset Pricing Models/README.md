# Section 5: Factor-Based Asset Pricing Models

## Overview

This section covers the theoretical foundations and practical applications of factor-based asset pricing models, with a focus on the Fama-French models. You will learn how CAPM's limitations led to the development of multi-factor models, understand the theoretical background of Fama-French models, and implement practical factor-based investment strategies.

## 🎓 Learning Objectives

By the end of this section, you will be able to:
1. Understand the limitations of CAPM and why multi-factor models were developed
2. Explain the theoretical foundations of Fama-French 3-Factor and 5-Factor models
3. Implement factor exposure calculations using regression analysis
4. Build factor-based portfolios with proper risk management
5. Conduct rigorous backtesting with train/test data separation
6. Evaluate portfolio performance using standard metrics

## 📁 Repository Structure

```
Section5.Factor-Based Asset Pricing Models/
├── Chapter1.CAPM Limitations and Fama-French Model Origins/
│   ├── capm_limitations.py          # CAPM 한계 실증 및 시각화
│   └── (output files: capm_limitations_analysis.png, capm_regression_results.csv)
├── Chapter2.Fama-French 3-Factor Model/
│   └── fama_french_3factor.py       # 3-Factor 모델 구현 및 CAPM 비교
├── Chapter3.Fama-French 5-Factor and Extended Models/
│   └── fama_french_5factor.py       # 5-Factor, 6-Factor 모델 비교
├── Chapter4.Practical Application and Backtesting/
│   ├── factor_portfolio_backtest.py # 팩터 기반 포트폴리오 백테스팅
│   └── (output files: factor_portfolio_backtest.png, selected_stocks.csv)
├── utils/
│   ├── __init__.py
│   ├── data_loader.py               # Fama-French 데이터 로딩
│   ├── factor_utils.py              # 팩터 노출도 계산
│   └── metrics.py                   # 성과 지표 계산
├── requirements.txt
└── README.md
```

---

## 📚 Chapter 1: CAPM Limitations and Fama-French Model Origins

### Learning Objectives

By the end of this chapter, you will be able to:
1. Understand the assumptions and limitations of the Capital Asset Pricing Model (CAPM)
2. Identify market anomalies that CAPM cannot explain
3. Explain why multi-factor models were developed
4. Understand the theoretical motivation behind Fama-French models
5. Recognize the difference between market anomalies and risk factors

### Theoretical Background

#### The Capital Asset Pricing Model (CAPM)

**CAPM Equation:**
```
E(R_i) = R_f + β_i × (E(R_m) - R_f)
```

Where:
- `E(R_i)`: Expected return on asset i
- `R_f`: Risk-free rate
- `β_i`: Beta (sensitivity to market risk)
- `E(R_m)`: Expected market return

**Key Assumptions:**
1. Investors are rational and risk-averse
2. All investors have the same investment horizon
3. All investors have the same expectations about returns
4. There are no transaction costs or taxes
5. All assets are perfectly divisible and liquid
6. There is a single risk-free rate
7. **Market portfolio is the only source of systematic risk**

#### CAPM's Limitations

**1. Single Factor Limitation**
- CAPM assumes market beta is the **only** source of systematic risk
- In reality, stock returns are influenced by multiple factors
- Market beta alone cannot explain all return variations

**2. Market Anomalies**

**Size Effect (Banz, 1981):**
- Small-cap stocks earn higher returns than large-cap stocks
- CAPM cannot explain this using market beta alone
- Small firms have higher betas, but the return premium is larger than beta would predict

**Value Effect (Rosenberg, Reid, & Lanstein, 1985):**
- Value stocks (high book-to-market) outperform growth stocks (low book-to-market)
- This premium cannot be explained by market beta alone
- Value stocks have similar betas to growth stocks but earn higher returns

**Momentum Effect (Jegadeesh & Titman, 1993):**
- Stocks with high past returns continue to outperform
- Short-term momentum (3-12 months) contradicts market efficiency
- CAPM cannot explain this pattern

**3. Low R-squared**
- CAPM typically explains only 20-40% of return variation
- 60-80% of returns remain unexplained
- Suggests missing risk factors

**4. Non-zero Alpha**
- If CAPM were correct, alpha (abnormal returns) should be zero
- Many stocks show statistically significant alpha
- Indicates model misspecification

### The Birth of Fama-French Models

**Fama & French (1992): "The Cross-Section of Expected Stock Returns"**
- Documented size and value effects
- Showed these effects persist after controlling for market beta
- Suggested these might be risk factors, not market inefficiencies

**Fama & French (1993): "Common Risk Factors in the Returns on Stocks and Bonds"**
- Introduced the 3-factor model
- Added SMB (size) and HML (value) factors
- Explained that these represent risk premiums, not anomalies

**Key Insight:**
- Market anomalies might actually be risk factors
- Size and value represent systematic risks
- Investors require compensation for bearing these risks

### Code Implementation

**File: `capm_limitations.py`**

This script demonstrates CAPM limitations by:
1. Running CAPM regressions on multiple stocks
2. Showing low R-squared values (typically 0.2-0.4)
3. Identifying non-zero alpha values
4. Visualizing the results

**Usage:**
```bash
cd "Chapter1.CAPM Limitations and Fama-French Model Origins"
python capm_limitations.py
```

**Output:**
- `capm_limitations_analysis.png`: Visualization of CAPM limitations
- `capm_regression_results.csv`: Detailed regression results

---

## 📚 Chapter 2: Fama-French 3-Factor Model

### Learning Objectives

By the end of this chapter, you will be able to:
1. Understand the Fama-French 3-factor model equation and components
2. Explain the economic meaning of SMB and HML factors
3. Calculate factor exposures using regression analysis
4. Interpret regression results (betas, R-squared, alpha)
5. Perform GRS test for model validation
6. Understand how the model tests market efficiency

### Theoretical Background

#### Fama-French 3-Factor Model

**Model Equation:**
```
R_i - R_f = α_i + β_i(R_m - R_f) + s_i(SMB) + h_i(HML) + ε_i
```

Where:
- `R_i - R_f`: Excess return on stock i
- `R_m - R_f`: Market excess return (market risk premium)
- `SMB`: Small Minus Big (size factor)
- `HML`: High Minus Low (value factor)
- `β_i`: Market beta (sensitivity to market risk)
- `s_i`: Size factor loading (SMB beta)
- `h_i`: Value factor loading (HML beta)
- `α_i`: Alpha (abnormal return, should be zero if model is correct)

#### Factor Construction

**SMB (Small Minus Big) Factor**
- **Construction**: Return on small-cap portfolio minus return on large-cap portfolio
- **Economic Meaning**: Size risk premium
- **Interpretation**: 
  - Positive `s_i`: Stock behaves like small-cap (higher expected return)
  - Negative `s_i`: Stock behaves like large-cap (lower expected return)

**HML (High Minus Low) Factor**
- **Construction**: Return on high book-to-market (value) portfolio minus return on low book-to-market (growth) portfolio
- **Economic Meaning**: Value risk premium
- **Interpretation**:
  - Positive `h_i`: Stock behaves like value stock (higher expected return)
  - Negative `h_i`: Stock behaves like growth stock (lower expected return)

#### Factor Loadings (Betas)

**Market Beta (β_i):**
- Measures sensitivity to market movements
- Similar to CAPM beta
- Expected range: 0.5 - 1.5 for most stocks

**Size Beta (s_i):**
- Measures exposure to size factor
- Positive: Small-cap exposure
- Negative: Large-cap exposure
- Typical range: -0.5 to 1.0

**Value Beta (h_i):**
- Measures exposure to value factor
- Positive: Value stock exposure
- Negative: Growth stock exposure
- Typical range: -0.5 to 1.0

#### Model Interpretation

**Expected Return:**
```
E(R_i) = R_f + β_i × E(R_m - R_f) + s_i × E(SMB) + h_i × E(HML)
```

**Risk Decomposition:**
- Market risk: `β_i × (R_m - R_f)`
- Size risk: `s_i × SMB`
- Value risk: `h_i × HML`
- Idiosyncratic risk: `ε_i`

#### Alpha Interpretation

**Alpha (α_i):**
- Abnormal return not explained by factors
- If model is correct and market is efficient: `α_i = 0`
- If `α_i ≠ 0`:
  - Model misspecification (missing factors)
  - Market inefficiency
  - Measurement error

#### GRS Test (Gibbons-Ross-Shanken Test)

**Purpose:**
- Test if all alphas are jointly zero
- Tests market efficiency hypothesis
- More powerful than individual t-tests

**Null Hypothesis:**
- `H_0`: All alphas = 0 (market is efficient)
- `H_1`: At least one alpha ≠ 0 (market is inefficient or model is wrong)

**Interpretation:**
- If GRS test rejects H_0: Model cannot explain all returns (missing factors or inefficiency)
- If GRS test fails to reject H_0: Model adequately explains returns (market efficiency maintained)

### Code Implementation

**File: `fama_french_3factor.py`**

This script demonstrates:
1. Loading Fama-French 3-factor data
2. Calculating factor exposures for individual stocks
3. Comparing CAPM vs Fama-French 3-factor model
4. Interpreting regression results

**Usage:**
```bash
cd "Chapter2.Fama-French 3-Factor Model"
python fama_french_3factor.py
```

**Key Results:**
- Average CAPM R²: ~0.33
- Average FF3 R²: ~0.38
- Improvement: ~15% increase in explanatory power

---

## 📚 Chapter 3: Fama-French 5-Factor and Extended Models

### Learning Objectives

By the end of this chapter, you will be able to:
1. Understand the Fama-French 5-factor model and its components
2. Explain the economic meaning of RMW and CMA factors
3. Understand the 6-factor model (adding momentum)
4. Compare different factor models
5. Select appropriate model for different applications

### Theoretical Background

#### Fama-French 5-Factor Model (2015)

**Model Equation:**
```
R_i - R_f = α_i + β_i(R_m - R_f) + s_i(SMB) + h_i(HML) + r_i(RMW) + c_i(CMA) + ε_i
```

**New Factors:**
- **RMW** (Robust Minus Weak): Profitability factor
- **CMA** (Conservative Minus Aggressive): Investment factor

#### RMW (Robust Minus Weak) Factor

**Construction:**
- Robust: High profitability firms
- Weak: Low profitability firms
- RMW = Return on robust portfolio - Return on weak portfolio

**Economic Meaning:**
- Profitability risk premium
- More profitable firms earn higher returns
- Related to operating efficiency

**Interpretation:**
- Positive `r_i`: Stock behaves like profitable firm
- Negative `r_i`: Stock behaves like unprofitable firm

#### CMA (Conservative Minus Aggressive) Factor

**Construction:**
- Conservative: Low investment firms (low asset growth)
- Aggressive: High investment firms (high asset growth)
- CMA = Return on conservative portfolio - Return on aggressive portfolio

**Economic Meaning:**
- Investment risk premium
- Firms that invest conservatively earn higher returns
- Related to investment efficiency

**Interpretation:**
- Positive `c_i`: Stock behaves like conservative firm
- Negative `c_i`: Stock behaves like aggressive firm

#### 6-Factor Model (Adding Momentum)

**Model Equation:**
```
R_i - R_f = α_i + β_i(R_m - R_f) + s_i(SMB) + h_i(HML) + r_i(RMW) + c_i(CMA) + m_i(MOM) + ε_i
```

**MOM (Momentum) Factor:**
- Construction: Winners (high past returns) - Losers (low past returns)
- Economic meaning: Momentum risk premium
- Interpretation: Positive `m_i` = momentum exposure

#### Model Comparison

| Model | Factors | R² (Typical) | Use Case |
|-------|---------|---------------|----------|
| CAPM | Market | 0.2-0.4 | Basic analysis |
| FF 3-Factor | Market, SMB, HML | 0.4-0.6 | **Standard analysis (Recommended)** |
| FF 5-Factor | Market, SMB, HML, RMW, CMA | 0.5-0.7 | Comprehensive analysis |
| FF 6-Factor | + MOM | 0.6-0.8 | Full factor model |

**Why 3-Factor is Most Popular:**
- Best balance of explanatory power and simplicity
- Most widely used in practice
- Sufficient for most applications
- Lower overfitting risk

#### Factor Relationships

**Size and Value:**
- Small-cap value stocks: Highest expected returns
- Large-cap growth stocks: Lowest expected returns

**Profitability and Investment:**
- High profitability + Low investment: Best returns
- Low profitability + High investment: Worst returns

### Code Implementation

**File: `fama_french_5factor.py`**

This script demonstrates:
1. Fama-French 5-factor model
2. 6-factor model (adding momentum)
3. Model comparison
4. Factor interpretation

**Usage:**
```bash
cd "Chapter3.Fama-French 5-Factor and Extended Models"
python fama_french_5factor.py
```

**Key Results:**
- Average 3-Factor R²: ~0.38
- Average 5-Factor R²: ~0.40
- Average 6-Factor R²: ~0.41
- Conclusion: 3-Factor provides best balance

---

## 📚 Chapter 4: Practical Application and Backtesting

### Learning Objectives

By the end of this chapter, you will be able to:
1. Construct factor-based portfolios using multi-stage filtering
2. Implement proper train/test data separation
3. Conduct walk-forward validation
4. Evaluate portfolio performance with proper metrics
5. Consider transaction costs and liquidity
6. Build production-ready factor investment strategies

### Key Concepts

#### Train/Test Data Separation

**Why Separate?**
- Prevent overfitting
- Realistic performance estimation
- Avoid look-ahead bias

**Method:**
- Training set: Model estimation (factor exposures, thresholds)
- Test set: Out-of-sample performance evaluation
- Typical split: 70% train, 30% test

#### Walk-Forward Validation

**Concept:**
- Rolling window approach
- Train on historical data, test on future data
- Re-estimate model periodically

**Implementation:**
1. Use data up to time t for training
2. Test on data from t+1 to t+n
3. Roll forward and repeat

#### Multi-Stage Filtering

**Approach:**
1. **Stage 1**: Pre-filter (liquidity, data quality)
2. **Stage 2**: Factor exposure calculation
3. **Stage 3**: Factor-based selection
4. **Stage 4**: Risk management (diversification, correlation)
5. **Stage 5**: Final portfolio construction

#### Performance Evaluation

**Metrics:**
- Sharpe Ratio: Risk-adjusted return measure
- Alpha (Jensen's Alpha): Excess return over benchmark
- Information Ratio: Active return per unit of tracking error
- Maximum Drawdown: Largest peak-to-trough decline
- Annualized Return: Return scaled to annual basis
- Volatility: Standard deviation of returns

**Benchmark Comparison:**
- Market index (S&P 500, NASDAQ)
- Factor-matching benchmark
- Buy-and-hold strategy

#### Transaction Costs

**Components:**
- Commission fees
- Bid-ask spread
- Market impact
- Taxes

**Impact:**
- Reduces net returns
- Affects rebalancing frequency
- Important for realistic backtesting

### Code Implementation

**File: `factor_portfolio_backtest.py`**

This script implements:
1. Train/test data separation
2. Multi-stage factor-based portfolio construction
3. Walk-forward validation
4. Performance evaluation with proper metrics
5. Transaction cost consideration

**Usage:**
```bash
cd "Chapter4.Practical Application and Backtesting"
python factor_portfolio_backtest.py
```

**Output:**
- `factor_portfolio_backtest.png`: Performance visualization
- `selected_stocks.csv`: Selected portfolio stocks

**Key Features:**
- Automatic train/test split (70/30)
- Factor-based stock selection
- Transaction cost consideration (0.2% per trade)
- Comprehensive performance metrics

**Example Results:**
- Annualized Return: ~50-60%
- Sharpe Ratio: ~1.2-1.3
- Maximum Drawdown: ~-35% to -40%
- Alpha: ~15-20%

**⚠️ Important Notes:**
- Results are for demonstration purposes
- Real-world performance may differ
- Always validate with out-of-sample data
- Consider additional risk management

---

## 🚀 Setup and Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection (for downloading Fama-French factor data)

### Installing Packages

```bash
# Navigate to Section5 directory
cd "Section5.Factor-Based Asset Pricing Models"

# Install all required packages
pip install -r requirements.txt
```

### Required Packages
- `numpy`: Numerical computing
- `pandas`: Data manipulation
- `matplotlib`: Visualization
- `scipy`: Statistical functions
- `statsmodels`: Regression analysis
- `yfinance`: Stock data download
- `requests`: HTTP requests for Fama-French data

---

## 🛠️ Utilities

### Common Utilities (`utils/`)

**`data_loader.py`**
- `load_ff_factors(model_type, frequency)`: Load Fama-French factor data
  - `model_type`: '3-factor', '5-factor', or '6-factor'
  - `frequency`: 'daily' or 'monthly'
- `load_stock_data(ticker, start, end)`: Load stock price data using yfinance
- `align_data(stock_data, factor_data)`: Align stock and factor data by date

**`factor_utils.py`**
- `calculate_factor_exposures(stock_returns, factor_data, factor_cols)`: Calculate factor exposures using OLS regression
- `construct_factor_portfolio(...)`: Construct factor-based portfolio

**`metrics.py`**
- `calculate_sharpe_ratio(returns, risk_free)`: Calculate Sharpe ratio
- `calculate_alpha(portfolio_returns, benchmark_returns, risk_free)`: Calculate Jensen's alpha
- `calculate_information_ratio(portfolio_returns, benchmark_returns)`: Calculate information ratio
- `calculate_max_drawdown(returns)`: Calculate maximum drawdown
- `calculate_annualized_return(returns)`: Calculate annualized return
- `calculate_volatility(returns)`: Calculate annualized volatility

### Usage Example

```python
from utils.data_loader import load_ff_factors, load_stock_data, align_data
from utils.factor_utils import calculate_factor_exposures
from utils.metrics import calculate_sharpe_ratio, calculate_alpha

# Load Fama-French 3-factor data
ff_factors = load_ff_factors('3-factor', 'daily')

# Load stock data
stock_data = load_stock_data('AAPL', start='2020-01-01', end='2024-01-01')

# Align data
stock_returns, factor_data = align_data(stock_data, ff_factors)

# Calculate factor exposures
exposures = calculate_factor_exposures(
    stock_returns,
    factor_data,
    factor_cols=['Mkt-RF', 'SMB', 'HML']
)

print(f"Market Beta: {exposures['Mkt-RF_beta']:.3f}")
print(f"SMB Beta: {exposures['SMB_beta']:.3f}")
print(f"HML Beta: {exposures['HML_beta']:.3f}")
print(f"R-squared: {exposures['r_squared']:.3f}")
```

---

## 📖 Theoretical Background

### Why Factor Models?

**CAPM's Limitations:**
- Single factor (market beta) cannot explain all return variations
- Market anomalies persist (size effect, value effect)
- Alpha (excess returns) cannot be explained by market risk alone
- Low R-squared (typically 0.2-0.4)

**Fama-French Solution:**
- Add systematic risk factors beyond market risk
- Size factor (SMB): Small firms earn higher returns than large firms
- Value factor (HML): Value stocks outperform growth stocks
- These factors represent risk premiums, not market inefficiencies
- Higher R-squared (typically 0.4-0.6 for 3-factor model)

### Key Papers

- **Fama & French (1992)**: "The Cross-Section of Expected Stock Returns"
  - Documented size and value effects
  - Showed these effects persist after controlling for market beta

- **Fama & French (1993)**: "Common Risk Factors in the Returns on Stocks and Bonds"
  - Introduced the 3-factor model
  - Explained size and value as risk factors

- **Fama & French (2015)**: "A Five-Factor Asset Pricing Model"
  - Added profitability (RMW) and investment (CMA) factors
  - Extended the model to 5 factors

- **Carhart (1997)**: "On Persistence in Mutual Fund Performance"
  - Added momentum factor
  - Created 4-factor model (now commonly extended to 6-factor)

### Factor Premiums

**Historical Average Annual Premiums (US Market):**
- Market Risk Premium: ~7-8%
- Size Premium (SMB): ~2-3%
- Value Premium (HML): ~3-5%
- Profitability Premium (RMW): ~2-3%
- Investment Premium (CMA): ~2-3%
- Momentum Premium (MOM): ~8-10% (but highly volatile)

**⚠️ Note:** These premiums are time-varying and can be negative in some periods.

---

## ⚠️ Important Notes

### Data Availability
- Fama-French factor data is downloaded from Kenneth French's website
- Ensure internet connection when running scripts
- Data is automatically downloaded and cached

### Backtesting Methodology
- **Always separate train and test data** to prevent overfitting
- Use walk-forward validation to prevent look-ahead bias
- Consider transaction costs and liquidity constraints
- Test on multiple time periods and market conditions

### Model Limitations
- Factor models explain past returns but don't guarantee future performance
- Factor premiums can decay over time
- Overfitting is a risk when optimizing parameters
- Model assumes linear relationships (may not hold in all cases)

### Practical Considerations
- Factor exposure calculations require sufficient historical data (at least 1-2 years)
- Rebalancing frequency affects transaction costs
- Portfolio size and diversification matter
- Consider sector and geographic diversification
- Monitor factor stability over time

### Code Execution Notes
- All output files are saved in the respective chapter directories
- Charts and CSV files are generated automatically
- Make sure to have write permissions in the chapter directories

---

## 📝 Code Quality Standards

- ✅ Proper train/test data separation
- ✅ Walk-forward validation implementation
- ✅ Comprehensive error handling
- ✅ Detailed documentation
- ✅ Performance metrics calculation
- ✅ Risk management considerations
- ✅ Transaction cost modeling
- ✅ Timezone handling for data alignment

---

## 🔗 References

### Academic Papers
- Fama, E. F., & French, K. R. (1992). The cross-section of expected stock returns. *Journal of Finance*, 47(2), 427-465.
- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.
- Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1-22.
- Carhart, M. M. (1997). On persistence in mutual fund performance. *Journal of Finance*, 52(1), 57-82.
- Gibbons, M. R., Ross, S. A., & Shanken, J. (1989). A test of the efficiency of a given portfolio. *Econometrica*, 57(5), 1121-1152.

### Data Sources
- Kenneth French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- yfinance: https://github.com/ranaroussi/yfinance

---

## 📄 License

This code is provided as educational material for the Udemy course.

---

## 🤝 Contributing

If you find any issues or have suggestions, please open an issue or submit a pull request.

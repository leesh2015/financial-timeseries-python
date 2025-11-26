# Chapter 2: Kalman Filter

## Overview

This chapter demonstrates Kalman Filter implementation using two main approaches:
1. **Custom Implementation** - From-scratch implementation for educational purposes
2. **FilterPy** - Most popular Python Kalman Filter library

**Note:** PyKalman is another option available, but it's not included here due to computational cost vs. utility considerations.

## 📁 Structure

```
Chapter2.Kalman Filter/
├── 1.Custom Implementation/
│   ├── kalman_filter.py          # Custom Kalman Filter implementation
│   └── backtest_kalman.py        # Backtest using custom implementation
├── 2.FilterPy/
│   ├── kalman_filter_filterpy.py    # FilterPy implementation
│   └── backtest_filterpy.py          # Backtest using FilterPy
└── README.md
```

## 🎯 Applications

Each implementation demonstrates:

1. **Price Trend Estimation**
   - State: `[price, velocity]`
   - Observation: `price`
   - Purpose: Noise removal and trend extraction

2. **Time-Varying Beta Estimation**
   - State: `[alpha, beta]`
   - Observation: `TQQQ return = alpha + beta * NASDAQ return + noise`
   - Purpose: Dynamic tracking of index-ETF relationship

## 📚 Implementation Details

### 1. Custom Implementation

**Pros:**
- Educational value - understand Kalman Filter mechanics
- Full control over implementation
- No external dependencies (except numpy)

**Cons:**
- More code to maintain
- May have bugs or edge cases
- Less optimized

**Usage:**
```bash
cd "1.Custom Implementation"
python kalman_filter.py
python backtest_kalman.py
```

### 2. FilterPy

**Pros:**
- Most popular and well-documented
- Supports Extended Kalman Filter, Unscented Kalman Filter
- Active community and examples
- Part of "Kalman and Bayesian Filters in Python" book

**Cons:**
- Additional dependency

**Installation:**
```bash
pip install filterpy
```

**Usage:**
```bash
cd "2.FilterPy"
python kalman_filter_filterpy.py
python backtest_filterpy.py
```

## 🔍 Comparison

| Feature | Custom | FilterPy |
|---------|--------|----------|
| **Ease of Use** | Medium | ⭐⭐⭐⭐⭐ |
| **Documentation** | Low | ⭐⭐⭐⭐⭐ |
| **Flexibility** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Performance** | Medium | High |
| **Time-Varying Beta** | ✅ Easy | ✅ Easy |
| **Smoothing** | ❌ | ✅ |
| **Parameter Learning** | ❌ Manual | ⚠️ Adaptive |
| **Unique Features** | Educational | Adaptive Noise |

**Note:** PyKalman is another option with EM algorithm support, but it's not included here due to computational cost vs. utility considerations.

## 🎯 Key Differences (Why They Perform Differently)

### 1. Custom Implementation
- **Fixed Parameters**: Uses hardcoded Q and R values
- **Purpose**: Educational - understand the algorithm
- **Performance**: Baseline for comparison

### 2. FilterPy
- **Adaptive Noise Estimation**: Dynamically adjusts Q and R based on prediction/innovation errors
- **Advantage**: Adapts to changing market conditions in real-time
- **Performance**: Better in volatile markets due to adaptive nature


## 💡 Recommendations

1. **For Learning**: Start with Custom Implementation
2. **For Production**: Use FilterPy (most popular and reliable, adaptive noise works well)
3. **For Rolling Window Backtesting**: FilterPy (adaptive) > Custom (fixed parameters)

## 📊 Backtest Strategy

All backtest implementations use the same strategy:

- **Signal Generation**: Based on filtered trend and price deviation
- **Buy Signal**: When filtered trend is positive AND price is below filtered price
- **Sell Signal**: When filtered trend is negative OR price is above filtered price
- **Hold**: Otherwise

**Key Features:**
- Walk-forward approach (no look-ahead bias)
- 70% training, 30% testing split
- Minimal transaction costs (0.02%)
- No slippage assumption

## ⚠️ Notes

### Performance Differences (Backtesting Results)

**Expected Performance Characteristics:**

1. **Custom**: Baseline performance with fixed parameters
   - Consistent, predictable results
   - Good for understanding the algorithm

2. **FilterPy (Adaptive)**: Typically shows better performance
   - Adaptive noise estimation adapts to market volatility
   - Better risk-adjusted returns in volatile markets
   - **Run backtest to see actual results**: `cd "2.FilterPy" && python backtest_filterpy.py`

**Key Insights:**
- **Not all methods need to be optimal**: Each has different strengths
- **FilterPy's adaptive noise** typically works well in backtesting (adapts to market volatility)
- **Run the backtests yourself** to see performance on your data and time period

### Why Implementations Differ

1. **Custom**: Fixed parameters → consistent baseline
2. **FilterPy**: Sage-Husa adaptive noise → adapts Q and R based on prediction/innovation errors

**Note:** PyKalman is another option with EM algorithm support, but it's not included here due to computational cost vs. utility considerations for rolling window backtesting.

### When to Use Each

- **Learning/Education**: Custom Implementation (understand the algorithm)
- **Production Trading**: FilterPy (adaptive noise adapts to market conditions)
- **Real-time Systems**: FilterPy (adaptive noise estimation, fast computation)

### Important Notes

- **Adaptive vs Fixed**: FilterPy's adaptive approach shows clear advantage in backtesting
- **Not All Methods Are Equal**: Each package has different design goals and strengths
- **Context Matters**: What works in backtesting may differ from what works in production

## 📖 References

- **FilterPy**: https://github.com/rlabbe/filterpy
- **Kalman Filter Book**: "Kalman and Bayesian Filters in Python" by Roger Labbe
- **Note**: PyKalman (https://pykalman.github.io/) is another option but not included here due to computational cost vs. utility considerations


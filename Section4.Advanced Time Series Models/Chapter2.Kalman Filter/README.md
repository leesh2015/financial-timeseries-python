# Chapter 2: Kalman Filter

## Overview

This chapter demonstrates state-space estimation with four complementary approaches:
1. **Custom Implementation** - From-scratch Kalman filter for educational purposes
2. **FilterPy** - Production-ready Kalman filter with adaptive noise and EKF/UKF support
3. **PyKalman EM** - Log-likelihood maximization with PyKalman’s EM routine
4. **Particle Filter** - Monte-Carlo approach for non-Gaussian, non-linear tracking

## 📁 Structure

```
Chapter2.Kalman Filter/
├── 1.Custom Implementation/
│   ├── kalman_filter.py          # Custom Kalman Filter implementation
│   └── backtest_kalman.py        # Backtest using custom implementation
├── 2.FilterPy/
│   ├── kalman_filter_filterpy.py    # FilterPy implementation
│   └── backtest_filterpy.py          # Backtest using FilterPy
├── 3.PyKalman_EM/
│   └── pykalman_em_demo.py        # EM-based alpha/beta smoothing with PyKalman
├── 4.ParticleFilter/
│   └── particle_filter_demo.py    # Bootstrap particle filter for dynamic beta
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
> 이 구현은 `[alpha, beta]`를 상태로 두고 TQQQ vs NASDAQ 관계를 추정합니다.  
> 백테스트에서는 베타 분포의 하위/상위 분위수를 신호 트리거로 사용합니다.

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
> FilterPy 버전도 동일한 `[alpha, beta]` 상태를 추정하되, 선택적으로 적응형 잡음 업데이트를 적용하고 EKF/UKF로 확장할 수 있습니다.

### 3. PyKalman EM

Leverages `pykalman` to learn transition/observation covariances via EM and smooth the dynamic alpha/beta relationship.

**Install dependency**
```bash
pip install pykalman
```

**Usage**
```bash
cd "3.PyKalman_EM"
python pykalman_em_demo.py
```

### 4. Particle Filter

Implements a lightweight bootstrap particle filter (systematic resampling) to handle non-Gaussian noise or heavier state jumps.

**Usage**
```bash
cd "4.ParticleFilter"
python particle_filter_demo.py
```

## 🔍 Comparison

| Feature | Custom | FilterPy | PyKalman EM | Particle Filter |
|---------|--------|----------|-------------|-----------------|
| **Ease of Use** | Medium | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Documentation** | Low | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium |
| **Flexibility** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Linear Gaussian | Non-linear / non-Gaussian |
| **Performance** | Medium | High | High (if model fits) | Depends on particles |
| **Time-Varying Beta** | ✅ | ✅ | ✅ (with credible bands) | ✅ (percentile bands) |
| **Smoothing** | ❌ | ✅ | ✅ (Rauch smoother) | ✅ (posterior samples) |
| **Parameter Learning** | Manual | Adaptive noise | ✅ EM | Implicit via resampling |
| **Unique Features** | Educational | EKF/UKF + adaptive Q/R | Log-likelihood EM | Monte-Carlo inference |

## 🎯 Key Differences (Why They Perform Differently)

### 1. Custom Implementation
- Fixed Q/R → deterministic baseline

### 2. FilterPy
- Adaptive noise (Sage-Husa) + EKF/UKF hooks

### 3. PyKalman EM
- Learns Q/R by maximizing log-likelihood, provides smoothed states + credible intervals

### 4. Particle Filter
- Handles non-linearities and fat-tailed noise via resampling-based posterior approximation


## 💡 Recommendations

1. **First Principles**: Start with Custom Implementation
2. **Production Kalman**: FilterPy (adaptive, EKF/UKF)
3. **Parameter Learning / Credible Bands**: PyKalman EM demo
4. **Non-Gaussian / Stress Testing**: Particle Filter demo

## 📊 Backtest Strategy

The backtest scripts (`backtest_kalman.py`, `backtest_filterpy.py`, `backtest_pykalman.py`, `backtest_particle.py`) share the same walk-forward strategy:

- **Signal Generation**: Use yesterday의 동적 베타가 분위수 밴드(기본 25/75%) 바깥인지 확인
- **Buy Signal**: `beta < lower_quantile` (레버리지 ETF가 덜 민감 → 저평가 구간)
- **Sell Signal**: `beta > upper_quantile` (베타 과열/과민 반응 구간)
- **Hold**: Otherwise

**Key Features:**
- Walk-forward approach (no look-ahead bias)
- 70% training, 30% testing split
- Minimal transaction costs (0.02%)
- No slippage assumption

## ⚠️ Notes

### Performance Differences & Notes

- **Custom**: Baseline performance with fixed parameters. Consistent, predictable, great for intuition.
- **FilterPy (Adaptive)**: Typically shows better risk-adjusted returns thanks to adaptive Q/R. Run `backtest_filterpy.py` to benchmark.
- **PyKalman EM**: Focused on state-estimation quality—use it to study smoothed alpha/beta + log-likelihood diagnostics rather than PnL.
- **Particle Filter**: Demonstrates robustness under fat tails or regime jumps; extend with your own trading rules if desired.

**Key Insight** – Pick the tool that matches your modelling goal (intuition, production, parameter learning, or stress testing). Running the scripts on your own data horizon is strongly recommended.

## 📖 References

- **FilterPy**: https://github.com/rlabbe/filterpy
- **PyKalman**: https://pykalman.github.io/
- **Kalman Filter Book**: "Kalman and Bayesian Filters in Python" by Roger Labbe


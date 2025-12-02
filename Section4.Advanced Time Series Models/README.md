---

# Section 4: Advanced Time Series Models

## Overview

This section collects the advanced methodologies that complement the core Udemy curriculum. Every chapter demonstrates a different modeling philosophy—probabilistic, statistical, deep-learning, and dependency-driven—while enforcing the same engineering principles:

- Walk-forward training/backtesting (no look-ahead bias)
- Clear separation of train, validation, and signal-generation code paths
- Production-safety considerations such as logging, retries, fallback logic, and metadata export

> **Status**: 🚧 The code is production-ready, while the Udemy lecture content for this section is still being finalized.

## Learning Objectives

1. Derive and implement state-space models to track time-varying relationships (β(t)) between assets.
2. Build Kalman filters from scratch and with FilterPy, including adaptive noise tuning.
3. Operate Prophet for seasonality-aware forecasts with configurable retraining cadence.
4. Train LSTM classifiers that avoid data leakage and handle class imbalance via class weights and optimal threshold search.
5. Deploy tree-based binary classifiers (XGBoost) with rich technical features, class imbalance handling, and ROC-optimized thresholds.
6. Generate wavelet-based multi-resolution features for volatility clustering and denoising.
7. Model joint distributions with Gaussian / Clayton copulas, simulate tail scenarios, and feed them into risk-aware backtests.

## Repository Structure

```
Section4.Advanced Time Series Models/
├── Chapter1.State-Space Models/
├── Chapter2.Kalman Filter/
│   ├── 1.Custom Implementation/
│   └── 2.FilterPy/
├── Chapter3.Prophet Model/
├── Chapter4.Deep Learning/
├── Chapter5.Tree-Based ML/
├── Chapter6.Wavelet Transform/
├── Chapter7.Copula Models/
└── utils/
```

Each chapter contains:
- A descriptive/forecast script (`*_forecast.py`, `*_analysis.py`, etc.)
- A `backtest_*` script for walk-forward simulation
- Optional metadata or model artifacts saved under the chapter (excluded from Git via `.gitignore`)

---

## Chapter Details

### Chapter 1 — State-Space Models
**Theory**:
- TQQQ returns are modeled as \\(r_{t}^{TQQQ} = \alpha_t + \beta_t r_{t}^{NASDAQ} + \epsilon_t\\) with \\(\beta_t\\) evolving as a random walk.
- Rolling covariance/variance approximates the state update; this mirrors a simplified Kalman smoother.

**Implementation**:
- `state_space_model.py` computes rolling β, α, and tracking error using vectorized pandas operations.
- `backtest_state_space.py` converts β deviations into mean-reversion signals. All signals are shifted by one period so “today’s trade” only uses information up to **t-1**.

**Backtest Principle**:
- When β < lower threshold ⇒ TQQQ under-leveraged ⇒ long signal.
- When β > upper threshold ⇒ over-leveraged ⇒ exit or short (configurable).

### Chapter 2 — Kalman Filter
**Theory**:
- Linear state-space: \\(x_{t} = Fx_{t-1} + w_t\\), \\(z_t = H x_t + v_t\\) with Gaussian noise. In this chapter the hidden state is \\([\alpha_t, \beta_t]\\), so the entire module centers on dynamic beta tracking rather than raw price smoothing.

**Implementation**:
- `1.Custom Implementation/`: hand-written Kalman loop with fixed Q/R (very smooth beta path).
- `2.FilterPy/`: same model, but FilterPy plus Sage–Husa adaptive noise so Q/R widen/narrow based on innovation variance.
- `3.PyKalman_EM/`: uses `pykalman` EM to learn Q/R automatically before smoothing.
- `4.ParticleFilter/`: bootstrap particle filter for \\(\beta_t\\) when Gaussian/linear assumptions break.

**Backtest**:
- Every implementation outputs a beta series that feeds a quantile-based switching strategy (low quantile ⇒ long, high quantile ⇒ exit/sell); quantile bands differ per model (tight for custom, mid for FilterPy/PyKalman, wide for Particle Filter).
- Signals are shifted by one bar to eliminate look-ahead; walk-forward backtests live alongside each folder (`backtest_kalman.py`, `backtest_filterpy.py`, `backtest_pykalman.py`, `backtest_particle.py`).

### Chapter 3 — Prophet Model
**Theory**:
- Prophet decomposes time series into trend, seasonality, and holiday components via Bayesian additive regression.

**Implementation**:
- `prophet_forecast.py` builds the base model. `backtest_prophet.py` maintains a cached forecast dictionary for the next `forecast_horizon` days.
- Retraining occurs every `retrain_interval` observations or when cache is empty.

**Backtest**:
- If forecasted mean > last price by `price_threshold` **and** lower bound > last price ⇒ long.
- If forecasted mean < last price by `price_threshold` **and** upper bound < last price ⇒ sell.
- Signals are delayed by one bar to prevent same-bar execution.

### Chapter 4 — Deep Learning (LSTM)
**Theory**:
- Sequence-to-label classification where inputs are rolling log returns, normalized via MinMaxScaler fitted only on the training partition.
- Class weights + ROC-based threshold selection (Youden’s J statistic) mitigate class imbalance.

**Implementation**:
- `lstm_forecast.py` trains models per ticker, saves `.h5` weights and metadata (lookback, optimal threshold, ROC metrics).
- `backtest_lstm.py` loads metadata, scales incoming returns using the training scaler, and emits probabilistic signals.

**Backtest**:
- Probabilities are compared against the stored optimal threshold.
- Signals are shifted by one day; no trade occurs until enough history (lookback) accumulates.

### Chapter 5 — Tree-Based ML (XGBoost)
**Theory**:
- Gradient boosted trees perform **binary classification** (up/down direction) using rich technical features (extended lags, rolling statistics, RSI, momentum, volatility).
- Class imbalance is handled via `scale_pos_weight` and balanced class weights.
- Optimal classification threshold is determined using ROC curve analysis (Youden's J statistic).

**Implementation**:
- `xgboost_forecast.py` trains binary classifiers per asset, saves `.pkl` model + metadata JSON (best iteration, accuracy/F1/ROC-AUC, optimal threshold, feature names).
- Features include extended lag windows (1, 2, 3, 5, 10, 20, 30, 60), multiple rolling statistics, and technical indicators.
- Highly correlated features (correlation > 0.95) are automatically removed to reduce noise.
- Two-stage training: minimum iterations first, then early stopping to ensure sufficient learning.

**Backtest**:
- `backtest_xgboost.py` loads pre-trained model and metadata, precomputes feature matrix once, shifts by one day to prevent look-ahead.
- Buy when predicted probability of up > optimal threshold (from training).
- Sell when predicted probability of up < (1 - optimal threshold).
- Signals are shifted by one day to execute on the next bar.

### Chapter 6 — Wavelet Transform
**Theory**:
- Discrete wavelet transforms (Haar/Daubechies) separate series into approximation and detail coefficients, allowing noise removal or multi-scale analysis.

**Implementation**:
- `wavelet_analysis.py` applies transformations, reconstructs denoised prices, and exports diagnostic plots.
- `backtest_wavelet.py` uses smoothed vs. original price differentials to craft simple entry/exit logic.

**Backtest**:
- Signals focus on mean reversion between raw price and low-frequency component.
- Adjustable thresholds determine when the spread is considered “stretched.”

### Chapter 7 — Copula Models
**Theory**:
- Copulas capture dependence structure separately from marginals (Sklar’s theorem). Clayton copula, for example, models lower-tail dependence.

**Implementation**:
- `copula_analysis.py` fits Gaussian/Clayton copulas to normalized returns, provides QQ plots and tail probability diagnostics.
- `backtest_copula.py` simulates joint outcomes and triggers trades when simulated correlation/tail events predict divergence/convergence.

**Backtest**:
- Monte Carlo scenarios are generated each step to estimate conditional probabilities of joint moves.
- Trade triggers when probability of a certain tail event exceeds configurable bands.

---

## Common Backtest Methodology

1. **Train/Test Split** – Default is 70/30 chronological split before backtesting begins.
2. **Walk-Forward Execution** – During the test period, models re-train or update only with past data.
3. **Signal Delay** – All strategies shift signals by one step to ensure the trade uses information available strictly up to t-1.
4. **Metrics** – Equity curves, Sharpe, drawdown, win-rate, etc., are produced by `utils/backtest.py`.
5. **Artifact Handling** – Large model files live under `ChapterX/.../models/` and are ignored by Git.

---

## Running the Examples

```bash
# Example: run state-space analysis
cd "Section4.Advanced Time Series Models/Chapter1.State-Space Models"
python state_space_model.py

# Example: run LSTM backtest
cd "Section4.Advanced Time Series Models/Chapter4.Deep Learning"
python backtest_lstm.py
```

All requirements are already listed in the root `requirements.txt` (statsmodels, prophet, tensorflow, xgboost, pywavelets, copulae, etc.).

---

## Shared Utilities (`Section4.Advanced Time Series Models/utils/`)

| File | Description |
|------|-------------|
| `backtest.py` | Simple backtest engine generating equity curve + metrics |
| `data_loader.py` | NASDAQ/TQQQ download helpers, alignment utilities |
| `__init__.py` | Declares the package for relative imports |

Each chapter imports utilities via:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
```
so scripts can run standalone from their own folders.

---

## Notes & Status

- Sections 4 and 5 remain **Udemy “in preparation”**: code is open-source now, video lessons are being edited.
- All examples are educational; re-validate before live deployment.
- `models/` directories are `.gitignore`d. Re-run the training scripts to regenerate weights locally.

Happy experimenting!


# Soft Penalty Hyperparameter Bayesian Optimization

The hypothesize-driven VECM confidence adjustment and position optimization techniques are a robust quantitative approach. By introducing **Soft Penalty (Fractional Position Reduction)**, we can push the mathematical limits of hard-coded heuristics.

## 🧪 1. Hypothesis
The current penalty weights for **Oil Momentum (3.0x)**, **Treasury Momentum (5.0x)**, and the **70% Penalty Cap** are heuristics derived from logical inference and historical observation.

**Hypothesis:** By employing Bayesian optimization (`optuna`), we can identify the "Golden Ratio" that minimizes opportunity costs (profits lost during normal bull market minor pullbacks) while more aggressively reducing exposure during structural collapses (false positives). This should simultaneously improve the Sharpe Ratio and reduce MDD.

## ⚙️ 2. Search Space
The `optimize_soft_penalty.py` script explores the following key parameters:

1. `cl_multiplier` (Oil Penalty Multiplier): **1.0 ~ 10.0** (Baseline: 3.0)
2. `zb_multiplier` (Treasury Penalty Multiplier): **1.0 ~ 10.0** (Baseline: 5.0)
3. `max_penalty_cap` (Total Macro Penalty Limit): **0.30 ~ 0.90** (Baseline: 0.70)
4. `stag_penalty_floor` (Structural Shock Minimum Floor): **0.30 ~ 0.80** (Baseline: 0.50)
5. `ect_penalty_multiplier` (Error Correction Penalty): **1.0 ~ 10.0**
6. `ect_bonus_multiplier` (Error Correction Bonus): **0.1 ~ 5.0**

## 🏗️ 3. Execution Workflow

### Step 1: High-Speed Backtesting Infrastructure
Running the full `dynamic_simulation_rl.py` takes several minutes per trial, making hundreds of trials impossible. We use `precompute_signals.py` to generate a `precomputed_signals.csv` (daily VECM forecasts and macro momentum). The optimizer then runs simulations in milliseconds.

### Step 2: Bayesian Optimization (Optuna)
The `objective` function dynamically adjusts the Soft Penalty logic and maximizes a composite **Quality Score**:
`Score = (Sharpe * Return) / (MDD^2)`
This prioritizes risk-adjusted returns while heavily punishing large drawdowns.

### Step 3: Production Integration
The best-performing multipliers are identified and can be hard-coded into `rl_agent.py` or passed as dynamic configurations for the production environment.

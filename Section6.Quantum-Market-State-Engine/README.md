# Section 6: Quantum Market State Engine (Financial Demon)

Welcome to the **Quantum Market State Engine**. This section introduces a paradigm shift in quantitative trading. We completely abandon the illusion of "physical time" (like 1-minute candles or moving averages) and reconstruct the market as a **Quantum Fluid Dynamics** system driven purely by causality and event density.

## Core Philosophy

### 1. Moving Averages are Dead (Event-Time vs Clock-Time)
Traditional finance relies on arbitrary time intervals (e.g., closing price at 15:00). However, the market does not care about human clocks. In this engine, time only flows when an *event* occurs (a Quantum Jump). We track the probability distribution of future prices after $n$ jumps, completely decoupling from clock time.

### 2. The Financial Planck Constant
To remove arbitrary subjectivity, we rely only on the market's absolute structural constants:
*   **Space Quantization:** Minimum Tick Size
*   **Energy Quantization:** Minimum Trade Unit
All calculations (Kinetic Energy $T$, Potential Energy $V$) are normalized into integer multiples of this Financial Planck Constant ($h_f$).

### 3. Matrix Mechanics & Adaptive Dimensions
The engine dynamically resizes its unitary transition matrix (from 5x5 to 10x10) based on energy density ($T/V$). 
*   When energy bursts ($T > 0.8V$), the matrix expands to capture large volatility waves.
*   When energy dissipates ($T < 0.2V$ for 3 cycles), the matrix contracts to optimize CPU resources while preserving causality.

### 4. The Probability Dial (Calibration Control)
We introduce the **Probability Dial (Threshold Control)**. Instead of following a fixed signal, users can now dictate their desired win-rate quality. By adjusting the `--threshold` parameter, the engine filters out only the energy fragments that meet your specific probabilistic requirements (e.g., 83% confidence).

---

## Directory Structure

*   **`data/`**: Contains `QQQ_jump_sample.parquet` and `TQQQ_jump_data_20260422.parquet`, lossless accumulations of jump events.
*   **`core/`**: 
    *   `quantum_models.py`: Dataclasses for `KineticImpact` and `PotentialEnergy`.
    *   `quantum_engine.py`: The heart of the engine (Hamiltonian calculation, Virtual Potential, Matrix generation).
*   **`scripts/`**: Executable scripts for backtesting and live simulation.

---

## Quick Start Guide

You don't need any complex broker API connections to test this engine. We have provided lossless data samples and a mock streaming server.

### 1. Run the Quantum Backtester
To see the mathematical proof of the engine's predictive power on 3x Leveraged ETFs (TQQQ):
```bash
cd scripts
# Standard Testing
python quantum_backtester.py --file ../data/TQQQ_jump_data_20260422.parquet

# High-Confidence Dial (94% Win-Rate Proof)
python quantum_backtester.py --file ../data/TQQQ_jump_data_20260422.parquet --threshold 0.83
```

### 2. Run the Live Simulation (Mock Streaming)
To experience the "Live" Quantum Predictor UI, you need two terminal windows.

**Terminal 1 (Start the Data Streamer):**
This script reads the parquet file and streams it via a local TCP socket, simulating a live WebSocket feed.
```bash
cd scripts
python mock_collector.py
```

**Terminal 2 (Start the Predictor UI):**
This connects to the mock stream, processes the Hamiltonian zero-latency loop, and renders the probability density function (PDF) and matrix expansion/contraction.
```bash
cd scripts
python quantum_predictor.py --threshold 0.83
```
*(Note: You will need the `rich` library installed: `pip install rich`)*

---

## Educational Resources
This codebase is part of the comprehensive Udemy Course: **"Advanced Agentic Coding & Quantum Market States"**.
*   Learn the detailed mathematics behind Eigenvalue Flux.
*   Watch YouTube Shorts visualizing the Matrix Mechanics.
*   Build your own Event-Driven RL agents on top of this PDF generation engine.

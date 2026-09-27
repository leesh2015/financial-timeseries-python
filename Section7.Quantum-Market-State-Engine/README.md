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

### 3. Matrix Mechanics & Symmetric Parity (v2.3)
The engine dynamically resizes its unitary transition matrix (from 5x5 to 10x10) based on energy density ($T/V$). 
*   **Symmetric Parity**: In v2.3, the engine implements dimension-adaptive probability summation. It removes the mathematical bias in even/odd dimensions by ensuring perfectly symmetric probability splits.
*   **Expansion/Contraction**: When energy bursts ($T > 0.8V$), the matrix expands. When energy dissipates, it contracts to preserve causality while optimizing computational resources.

### 4. Zero-Lag Architecture
The v2.3 update introduces the **Zero-Lag** sync mechanism. The engine derives the physical horizon ($N$) *after* updating the matrix to the most current market state, eliminating the 1-jump lag inherent in previous versions.

---

## Directory Structure

*   **`data/`**: Contains `QQQ_jump_sample.parquet`, a mathematically pure, lossless accumulation of jump events.
*   **`core/`**: 
    *   `quantum_models.py`: Dataclasses for `KineticImpact` and `PotentialEnergy`.
    *   `quantum_engine.py`: The heart of the engine (Hamiltonian calculation, Virtual Potential, Matrix generation).
*   **`scripts/`**: Executable scripts for backtesting and live simulation.

---

## Quick Start Guide

You don't need any complex broker API connections to test this engine. We have provided a lossless data sample (`QQQ_jump_sample.parquet`) and a mock streaming server.

### 1. Run the Quantum Backtester
To see the mathematical proof of the engine's predictive power (incorporating dynamic thresholds and harsh slippage penalties):
```bash
cd scripts
python quantum_backtester.py
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
python quantum_predictor.py
```
*(Note: You will need the `rich` library installed: `pip install rich`)*

---

## Educational Resources
This codebase is part of the comprehensive Udemy Course: **"Advanced Agentic Coding & Quantum Market States"**.
*   Learn the detailed mathematics behind Eigenvalue Flux.
*   Watch YouTube Shorts visualizing the Matrix Mechanics.
*   Build your own Event-Driven RL agents on top of this PDF generation engine.

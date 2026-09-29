# Quantum Canonical Microstructure: Open-System Dynamics (Zenodo: 10.5281/zenodo.23042820)

[English](README.md) | [한국어](README_KR.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23042820.svg)](https://doi.org/10.5281/zenodo.23042820)
[![Course](https://img.shields.io/badge/Udemy-Section%208-green.svg)](https://www.udemy.com/instructor/course/6343207/manage/curriculum)
[![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

Official open-source educational and academic replication repository for the research paper:
> **"A Quantum Canonical Framework for Order Book Dynamics: Proper-Time Relaxation in Non-Equilibrium Markets"**  
> *Author:* Sanghyeok Lee (Independent Quantitative Research / Hilbert Labs)  
> *Target Categories:* Quantitative Finance (`q-fin.ST`, `q-fin.TR`), Mathematical Physics (`math-ph`)  
> *DOI:* [10.5281/zenodo.23042820](https://doi.org/10.5281/zenodo.23042820) | *Zenodo Record:* [https://zenodo.org/records/23042820](https://zenodo.org/records/23042820)

---

## 1. Mathematical Architecture & Paper Mapping

This repository provides a parameter-free, strictly closed reference implementation mapping the 5 core sections of the paper:

| Section in Paper | Mathematical Formalism | Module in `core/` | Description |
| :--- | :--- | :--- | :--- |
| **Section 1.1** | $h_f \equiv \Delta p_{\min} \cdot \Delta v_{\min}$ | [`core/planck_lattice.py`](core/planck_lattice.py) | **Financial Planck Constant**: Invariant phase-space cell bounding discrete order book lattices. |
| **Section 2.1 & 5.2** | $\hat{H} = \hat{T} + \hat{V}_{\text{eff}}$, $\hat{U}(d\tau) \in \mathrm{SU}(2)$ | [`core/canonical_hamiltonian.py`](core/canonical_hamiltonian.py) | **Effective Hamiltonian**: Multi-level depth potential wells, kinetic order flow momentum, & exact unitary rotation. |
| **Section 3.1 & 5.3** | $\frac{d\rho}{d\tau} = -i[\hat{H}, \rho] + \sum \mathcal{D}[\hat{L}_m]\rho$ | [`core/lindblad_dissipation.py`](core/lindblad_dissipation.py) | **Lindbladian Dissipation**: Asymmetric population pumping, phase dephasing, & Cauchy-Schwarz boundary defense ($|\rho_{01}| \le \sqrt{\rho_{00}\rho_{11}}$). |
| **Section 5.1** | $d\tau = dt \cdot \min(\max(0.05, \Phi / \bar{\Phi}), 20.0)$ | [`core/proper_time.py`](core/proper_time.py) | **Volume-Clocked Proper Time**: Relativistic dwell contraction and geodesic timescale evolution. |
| **Section 4 & 6.2** | $\Delta S(\tau) < -\epsilon_S$, $\vec{r} = (r_x, r_y, r_z)^T$ | [`core/engine.py`](core/engine.py) | **Thermodynamic Phase Gating**: Bloch sphere projection, von Neumann entropy collapse, & Golden Windows. |

---

## 2. Directory Structure

```text
quantum_lecture/
├── core/
│   ├── __init__.py
│   ├── planck_lattice.py         # Financial Planck Constant (h_f) lattice quantization
│   ├── canonical_hamiltonian.py  # Virtual potential field, kinetic impulse & SU(2) unitary step
│   ├── lindblad_dissipation.py   # Open quantum system jump dissipation & Bloch thermodynamics
│   ├── proper_time.py            # Volume-clocked proper time (dtau) & Golden Window gater
│   └── engine.py                 # Unified QuantumCanonicalEngine pipeline
├── notebooks/
│   └── ssrn_paper_replication.ipynb # End-to-end paper simulation, quench replication & figures
├── tests/
│   ├── __init__.py
│   └── test_su2_invariance.py    # Theorem 1 (SU(2) Trace Preservation & Hermiticity) unit tests
├── CURRICULUM.md                 # Udemy Course Section 8 Detailed Lecture Plan (Lectures 39~44)
├── requirements.txt              # Ultra-lightweight dependencies (numpy, scipy, matplotlib, pandas)
└── README.md                     # Comprehensive academic guide
```

---

## 3. Quick Start

### Installation
```bash
git clone https://github.com/leesh2015/financial-timeseries-python.git
cd Section8.Quantum-Canonical-Microstructure
pip install -r requirements.txt
```

### Running the Mathematical Invariance Tests (Theorem 1)
```bash
python -c "import tests.test_su2_invariance as t; t.test_su2_unitary_invariance(); print('Theorem 1 Verified!')"
```

### Interactive Simulation & Visualizations
Launch the replication notebook:
```bash
jupyter notebook notebooks/ssrn_paper_replication.ipynb
```

---

## 4. Citation (BibTeX)

If you utilize this canonical open-system framework or reproduction codebase in your academic research or quantitative trading systems, please cite:

```bibtex
@article{lee2026quantumcanonical,
  title={A Quantum Canonical Framework for Order Book Dynamics: Proper-Time Relaxation in Non-Equilibrium Markets},
  author={Lee, Sanghyeok},
  journal={Zenodo Preprint},
  year={2026},
  month={Sep},
  version={v1.0},
  doi={10.5281/zenodo.23042820},
  url={https://doi.org/10.5281/zenodo.23042820}
}
```

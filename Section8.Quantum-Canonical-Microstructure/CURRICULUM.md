# 섹션 8: Quantum Canonical Microstructure & Open-System Relaxation (DOI: 10.5281/zenodo.23042820)

[English](CURRICULUM_EN.md) | [한국어](CURRICULUM.md)

> **강의명**: 물리학 제1원리 기반 금융 고빈도 퀀트 트레이딩 마스터클래스  
> **공식 논문**: *A Quantum Canonical Framework for Order Book Dynamics: Proper-Time Relaxation in Non-Equilibrium Markets* ([DOI: 10.5281/zenodo.23042820](https://doi.org/10.5281/zenodo.23042820))  
> **실습 리포지토리**: `Section8.Quantum-Canonical-Microstructure` (`quantum_lecture`)  
> **핵심 패키지**: `core/planck_lattice.py`, `core/canonical_hamiltonian.py`, `core/lindblad_dissipation.py`, `core/proper_time.py`, `core/engine.py`

---

## 🎯 섹션 8 마스터 목표
기존 섹션 7의 이산 라플라스 악마(격자 마코프) 모델의 한계를 극복하고, 개방 양자계(Open Quantum Systems)의 제1원리를 적용합니다. 실시간 선물 오더북을 **금융 플랑크 상수($h_f$)**, **린드블라드 마스터 방정식**, **볼륨 클록 고유시간($d\tau$)**, 그리고 **폰 노이만 엔트로피 붕괴(골든 윈도우)**로 해석하는 최첨단 수리물리학적 퀀트 엔진을 직접 구축합니다.

---

## 📚 세부 강의 커리큘럼 (강의 39 ~ 44)

```
[강의 39: 닫힌계의 한계와 플랑크 상수] ──> [강의 40: 유효 해밀토니안 & SU(2) 회전]
                                                              │
[강의 42: 볼륨 클록 고유시간과 체류 척도] <── [강의 41: 린드블라드 개방계 & 탈위상]
      │
[강의 43: 폰 노이만 엔트로피 & 골든 윈도우] ──> [강의 44: Zenodo 논문 (DOI: 10.5281/zenodo.23042820) 재현]
```

---

### 📖 강의 39: The Financial Planck Constant: Invariant Lattice Discretization
*(금융 플랑크 상수: 불변 격자 이산화와 상태 공간)*
- **학습 목표**: 시계 시간(Clock-time) 샘플링의 자의성을 배제하고 금융의 기본 양자 셀 $h_f = \Delta p_{\min} \cdot \Delta v_{\min}$을 수학적으로 정의합니다.
- **물리 제1원리**: 연속 공간의 정준 양자화(Canonical Quantization) 및 위상 공간 최소 체적 셀.
- **코드 실습**: `core/planck_lattice.py` (`PlanckLattice`, `quantize_order_book`)

---

### 📖 강의 40: Effective Hamiltonian Dynamics & The SU(2) Unitary Step
*(유효 해밀토니안 동역학: 가상 포텐셜 우물과 SU(2) 유니터리 회전)*
- **학습 목표**: 오더북 누적 잔량을 포텐셜 우물($V_{\text{eff}}$)로, 체결 틱 충격을 운동량($\Delta$)으로 사영하여 유효 해밀토니안을 구축하고, 정리 1($\mathrm{SU}(2)$ 불변성)을 증명합니다.
- **물리 제1원리**: 해밀토니안 보존계 $\hat{H} = \hat{T} + \hat{V}$, 브라-켓 힐베르트 회전, 대각합(Trace) 보존.
- **코드 실습**: `core/canonical_hamiltonian.py` (`CanonicalHamiltonianBuilder`, `unitary_step`) & `tests/test_su2_invariance.py`

---

### 📖 강의 41: Open Quantum Systems: Lindblad Dissipation & Phase Dephasing
*(개방 양자계: 린드블라드 소산과 위상 탈결맞음)*
- **학습 목표**: 외부 자금 유입과 호가 취소가 존재하는 열린 시장을 린드블라드 마스터 방정식으로 모델링하고 코시-슈바르츠 물리적 경계 조건을 방어합니다.
- **물리 제1원리**: 밀도 행렬 $\rho$, 점프 연산자 $\hat{L}_m$, 종방향 에너지 완화($T_1$) 및 횡방향 탈위상($T_2$).
- **코드 실습**: `core/lindblad_dissipation.py` (`LindbladDissipator`, `apply_dissipation`)

---

### 📖 강의 42: Volume-Clocked Proper Time: Dynamic Dwell Geodesics
*(볼륨 클록 고유시간: 정보 플럭스와 체류 척도의 상대론적 수축)*
- **학습 목표**: 거래량 플럭스($\Phi$)에 따라 고유시간($d\tau$)을 가속/감속하고, 시장 충격 시 고유 체류 시간 척도($\tau$)가 자동 수축하는 메커니즘을 구현합니다.
- **물리 제1원리**: 일반상대론의 고유시간 측지선 방정식, 국소 정보 밀도 기반 시간 지연/수축.
- **코드 실습**: `core/proper_time.py` (`ProperTimeClock`, `update_proper_time`)

---

### 📖 강의 43: Non-Equilibrium Phase Gating: Entropy Drop & Golden Windows
*(비평형 상전이: 폰 노이만 엔트로피 붕괴와 골든 샌드위치 윈도우)*
- **학습 목표**: 기저 열적 혼합 상태($S \to \ln 2$)에서 국소적 엔트로피 급감($\Delta S < -\epsilon_S$)을 감지하여 비평형 1차 상전이 구간을 포착합니다.
- **물리 제1원리**: 3D 블로흐 구면 벡터 $\vec{r}$, 폰 노이만 정보 순도 $\mathcal{P}_{\text{info}}$, 정준 분배 함수 $\mathcal{Z}(\beta)$.
- **코드 실습**: `core/proper_time.py` (`GoldenWindowPhaseGater`) & `core/engine.py`

---

### 📖 강의 44: Full Zenodo Paper Replication & Visualizations (DOI: 10.5281/zenodo.23042820)
*(Zenodo 논문 (DOI: 10.5281/zenodo.23042820) 원천 재현: 파이프라인 통합 및 실습)*
- **학습 목표**: 논문 제6절의 고빈도 선물 실증 데이터 시뮬레이션을 한 번에 실행하고 Figure 1(퀜치 곡선)과 Figure 2(열역학 지형도)를 재현합니다.
- **실습 환경**: `notebooks/paper_replication.ipynb` (또는 `ssrn_paper_replication.ipynb`) 주피터 인터랙티브 실습.

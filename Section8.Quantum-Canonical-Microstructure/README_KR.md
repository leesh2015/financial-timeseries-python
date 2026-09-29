# 양자 정준 미세구조: 개방계 동역학 (Zenodo: 10.5281/zenodo.23042820)

[English](README.md) | [한국어](README_KR.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23042820.svg)](https://doi.org/10.5281/zenodo.23042820)
[![Course](https://img.shields.io/badge/Udemy-Section%208-green.svg)](https://www.udemy.com/instructor/course/6343207/manage/curriculum)
[![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

학술 연구 논문의 공식 오픈소스 교육 및 재현 리포지토리:
> **"A Quantum Canonical Framework for Order Book Dynamics: Proper-Time Relaxation in Non-Equilibrium Markets"**  
> *저자:* 이상혁 (상임 퀀트 연구원 / 힐베르트 랩스)  
> *분류 분야:* 정량금융(`q-fin.ST`, `q-fin.TR`), 수리물리학(`math-ph`)  
> *DOI:* [10.5281/zenodo.23042820](https://doi.org/10.5281/zenodo.23042820) | *Zenodo Record:* [https://zenodo.org/records/23042820](https://zenodo.org/records/23042820)

---

## 1. 수리물리학적 구조 및 논문 매핑

본 리포지토리는 논문의 핵심 5대 섹션을 수식과 1:1로 대응시킨 파라미터-프리(Parameter-Free) 수학적 구현체를 제공합니다:

| 논문 섹션 | 수식 체계 | 구현 모듈 (`core/`) | 상세 설명 |
| :--- | :--- | :--- | :--- |
| **제1.1절** | $h_f \equiv \Delta p_{\min} \cdot \Delta v_{\min}$ | [`core/planck_lattice.py`](core/planck_lattice.py) | **금융 플랑크 상수**: 불연속 호가창 격자를 하한 바운딩하는 위상 공간의 최소 단위. |
| **제2.1절 & 5.2절** | $\hat{H} = \hat{T} + \hat{V}_{\text{eff}}$, $\hat{U}(d\tau) \in \mathrm{SU}(2)$ | [`core/canonical_hamiltonian.py`](core/canonical_hamiltonian.py) | **유효 해밀토니안**: 호가 잔량 포텐셜 우물, 시장가 체결 운동량 충격, 폐쇄형 $\mathrm{SU}(2)$ 유니터리 회전. |
| **제3.1절 & 5.3절** | $\frac{d\rho}{d\tau} = -i[\hat{H}, \rho] + \sum \mathcal{D}[\hat{L}_m]\rho$ | [`core/lindblad_dissipation.py`](core/lindblad_dissipation.py) | **린드블라드 소산**: 비대칭 상태 펌핑, 위상 탈결맞음(Dephasing), 코시-슈바르츠 경계 방어 ($|\rho_{01}| \le \sqrt{\rho_{00}\rho_{11}}$). |
| **제5.1절** | $d\tau = dt \cdot \min(\max(0.05, \Phi / \bar{\Phi}), 20.0)$ | [`core/proper_time.py`](core/proper_time.py) | **볼륨 클록 고유시간**: 정보 플럭스 기반 상대론적 시간 지연 및 체류 척도($\tau$)의 자동 수축. |
| **제4절 & 6.2절** | $\Delta S(\tau) < -\epsilon_S$, $\vec{r} = (r_x, r_y, r_z)^T$ | [`core/engine.py`](core/engine.py) | **열역학적 상전이 게이팅**: 3D 블로흐 구면 사영, 폰 노이만 엔트로피 붕괴 및 골든 샌드위치 윈도우. |

---

## 2. 디렉토리 구조

```text
quantum_lecture/
├── core/
│   ├── __init__.py
│   ├── planck_lattice.py         # 금융 플랑크 상수(h_f) 격자 이산화 모듈
│   ├── canonical_hamiltonian.py  # 가상 포텐셜 장벽, 운동량 충격 및 SU(2) 유니터리 연산자
│   ├── lindblad_dissipation.py   # 개방 양자계 점프 소산, 탈위상 감쇠 및 블로흐 열역학
│   ├── proper_time.py            # 볼륨 클록 고유시간(dtau) 및 골든 윈도우 위상 게이터
│   └── engine.py                 # 논문 전체 파이프라인 통합 QuantumCanonicalEngine
├── notebooks/
│   └── ssrn_paper_replication.ipynb # 논문 전수 시뮬레이션, 퀜치 곡선 및 Figure 1, 2 재현 노트북
├── tests/
│   ├── __init__.py
│   └── test_su2_invariance.py    # 정리 1 (Theorem 1: SU(2) 대각합 보존 정리) 단위 테스트
├── CURRICULUM.md                 # Udemy 섹션 8 상세 강의 계획서 (강의 39~44) - 한글 가이드
├── CURRICULUM_EN.md              # Udemy Section 8 Curriculum - English Guide
├── requirements.txt              # 초경량 필수 패키지 (numpy, scipy, matplotlib, pandas)
├── README.md                     # 영문 공식 설명서 (English Guide)
└── README_KR.md                  # 국문 공식 설명서 (Korean Guide)
```

---

## 3. 빠른 시작 (Quick Start)

### 의존성 설치
```bash
git clone https://github.com/leesh2015/financial-timeseries-python.git
cd Section8.Quantum-Canonical-Microstructure
pip install -r requirements.txt
```

### 수학적 불변성 검증 테스트 (정리 1)
```bash
python -c "import tests.test_su2_invariance as t; t.test_su2_unitary_invariance(); print('Theorem 1 Verified!')"
```

### 주피터 시뮬레이션 및 시각화 실행
```bash
jupyter notebook notebooks/ssrn_paper_replication.ipynb
```

---

## 4. 인용 양식 (BibTeX)

학술 연구나 상업용 알고리즘에 본 개방계 정준 프레임워크 또는 코드를 인용하실 경우 아래 양식을 사용해 주십시오:

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

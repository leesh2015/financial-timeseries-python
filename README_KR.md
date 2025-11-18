# Python을 활용한 금융 시계열 분석

[English](README.md) | [한국어](README_KR.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Udemy Course](https://img.shields.io/badge/Udemy-Course-orange)](https://www.udemy.com/course/mastering-financial-time-series-analysis-with-python/?referralCode=BA6CA9A3E5406E41359E&couponCode=KRLETSLEARNNOW)

Python을 활용한 금융 시계열 분석, 알고리즘 트레이딩 전략, 그리고 프로덕션 수준의 자동매매 시스템을 마스터하기 위한 종합 오픈소스 프로젝트입니다. 이 프로젝트는 이론적 지식과 실전 구현을 결합하며, **실제 자금으로 운영 중인 라이브 프로덕션 트레이딩 시스템**을 포함하고 있습니다.

## 🎓 Udemy 강의

**Mastering Financial Time Series Analysis with Python**

👉 [강의 등록하기](https://www.udemy.com/course/mastering-financial-time-series-analysis-with-python/?referralCode=BA6CA9A3E5406E41359E&couponCode=KRLETSLEARNNOW)

- **섹션 1 & 2**: 강의 자료가 완전히 업로드되어 제공됩니다
- **섹션 3**: 실시간 거래 기록이 있는 라이브 프로덕션 시스템

## 📊 실시간 거래 기록

**현재 라이브 프로덕션 운영 중**

이 프로젝트는 **실제 자금으로 활발히 거래 중인 프로덕션 트레이딩 시스템**을 포함합니다. 일일 거래 기록은 자동으로 다음 사이트에 업로드됩니다:

🌐 **[거래 기록 대시보드](https://leenaissance.com/trading-history)**

시스템은 증권사 API 연동을 통한 자동매매를 사용하며, 모든 거래는 웹사이트에 투명하게 표시됩니다.

## 📚 프로젝트 구조

### Section 1: 금융 시계열 분석

**상태**: ✅ 강의 제공 중

시계열 기초 및 고급 기법을 포괄적으로 다룹니다:

- **Chapter 1**: 시계열 데이터 분석 기초
  - 정상성과 비정상성
  - 차분 및 변환
  - 계절성 분해

- **Chapter 2**: 고급 시계열 분석
  - ADF (Augmented Dickey-Fuller) 검정
  - AR (자기회귀) 모델
  - PACF (부분자기상관함수) 분석
  - 랜덤 워크 이론

- **Chapter 3**: 단변량 시계열 분석
  - AR, MA, ARMA 모델
  - AIC vs BIC 모델 선택
  - Auto-ARIMA
  - 잔차 분석을 위한 Ljung-Box 검정

- **Chapter 4**: 고급 변동성 모델링 및 예측
  - ARCH 모델
  - GARCH 모델
  - ARIMA-GARCH 하이브리드 모델
  - 백테스팅 전략

- **Chapter 5**: 다변량 시계열 분석
  - VAR (벡터 자기회귀) 모델
  - VARMA 모델
  - Granger 인과관계 분석

- **Chapter 6**: 고급 다변량 시계열 분석
  - VECM (벡터 오차수정 모델)
  - Johansen 공적분 검정
  - VAR IRF (충격반응함수)
  - VAR FEVD (예측오차 분산분해)
  - VECM-APARCH 하이브리드 모델

### Section 2: 고급 투자 전략 설계

**상태**: ✅ 강의 제공 중

트레이딩 전략의 실전 구현:

- **Chapter 1**: 동적 시계열 시뮬레이션
  - VECM-EGARCH 하이브리드 모델
  - 동적 재최적화
  - 롱/숏 포지션 관리

- **Chapter 2**: 비트코인 트레이딩에 전략 적용
  - 비트코인 특화 최적화
  - 수수료 고려사항
  - 변동성 기반 재최적화

- **Chapter 3**: Binance를 활용한 AI 트레이딩
  - Binance API 연동
  - 실시간 신호 생성
  - 자동 주문 실행

### Section 3: 프로덕션 투자 전략

**상태**: 🚀 **라이브 프로덕션 시스템**

실제 자금으로 운영 중인 프로덕션 수준의 트레이딩 시스템:

- **VECM-EGARCH 하이브리드 모델**
  - 공적분 관계를 위한 벡터 오차수정 모델
  - 변동성 모델링을 위한 지수 GARCH
  - 모델 신뢰도 기반 동적 포지션 크기 조절

- **주요 기능**:
  - ✅ 정보 누수 방지 (워킹 포워드 검증)
  - ✅ ECT alpha 기반 동적 재최적화
  - ✅ 신뢰도 기반 포지션 크기 조절 (0.2~0.8 비율 범위)
  - ✅ 매수(4일) 및 매도(7일)를 위한 별도 예측 기간
  - ✅ 실시간 증권사 API 연동
  - ✅ 자동 일일 거래 기록 업로드

- **수학적 모델**:

  **VECM 모델**:
  ```
  ΔY_t = αβ'Y_{t-1} + Γ₁ΔY_{t-1} + ... + Γ_{p-1}ΔY_{t-p+1} + ε_t
  ```
  - α: 조정 계수 (균형으로의 조정 속도)
  - β: 공적분 벡터 (장기 관계)
  - Γᵢ: 단기 동태 계수

  **EGARCH 모델**:
  ```
  log(σ²_t) = ω + Σᵢ₌₁ᵖ (αᵢ|z_{t-i}| + γᵢz_{t-i}) + Σⱼ₌₁ᵠ βⱼlog(σ²_{t-j})
  ```
  - 비대칭 변동성 효과 포착
  - 로그 변환을 통한 양의 분산 보장

  **하이브리드 예측**:
  ```
  Ŷ_{t+1} = VECM_forecast + EGARCH_mean_adjustment
  ```

- **트레이딩 전략**:
  - 롱 진입: `hybrid_yhat_buy > actual_price AND lower_price < lower_bound_mean`
  - 롱 청산: `upper_price > upper_bound_mean`
  - 동적 재최적화: ECT alpha가 음수에서 양수로 변경될 때

## 🚀 빠른 시작

### 설치

1. 저장소 클론:
```bash
git clone https://github.com/leesh2015/financial-timeseries-python.git
cd financial-timeseries-python
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

### 예제 실행

**Section 1 - 시계열 분석:**
```bash
cd "Section1.Financial Time Series Analysis/Chapter1.Fundamentals of Time Series Data Analysis"
python stable_data.py
```

**Section 2 - 전략 설계:**
```bash
cd "Section2.Advanced Investment Strategy Design/Chapter1.Dynamic Time Series Simulations"
python dynamic_simulation.py
```

**Section 3 - 프로덕션 시뮬레이션:**
```bash
cd "Section3.Production Investment Strategy"
python production_simulation.py
```

결과는 각 섹션의 `results/` 폴더에 저장됩니다.

## 📦 의존성

핵심 의존성 (전체 목록은 `requirements.txt` 참조):

- **데이터 사이언스**: `numpy`, `pandas`, `scipy`
- **시계열**: `statsmodels`, `arch`, `pmdarima`
- **데이터 수집**: `yfinance`
- **시각화**: `matplotlib`, `seaborn`
- **머신러닝**: `scikit-learn`
- **암호화폐**: `ccxt` (Binance 연동용)
- **Excel 지원**: `openpyxl`

## 🎯 주요 기능

### 정보 누수 방지
- 시간 t의 가격을 예측하기 위해 t-1까지의 데이터만 사용
- 각 단계에서 과거 데이터만 사용하여 모델 재훈련 (워킹 포워드)
- 예측 또는 최적화 단계에서 미래 데이터 사용 금지

### 동적 모델 적응
- 시장 조건 변화 시 자동 재최적화
- 공적분 관계 건강도 모니터링을 위한 ECT alpha 추적
- 변동성 기반 모델 조정

### 신뢰도 기반 포지션 크기 조절
- VECM 모델 신뢰도 기반 동적 포지션 크기 조절
- 비율 범위: 0.2 (낮은 신뢰도) ~ 0.8 (높은 신뢰도)
- 롤링 윈도우를 사용한 적응형 임계값 계산

### 프로덕션 준비 완료
- 실시간 증권사 API 연동
- 자동 거래 실행
- 일일 성과 추적 및 리포팅
- 웹 대시보드의 투명한 거래 기록

## 📈 성과 지표

프로덕션 시스템은 포괄적인 성과 지표를 추적합니다:

- 총 손익 (Total P&L)
- 승률 (Win Rate)
- 샤프 비율 (Sharpe Ratio)
- 최대 낙폭 (Maximum Drawdown)
- 연환산 수익률 (Annualized Returns)
- 바이앤홀드 비교 (Buy-and-Hold Comparison)

실시간 지표 보기: [거래 기록 대시보드](https://leenaissance.com/trading-history)

## 🔬 연구 및 방법론

이 프로젝트는 최신 금융 계량경제 기법을 구현합니다:

- **공적분 분석**: 장기 균형 관계 식별
- **오차수정 모델**: 균형으로부터의 단기 편차 포착
- **GARCH 계열 모델**: 변동성 클러스터링 및 비대칭성 모델링
- **하이브리드 예측**: 정확도 향상을 위한 다중 모델 결합
- **동적 최적화**: 변화하는 시장 체제에 적응

## 📖 강의-코드 매핑

| 강의 섹션 | 저장소 섹션 | 상태 |
|----------|-----------|------|
| Section 1: 시계열 기초 | `Section1.Financial Time Series Analysis/` | ✅ 제공 중 |
| Section 2: 전략 설계 | `Section2.Advanced Investment Strategy Design/` | ✅ 제공 중 |
| Section 3: 프로덕션 시스템 | `Section3.Production Investment Strategy/` | 🚀 라이브 |

## 🤝 기여하기

기여를 환영합니다! Pull Request를 자유롭게 제출해 주세요. 주요 변경사항의 경우, 먼저 이슈를 열어 변경하고자 하는 내용에 대해 논의해 주세요.

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 제공됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## ⚠️ 면책 조항

**중요**: 이 프로젝트는 교육 및 연구 목적으로 제공됩니다. 프로덕션 트레이딩 시스템은 강의에서 가르치는 개념의 시연으로 제공됩니다.

- 과거 성과는 미래 결과를 보장하지 않습니다
- 거래는 금융 손실의 위험이 따릅니다
- 어떤 트레이딩 전략을 배포하기 전에 항상 철저한 백테스팅을 수행하세요
- 작성자는 이 코드 사용으로 인한 금융 손실에 대해 책임지지 않습니다

## 🔗 링크

- **Udemy 강의**: [Mastering Financial Time Series Analysis with Python](https://www.udemy.com/course/mastering-financial-time-series-analysis-with-python/?referralCode=BA6CA9A3E5406E41359E&couponCode=KRLETSLEARNNOW)
- **거래 대시보드**: [leenaissance.com/trading-history](https://leenaissance.com/trading-history)

## 📧 문의

질문, 제안 또는 협업 기회에 대해서는 GitHub에 이슈를 열거나 Udemy 강의 플랫폼을 통해 연락해 주세요.

---

**알고리즘 트레이딩 커뮤니티를 위해 ❤️로 제작되었습니다**

*이 프로젝트는 이론적 시계열 분석부터 프로덕션 수준의 자동매매 시스템까지의 완전한 여정을 보여줍니다.*


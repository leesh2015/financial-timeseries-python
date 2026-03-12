import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VECM
from arch import arch_model
import warnings
from datetime import datetime, timedelta
from statsmodels.tsa.vector_ar.vecm import select_order
from statsmodels.tsa.vector_ar.vecm import select_coint_rank
import sys
import os

# 같은 폴더의 모듈 import
from functions_ import find_garch, max_drawdown, max_loss

# 레짐 감지 시스템 import
try:
    from regime_detector import ModelBasedRegimeDetector
    USE_REGIME_DETECTION = True
except ImportError as e:
    print(f"Warning: Could not import regime detector: {e}")
    print(f"  Make sure 'regime_detector.py' is in the same directory as this script.")
    USE_REGIME_DETECTION = False
    ModelBasedRegimeDetector = None

# 강화학습 시스템 import (Simple Policy만 사용)
USE_RL_AGENT = True  # False로 설정하면 RL 비활성화
# RL_BLEND_FACTOR는 환경 변수에서 읽거나 기본값 사용
RL_BLEND_FACTOR = float(os.environ.get('RL_BLEND_FACTOR', '0.6'))  # RL 에이전트 혼합 비율 (0.6)

try:
    from rl_agent import VECMRLAgent
    if not USE_RL_AGENT:
        VECMRLAgent = None
except ImportError as e:
    print(f"Warning: Could not import RL agent: {e}")
    print(f"  Make sure 'rl_agent.py' is in the same directory as this script.")
    USE_RL_AGENT = False
    VECMRLAgent = None

# Ignore warnings
warnings.filterwarnings("ignore")

# Set today's date
end_date = (datetime.today()).strftime('%Y-%m-%d')

# Set start date based on the interval
interval = '1d'  # Daily interval
start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')  

# List of tickers
tickers = ['TQQQ', 'NQ=F', 'ZB=F', 'UNG', 'RB=F', 'BZ=F', 'CL=F']
target_index = "TQQQ"

# Download data
df = yf.download(tickers, start=start_date, end=end_date, 
                 interval=interval, auto_adjust=True, progress=False)
df.reset_index(inplace=True)

# Set index to Date and add frequency information
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.asfreq('D')  # 'D' means daily frequency

# Remove rows with negative values
df = df[(df >= 0).all(axis=1)]

# Remove NaN values
df = df.dropna()

# Split data into training and testing sets
split_index = int(len(df) * 0.7)  # Use 70% for model estimation
train_data = df['Open'].iloc[:split_index]
test_data = df['Open'].iloc[split_index:]
ohlc_data = df.iloc[split_index:]

# Find optimal lag order
lag_order = select_order(train_data, maxlags=15, deterministic="colo")
time_lag = lag_order.aic
print(f"time_lag: {time_lag}")

# Find cointegration rank using Johansen test
coint_rank_test = select_coint_rank(train_data, det_order=1, k_ar_diff=time_lag, method='trace')
coint_rank_opt = coint_rank_test.rank
k_ar_diff_opt = time_lag

# Fit VECM model with optimal parameters
model = VECM(train_data, k_ar_diff=k_ar_diff_opt, coint_rank=coint_rank_opt, deterministic="colo")
model_fitted = model.fit()
print(f"k_ar_diff_opt: {k_ar_diff_opt}")
print(f"coint_rank_opt: {coint_rank_opt}")

# ECT alpha 값을 기준으로 재추정하기 위한 초기 alpha 값 계산
def get_alpha_value(model_fitted, target_index, history, method='weighted_mean'):
    """VECM의 오차 수정 계수(alpha) 값을 추출
    
    Alpha는 오차 수정 속도(error correction speed):
    - Alpha < 0: 평형으로 수렴 (mean-reverting) ✓ 바람직
    - Alpha > 0: 평형에서 발산 (divergent) ✗ 문제
    - Alpha = 0: 평형 조정 없음
    
    여러 공적분 관계가 있을 때:
    - VECM 이론상 모든 공적분 관계가 동시에 작용
    - 각 관계의 alpha는 독립적으로 작용하지만, 전체 수정 속도는 종합적
    
    Parameters:
    -----------
    method : str
        'weighted_mean': 모든 음수 alpha의 절댓값 가중 평균 (이론적으로 가장 정확)
        'min_negative': 가장 음수인 alpha (가장 강한 수렴 관계)
        'sum_negative': 모든 음수 alpha의 합 (모든 수렴 관계 고려)
    
    Returns:
    --------
    float: 종합 alpha 값 (수렴 판단용)
    """
    try:
        alpha = model_fitted.alpha
        target_idx = history.columns.get_loc(target_index)
        
        if alpha.ndim == 2:
            # 2차원 배열: (n_vars, n_coint_relations)
            # 예: (7, 3) = 7개 변수, 3개 공적분 관계
            if target_idx < alpha.shape[0] and alpha.shape[1] > 0:
                # 대상 변수의 모든 공적분 관계에 대한 alpha 값들
                target_alphas = alpha[target_idx, :]
                
                # 음수 alpha 값들만 필터링 (수렴하는 것들)
                negative_alphas = target_alphas[target_alphas < 0]
                
                if len(negative_alphas) > 0:
                    if method == 'weighted_mean':
                        # 이론적으로 가장 정확: 모든 수렴 관계를 절댓값으로 가중 평균
                        # 절댓값이 클수록 더 강한 수렴 → 더 큰 가중치
                        abs_negative = np.abs(negative_alphas)
                        weights = abs_negative / np.sum(abs_negative)
                        weighted_alpha = np.sum(negative_alphas * weights)
                        return float(weighted_alpha)
                    elif method == 'sum_negative':
                        # 모든 수렴 관계의 합 (모든 관계 고려)
                        return float(np.sum(negative_alphas))
                    elif method == 'min_negative':
                        # 가장 음수인 alpha (가장 강한 수렴 관계)
                        return float(np.min(negative_alphas))
                    else:
                        # 기본값: weighted_mean
                        abs_negative = np.abs(negative_alphas)
                        weights = abs_negative / np.sum(abs_negative)
                        return float(np.sum(negative_alphas * weights))
                else:
                    # 음수 alpha가 없으면 모든 alpha의 평균
                    # 하지만 이 경우는 문제가 있으므로 경고
                    return float(np.mean(target_alphas))
            else:
                return 0.0
        elif alpha.ndim == 1:
            # 1차원 배열: (n_coint_relations,)
            if target_idx < len(alpha):
                alpha_val = float(alpha[target_idx])
                return alpha_val
            else:
                return 0.0
        else:
            return 0.0
    except Exception as e:
        print(f"Warning: Error extracting alpha value: {e}")
        return 0.0

# Alpha 계산 방법 설정 (이론적으로 가장 정확한 방법)
ALPHA_METHOD = 'weighted_mean'  # 'weighted_mean', 'sum_negative', 'min_negative'

# 학습 데이터에서 초기 alpha 값 계산
initial_alpha = get_alpha_value(model_fitted, target_index, train_data, method=ALPHA_METHOD)
print(f"Initial ECT alpha value: {initial_alpha:.6f} (method: {ALPHA_METHOD})")

# 디버깅: 모든 공적분 관계의 alpha 값 출력
try:
    alpha = model_fitted.alpha
    target_idx = train_data.columns.get_loc(target_index)
    if alpha.ndim == 2:
        print(f"\n[Debug] Alpha matrix shape: {alpha.shape}")
        print(f"[Debug] Target variable index: {target_idx} ({target_index})")
        print(f"[Debug] All alpha values for {target_index}:")
        negative_alphas = []
        for coint_idx in range(alpha.shape[1]):
            alpha_val = alpha[target_idx, coint_idx]
            status = "✓ 수렴" if alpha_val < 0 else "✗ 발산"
            print(f"  Cointegration relation {coint_idx+1}: {alpha_val:>10.6f} {status}")
            if alpha_val < 0:
                negative_alphas.append(alpha_val)
        
        if len(negative_alphas) > 1:
            print(f"\n[Debug] 수렴 관계 종합:")
            print(f"  - 음수 alpha 개수: {len(negative_alphas)}")
            print(f"  - 가장 음수: {np.min(negative_alphas):.6f}")
            print(f"  - 합계: {np.sum(negative_alphas):.6f}")
            if ALPHA_METHOD == 'weighted_mean':
                abs_negative = np.abs(negative_alphas)
                weights = abs_negative / np.sum(abs_negative)
                weighted = np.sum(negative_alphas * weights)
                print(f"  - 가중 평균 (weighted_mean): {weighted:.6f}")
                print(f"    (각 관계의 절댓값으로 가중치 계산)")
except Exception as e:
    print(f"[Debug] Error printing alpha values: {e}")

if initial_alpha >= 0:
    print(f"\n  ⚠️ Warning: Initial alpha is non-negative (should be negative for convergence)")
    print(f"  ⚠️ This combination may not converge properly. Consider using a different combination.")

# Alpha 기준 재추정 임계값 설정
# Alpha가 양수로 변하거나 절댓값이 크게 변하면 재추정
alpha_change_threshold = 0.5  # 절댓값 기준 50% 변화 시 재추정

# VECM 신뢰도 기반 가변 비중 시스템
def get_vecm_confidence(model_fitted, target_index, history, lower_bound, upper_bound, predicted_mean):
    """VECM 모델의 신뢰도를 계산
    
    [중요] 예측 구간 일관성: lower_bound와 upper_bound는 같은 예측 기간(forecast_steps)에서 나온 값이어야 함
    - 이전 버전에서는 매수용 lower와 매도용 upper를 혼용하여 논리적 모순 발생
    - 현재는 매수용 예측 구간만 사용하여 일관성 확보
    
    R² 대신 여러 지표를 결합하여 신뢰도를 측정:
    1. 예측 구간의 폭: (upper - lower) / predicted_mean (작을수록 신뢰도 높음)
    2. 잔차의 표준편차: 작을수록 신뢰도 높음 (과거 적합도 지표, 미래 예측 신뢰도와 직접 연결되지 않을 수 있음)
    3. Alpha 절댓값: 클수록 수렴 속도 빠름 (공적분 관계 강도, 단기 예측 신뢰도와 직접 연결되지 않을 수 있음)
    
    Parameters:
    -----------
    model_fitted : VECMResults
        적합된 VECM 모델
    target_index : str
        대상 변수명
    history : pd.DataFrame
        히스토리 데이터
    lower_bound : float
        예측 하한선 (같은 예측 기간의 upper_bound와 쌍을 이뤄야 함)
    upper_bound : float
        예측 상한선 (같은 예측 기간의 lower_bound와 쌍을 이뤄야 함)
    predicted_mean : float
        예측 평균값
    
    Returns:
    --------
    float: 신뢰도 점수 (0~1 범위, 높을수록 신뢰도 높음)
    """
    try:
        target_idx = history.columns.get_loc(target_index)
        
        # 1. 예측 구간의 폭 (작을수록 신뢰도 높음)
        if predicted_mean > 0:
            interval_width = (upper_bound - lower_bound) / predicted_mean
        else:
            interval_width = 1.0  # 예측값이 0 이하면 최악으로 간주
        
        # 2. 잔차의 표준편차 (작을수록 신뢰도 높음)
        residuals = model_fitted.resid
        if residuals.ndim == 2:
            target_residuals = residuals[:, target_idx]
        else:
            target_residuals = residuals
        
        residual_std = np.std(target_residuals)
        # 잔차 표준편차를 예측값 대비로 정규화
        if predicted_mean > 0:
            normalized_residual_std = residual_std / predicted_mean
        else:
            normalized_residual_std = 1.0
        
        # 3. Alpha 절댓값 (클수록 수렴 속도 빠름, 신뢰도 높음)
        alpha = model_fitted.alpha
        if alpha.ndim == 2:
            target_alpha = alpha[target_idx, :]
            # 음수 alpha의 절댓값 평균 (수렴하는 것들만)
            negative_alphas = target_alpha[target_alpha < 0]
            if len(negative_alphas) > 0:
                alpha_abs_mean = np.mean(np.abs(negative_alphas))
            else:
                alpha_abs_mean = 0.0
        else:
            if target_idx < len(alpha):
                alpha_val = alpha[target_idx]
                alpha_abs_mean = abs(alpha_val) if alpha_val < 0 else 0.0
            else:
                alpha_abs_mean = 0.0
        
        # 각 지표를 0~1 범위로 정규화하고 결합
        # 예측 구간 폭: 더 넓은 범위로 매핑하여 민감도 증가
        # 0~1.0 범위를 1~0으로 매핑 (1.0 이상이면 0)
        confidence_interval = max(0, 1 - min(interval_width / 1.0, 1.0))
        
        # 잔차 표준편차: 더 넓은 범위로 매핑
        # 0~0.5 범위를 1~0으로 매핑
        confidence_residual = max(0, 1 - min(normalized_residual_std / 0.5, 1.0))
        
        # Alpha: 더 넓은 범위로 매핑
        # 0~0.05 범위를 0~1로 매핑 (0.05 이상이면 1)
        confidence_alpha = min(alpha_abs_mean / 0.05, 1.0)
        
        # 가중 평균으로 결합 (Alpha 수렴 속도 비중 상향)
        confidence = (0.4 * confidence_interval + 
                     0.3 * confidence_residual + 
                     0.3 * confidence_alpha)
        
        # 신뢰도를 더 넓은 범위로 확장 (0~1을 0.2~0.9로 매핑하여 더 큰 차이 만들기)
        # 이렇게 하면 신뢰도가 0.5~0.8 범위에서도 비중이 0.3~0.7 사이에서 더 크게 변동
        confidence_expanded = 0.2 + (confidence * 0.7)  # 0~1을 0.2~0.9로 확장
        
        return float(np.clip(confidence_expanded, 0.0, 1.0))
        
    except Exception as e:
        if not hasattr(get_vecm_confidence, '_error_count'):
            get_vecm_confidence._error_count = 0
        if get_vecm_confidence._error_count < 3:
            print(f"Warning: Error calculating VECM confidence: {e}")
            get_vecm_confidence._error_count += 1
        return 0.5  # 기본값

def calculate_adaptive_threshold(confidence_history, method='percentile', percentile=10, min_threshold=0.2, max_threshold=0.5):
    """과거 신뢰도 히스토리를 기반으로 적응형 임계값 계산
    
    여러 방법을 통해 동적으로 임계값을 계산:
    1. 'percentile': 하위 백분위수 사용 (기본값: 10th percentile)
    2. 'min': 최소값 사용
    3. 'mean_std': 평균 - 표준편차 사용
    4. 'rolling_min': Rolling window 최소값 사용
    
    Parameters:
    -----------
    confidence_history : list
        신뢰도 히스토리
    method : str
        계산 방법 ('percentile', 'min', 'mean_std', 'rolling_min')
    percentile : float
        백분위수 (0~100, 기본값: 10 = 하위 10%)
    min_threshold : float
        최소 임계값 (기본값: 0.2)
    max_threshold : float
        최대 임계값 (기본값: 0.5)
    
    Returns:
    --------
    float: 계산된 적응형 임계값
    """
    if len(confidence_history) < 2:
        return (min_threshold + max_threshold) / 2
    
    conf_array = np.array(confidence_history)
    
    if method == 'percentile':
        # 하위 백분위수 사용 (예: 10th percentile = 하위 10%)
        threshold = np.percentile(conf_array, percentile)
    elif method == 'min':
        # 최소값 사용
        threshold = np.min(conf_array)
    elif method == 'mean_std':
        # 평균 - 표준편차 사용 (하위 1 시그마)
        threshold = np.mean(conf_array) - np.std(conf_array)
    elif method == 'rolling_min':
        # 최근 30일 최소값 사용
        window = min(30, len(conf_array))
        threshold = np.min(conf_array[-window:])
    else:
        # 기본값: percentile
        threshold = np.percentile(conf_array, percentile)
    
    # 최소/최대 임계값으로 제한
    threshold = np.clip(threshold, min_threshold, max_threshold)
    return float(threshold)

def normalize_confidence_to_fraction(confidence_current, confidence_history, min_fraction=0.3, max_fraction=0.7, 
                                     window_size=60, method='zscore_sigmoid', absolute_threshold=None,
                                     threshold_method='percentile', threshold_percentile=10):
    """신뢰도 값을 정규화하여 비중(fraction)으로 변환
    
    [수정] 절대적 신뢰도 임계값 추가: 상대적 순위만 사용하는 문제 해결
    - 절대적 신뢰도가 너무 낮으면 최소 비중 사용
    - 전체 시장이 불확실한 시기에도 적절한 리스크 관리
    - [개선] 동적 임계값 계산 지원: 과거 히스토리 기반 적응형 임계값
    
    Parameters:
    -----------
    confidence_current : float
        현재 신뢰도 값
    confidence_history : list
        신뢰도 히스토리 (rolling window용)
    min_fraction : float
        최소 비중 (기본값: 0.3)
    max_fraction : float
        최대 비중 (기본값: 0.7)
    window_size : int
        Rolling window 크기 (기본값: 60일)
    method : str
        정규화 방법
        - 'zscore_sigmoid': Z-score 정규화 후 sigmoid 변환 (기본값, 부드러운 변화)
        - 'minmax': Min-Max scaling
        - 'percentile': Percentile 기반 변환
    absolute_threshold : float or None
        절대적 신뢰도 임계값
        - None: 동적 계산 (threshold_method 사용)
        - float: 고정값 사용
    threshold_method : str
        동적 임계값 계산 방법 (absolute_threshold=None일 때 사용)
        - 'percentile': 하위 백분위수 (기본값: 10th percentile)
        - 'min': 최소값
        - 'mean_std': 평균 - 표준편차
        - 'rolling_min': 최근 30일 최소값
    threshold_percentile : float
        백분위수 (0~100, 기본값: 10)
    
    Returns:
    --------
    float: 정규화된 비중 값 (min_fraction ~ max_fraction)
    """
    # [개선] 동적 임계값 계산
    if absolute_threshold is None:
        # 과거 히스토리 기반으로 동적 계산
        absolute_threshold = calculate_adaptive_threshold(
            confidence_history, 
            method=threshold_method,
            percentile=threshold_percentile
        )
    
    # [수정] 절대적 신뢰도 임계값 체크
    if confidence_current < absolute_threshold:
        return min_fraction
    
    if len(confidence_history) < 2:
        return (min_fraction + max_fraction) / 2
    
    window_conf = confidence_history[-window_size:] if len(confidence_history) >= window_size else confidence_history
    
    if method == 'zscore_sigmoid':
        mean_conf = np.mean(window_conf)
        std_conf = np.std(window_conf) if len(window_conf) > 1 else 0.1
        
        if std_conf < 1e-6:
            # 표준편차가 너무 작으면 minmax 방식으로 fallback
            min_conf = np.min(window_conf)
            max_conf = np.max(window_conf)
            if max_conf - min_conf < 1e-6:
                return (min_fraction + max_fraction) / 2
            normalized = (confidence_current - min_conf) / (max_conf - min_conf)
            return float(min_fraction + (max_fraction - min_fraction) * normalized)
        else:
            # Z-score를 더 민감하게 조정 (스케일 팩터 증가)
            z_score = (confidence_current - mean_conf) / (std_conf * 0.5)  # 더 민감하게
            z_score = np.clip(z_score, -2, 2)  # 범위 축소하여 더 극단적 반응
            sigmoid_value = 1 / (1 + np.exp(-z_score))
            normalized = min_fraction + (max_fraction - min_fraction) * sigmoid_value
        
        return float(np.clip(normalized, min_fraction, max_fraction))
    
    elif method == 'minmax':
        min_conf = np.min(window_conf)
        max_conf = np.max(window_conf)
        
        if max_conf - min_conf < 1e-6:
            return (min_fraction + max_fraction) / 2
        
        normalized = (confidence_current - min_conf) / (max_conf - min_conf)
        fraction = min_fraction + (max_fraction - min_fraction) * normalized
        return float(np.clip(fraction, min_fraction, max_fraction))
    
    elif method == 'percentile':
        percentile = np.sum(np.array(window_conf) <= confidence_current) / len(window_conf)
        fraction = min_fraction + (max_fraction - min_fraction) * percentile
        return float(np.clip(fraction, min_fraction, max_fraction))
    
    else:
        return normalize_confidence_to_fraction(confidence_current, confidence_history, min_fraction, max_fraction, 
                                                window_size, 'zscore_sigmoid')

# Determine optimal order for GARCH model
residuals = model_fitted.resid
best_aic, best_order, best_model = find_garch(residuals[:, train_data.columns.get_loc(target_index)])
print(f"Best AIC: {best_aic}")
print(f"Best Order: {best_order}")

# Fit EGARCH model with optimal order
p_opt, o_opt, q_opt = best_order
garch_model = arch_model(residuals[:, train_data.columns.get_loc(target_index)], vol='EGARCH', 
                         p=p_opt, o=o_opt, q=q_opt, rescale=True)
garch_fit = garch_model.fit(disp='off')

# Initialize variables
initial_capital = 10000
capital = initial_capital
long_count = 0
short_count = 0
unrealized_pnl = 0
total_assets = initial_capital
total_shares = 0
average_price = 0
position = None
peak_price = 0  # 트레일링 스탑용 최고점 기록
# Set commission rate as 0.02% (0.0002)
commission_rate = 0.0002
profit_after_commission = 0

# 리스크 관리 파라미터 추가
# [설명] 리스크는 크게 두 가지 방식으로 관리됩니다:
# 1. 기본 관리: 레짐(Regime) 감지 시스템을 통한 투자 비중(Fraction) 조절. 하락장/고변동성에서는 노출도를 대폭 축소.
# 2. 선택적 관리(USE_RISK_MANAGEMENT): 15% 이상 수익권 진입 시 트레일링 스탑을 통해 수익을 보존 (Profit Protection).
#    *사용자 피드백 반영: 35% 하드 스탑로스는 모델의 '물타기(Averaging down)' 로직을 방해할 수 있어 제거되었습니다.
# [수정] 슬리피지 방지를 위해 트레일링 스탑 기능을 비활성화했습니다. 
# 리스크 관리는 레짐별 투자 비중(Fraction) 최적화를 통해 수행됩니다.
USE_RISK_MANAGEMENT = False
STOP_LOSS_MULT = 8.0          # 적절한 수준의 변동성 보호
TRAILING_STOP_BASE = 0.25     # TQQQ에 적합한 수준의 트레일링 스탑 (수익 보존)
MIN_STOP_LOSS = 0.15          
MAX_STOP_LOSS = 0.50

# 진입 필터 파라미터
ENTRY_EDGE_PCT = 0.0          # 베이스라인 수준으로 복구

results = []
trade_history = []
simulation_start_date = test_data.index[0]
simulation_end_date = test_data.index[-1]
history = train_data.copy()

# 매수와 매도용 예측 기간 설정 (베이지안 최적화 결과 반영)
forecast_steps_buy = 4   # 베이스라인 복구
forecast_steps_sell = 7  # 베이스라인 복구

    # VECM 신뢰도 기반 가변 비중 설정 (기본값)
CONFIDENCE_FRACTION_CONFIG_BASE = {
    'min_fraction': 0.2,      # 최소 비중 (신뢰도가 낮을 때) - 더 넓은 범위
    'max_fraction': 0.8,       # 최대 비중 (신뢰도가 높을 때) - 더 넓은 범위
    'window_size': 60,         # Rolling window 크기 (일)
    'method': 'minmax',        # Min-Max scaling 사용 (더 민감한 반응)
    # [개선] 절대적 신뢰도 임계값 설정
    # 옵션 1: 고정값 사용
    # 'absolute_threshold': 0.3,
    # 옵션 2: 동적 계산 (권장) - None으로 설정하면 과거 히스토리 기반으로 자동 계산
    'absolute_threshold': None,  # None = 동적 계산 활성화
    'threshold_method': 'percentile',  # 'percentile', 'min', 'mean_std', 'rolling_min'
    'threshold_percentile': 10  # 하위 10% 백분위수 사용 (threshold_method='percentile'일 때)
}

# 레짐별 fraction 범위 설정 (Optuna 최적화 결과 - 1단계 Soft Penalty 적용)
REGIME_FRACTION_CONFIG = {
    'bull': {
        'min_fraction': 0.68,  # 최적화: 0.679
        'max_fraction': 0.97    # 최적화: 0.968
    },
    'bear': {
        'min_fraction': 0.48,  # 최적화: 0.479
        'max_fraction': 0.77    # 최적화: 0.770
    },
    'sideways': {
        'min_fraction': 0.43,  # 최적화: 0.426
        'max_fraction': 0.86    # 최적화: 0.860
    },
    'high_vol': {
        'min_fraction': 0.47,  # 최적화: 0.470
        'max_fraction': 0.70    # 최적화: 0.697
    }
}

# 레짐 감지 시스템 초기화
if USE_REGIME_DETECTION and ModelBasedRegimeDetector is not None:
    # [수정] 반응성 향상을 위해 hysteresis_threshold를 0.15 -> 0.05로 낮춤
    regime_detector = ModelBasedRegimeDetector(window=60, hysteresis_threshold=0.05)
    print(f"\n[레짐 감지 시스템 초기화 완료]")
    print(f"  Window: 60일")
    print(f"  Hysteresis Threshold: 0.05 (High Sensitivity)")
else:
    regime_detector = None
    print(f"\n[레짐 감지 시스템 비활성화]")

# RL 에이전트 초기화 (Simple Policy만 사용)
rl_agent = None
if USE_RL_AGENT and VECMRLAgent is not None:
    try:
        # 최적화된 Simple Policy 파라미터 직접 설정 (수익률 향상용)
        simple_policy_params = {
            'buy_confidence_threshold': 0.65,  # 더 낮은 문턱값으로 진입 기회 확대
            'sell_confidence_threshold': 0.55,  # 조금 더 빠른 이익 실현
            'max_position_ratio': 0.90,        # 최대 포지션 확대
            'min_position_ratio_for_sell': 0.20,
            'max_position_size': 0.95
        }
        print(f"\n[커스텀 최적화 Simple Policy 파라미터 사용]")
        for key, value in simple_policy_params.items():
            print(f"  {key}: {value:.4f}")
        
        # Simple Policy 사용
        rl_agent = VECMRLAgent(simple_policy_params=simple_policy_params)
        print(f"\n[RL 에이전트 초기화 완료]")
        print(f"  Mode: Simple Policy (VECM 신뢰도 기반)")
        print(f"  Blend Factor: {RL_BLEND_FACTOR * 100:.1f}%")
    except Exception as e:
        print(f"\n[RL 에이전트 초기화 실패] {e}")
        USE_RL_AGENT = False
        rl_agent = None
else:
    rl_agent = None
    USE_RL_AGENT = False
    print(f"\n[RL 시스템 비활성화]")

# 기본 설정 사용
CONFIDENCE_FRACTION_CONFIG = CONFIDENCE_FRACTION_CONFIG_BASE.copy()
# 초기 신뢰도 계산 (학습 데이터의 예측 구간 사용)
# 매수용 초기 신뢰도
initial_output_buy, initial_lower_buy, initial_upper_buy = model_fitted.predict(steps=forecast_steps_buy, alpha=0.5)
initial_predicted_mean_buy = initial_output_buy.mean(axis=0)[train_data.columns.get_loc(target_index)]
initial_lower_mean_buy = initial_lower_buy.mean(axis=0)[train_data.columns.get_loc(target_index)]
initial_upper_mean_buy = initial_upper_buy.mean(axis=0)[train_data.columns.get_loc(target_index)]
initial_confidence_buy = get_vecm_confidence(model_fitted, target_index, train_data, 
                                            initial_lower_mean_buy, initial_upper_mean_buy, initial_predicted_mean_buy)

# 매도용 초기 신뢰도
initial_output_sell, initial_lower_sell, initial_upper_sell = model_fitted.predict(steps=forecast_steps_sell, alpha=0.5)
initial_predicted_mean_sell = initial_output_sell.mean(axis=0)[train_data.columns.get_loc(target_index)]
initial_lower_mean_sell = initial_lower_sell.mean(axis=0)[train_data.columns.get_loc(target_index)]
initial_upper_mean_sell = initial_upper_sell.mean(axis=0)[train_data.columns.get_loc(target_index)]
initial_confidence_sell = get_vecm_confidence(model_fitted, target_index, train_data, 
                                             initial_lower_mean_sell, initial_upper_mean_sell, initial_predicted_mean_sell)

# 신뢰도 히스토리 초기화 (매수용과 매도용 각각)
confidence_history_buy = [initial_confidence_buy]
confidence_history_sell = [initial_confidence_sell]
fraction = normalize_confidence_to_fraction(initial_confidence_buy, confidence_history_buy, 
                                            **CONFIDENCE_FRACTION_CONFIG)

# 동적 임계값 계산 (로깅용)
if CONFIDENCE_FRACTION_CONFIG.get('absolute_threshold') is None:
    adaptive_threshold_buy = calculate_adaptive_threshold(
        confidence_history_buy,
        method=CONFIDENCE_FRACTION_CONFIG.get('threshold_method', 'percentile'),
        percentile=CONFIDENCE_FRACTION_CONFIG.get('threshold_percentile', 10)
    )
    adaptive_threshold_sell = calculate_adaptive_threshold(
        confidence_history_sell,
        method=CONFIDENCE_FRACTION_CONFIG.get('threshold_method', 'percentile'),
        percentile=CONFIDENCE_FRACTION_CONFIG.get('threshold_percentile', 10)
    )
    threshold_info = f"Dynamic (Buy: {adaptive_threshold_buy:.4f}, Sell: {adaptive_threshold_sell:.4f})"
else:
    threshold_info = f"Fixed: {CONFIDENCE_FRACTION_CONFIG['absolute_threshold']:.4f}"

print(f"\n[VECM 신뢰도 기반 가변 비중 시스템 초기화]")
print(f"  Initial Confidence (Buy): {initial_confidence_buy:.6f}")
print(f"  Initial Confidence (Sell): {initial_confidence_sell:.6f}")
print(f"  Initial Fraction: {fraction:.4f}")
print(f"  Absolute Threshold: {threshold_info}")
print(f"  Config: min={CONFIDENCE_FRACTION_CONFIG['min_fraction']}, "
      f"max={CONFIDENCE_FRACTION_CONFIG['max_fraction']}, "
      f"window={CONFIDENCE_FRACTION_CONFIG['window_size']}, "
      f"method={CONFIDENCE_FRACTION_CONFIG['method']}")
if USE_REGIME_DETECTION and regime_detector is not None:
    print(f"\n[레짐별 Fraction 범위 설정]")
    for regime, config in REGIME_FRACTION_CONFIG.items():
        print(f"  {regime.upper()}: min={config['min_fraction']:.2f}, max={config['max_fraction']:.2f}")
unrealized_pnl_history = []
# Initialize cumulative commission
cumulative_commission = 0
# Track shares history for charting
shares_history = []
shares_dates = []
# Track fraction history for analysis
fraction_history = [fraction]

for t in range(len(test_data)):
    # Update history with new test data
    history = pd.concat([history, test_data.iloc[[t]]])
    
    # Print logs only every 100 iterations
    should_log = (t % 100 == 0)
    
    # Calculate upward probability using VECM model
    model = VECM(history, k_ar_diff=k_ar_diff_opt, coint_rank=coint_rank_opt, deterministic="colo")
    model_fitted = model.fit()
    
    # 매수용 예측 (forecast_steps_buy 사용)
    output_buy, lower_bound_buy, upper_bound_buy = model_fitted.predict(steps=forecast_steps_buy, alpha=0.5)
    lower_mean = lower_bound_buy.mean(axis=0)[history.columns.get_loc(target_index)]
    predicted_mean_buy = output_buy.mean(axis=0)[history.columns.get_loc(target_index)]
    
    # 매도용 예측 (forecast_steps_sell 사용)
    output_sell, lower_bound_sell, upper_bound_sell = model_fitted.predict(steps=forecast_steps_sell, alpha=0.5)
    upper_mean = upper_bound_sell.mean(axis=0)[history.columns.get_loc(target_index)]
    predicted_mean_sell = output_sell.mean(axis=0)[history.columns.get_loc(target_index)]
    
    # NaN 원인 파악용 디버깅 (폴백 없이 원인만 로깅)
    if np.isnan(predicted_mean_buy) or np.isnan(predicted_mean_sell):
        print(f"\n[DEBUG] NaN detected at t={t}, date={test_data.index[t]}")
        print(f"  predicted_mean_buy: {predicted_mean_buy}, predicted_mean_sell: {predicted_mean_sell}")
        print(f"  output_buy shape: {output_buy.shape}, has_nan: {np.isnan(output_buy).any()}")
        print(f"  output_sell shape: {output_sell.shape}, has_nan: {np.isnan(output_sell).any()}")
        print(f"  output_buy sample: {output_buy[:3, :3] if output_buy.size > 0 else 'empty'}")
        print(f"  history length: {len(history)}, history shape: {history.shape}")
        print(f"  target_index: {target_index}, target_index position: {history.columns.get_loc(target_index)}")
    
    # VECM 신뢰도 기반 가변 비중 업데이트
    # [개선] 매수용과 매도용 신뢰도를 각각 계산하여 일관성 확보
    upper_mean_buy = upper_bound_buy.mean(axis=0)[history.columns.get_loc(target_index)]
    lower_mean_sell = lower_bound_sell.mean(axis=0)[history.columns.get_loc(target_index)]
    
    # 매수용 신뢰도 계산 (매수용 예측 구간 사용)
    confidence_buy = get_vecm_confidence(model_fitted, target_index, history, 
                                         lower_mean, upper_mean_buy, predicted_mean_buy)
    
    # 매도용 신뢰도 계산 (매도용 예측 구간 사용)
    confidence_sell = get_vecm_confidence(model_fitted, target_index, history, 
                                          lower_mean_sell, upper_mean, predicted_mean_sell)
    
    # 신뢰도 히스토리 업데이트 (매수용과 매도용 각각 유지)
    confidence_history_buy.append(confidence_buy)
    confidence_history_sell.append(confidence_sell)
    # 최근 window_size만 유지 (메모리 효율성)
    if len(confidence_history_buy) > CONFIDENCE_FRACTION_CONFIG['window_size'] * 2:
        confidence_history_buy = confidence_history_buy[-CONFIDENCE_FRACTION_CONFIG['window_size']:]
    if len(confidence_history_sell) > CONFIDENCE_FRACTION_CONFIG['window_size'] * 2:
        confidence_history_sell = confidence_history_sell[-CONFIDENCE_FRACTION_CONFIG['window_size']:]
    
    # GARCH 계산용으로는 매수용 예측 사용
    predicted_mean = predicted_mean_buy

    # Update GARCH model
    residuals = model_fitted.resid[:, history.columns.get_loc(target_index)]
    garch_model = arch_model(residuals, vol='EGARCH', p=p_opt, o=o_opt, q=q_opt, rescale=True)
    garch_fit = garch_model.fit(disp='off')
    garch_forecast = garch_fit.forecast(horizon=1, start=len(residuals)-p_opt)
    garch_volatility = np.sqrt(garch_forecast.variance.values[-1, :])
    garch_mean_values = garch_forecast.mean.values
    # NaN이 있는 경우에만 특별 처리, 없으면 원래대로 평균 계산
    if np.isnan(garch_mean_values).any():
        # NaN이 하나라도 있으면 nan을 제외한 유효한 값들의 평균 사용
        valid_values = garch_mean_values[~np.isnan(garch_mean_values)]
        if len(valid_values) > 0:
            garch_mean_average = np.mean(valid_values)
        else:
            garch_mean_average = 0.0  # 모두 nan이면 0으로 설정
    else:
        # NaN이 없으면 원래대로 전체 평균 계산
        garch_mean_average = np.mean(garch_mean_values)
    
    # 레짐 감지 및 fraction 범위 조정 (fraction 계산 전에 수행)
    current_regime = 'sideways'  # 기본값
    if USE_REGIME_DETECTION and regime_detector is not None:
        try:
            # GARCH 변동성 평균 계산
            garch_vol_mean = float(np.mean(garch_volatility)) if isinstance(garch_volatility, np.ndarray) else float(garch_volatility)
            
            # 레짐 감지
            current_regime = regime_detector.detect_regime(
                vecm_pred=predicted_mean_buy,
                kalman_beta=None,  # 원본 파일에는 Kalman Filter가 없음
                garch_vol=garch_vol_mean,
                model_confidences={'vecm': confidence_buy},
                copula_risk=None
            )
            
            # 레짐별 fraction 범위 적용
            if current_regime in REGIME_FRACTION_CONFIG:
                regime_config = REGIME_FRACTION_CONFIG[current_regime]
                CONFIDENCE_FRACTION_CONFIG['min_fraction'] = regime_config['min_fraction']
                CONFIDENCE_FRACTION_CONFIG['max_fraction'] = regime_config['max_fraction']
                
                if should_log:
                    print(f"  [Regime] {current_regime.upper()}: min={regime_config['min_fraction']:.2f}, max={regime_config['max_fraction']:.2f}")
        except Exception as e:
            if should_log:
                print(f"  Warning: Regime detection failed: {e}")
            current_regime = 'sideways'
            # 기본값으로 복원
            CONFIDENCE_FRACTION_CONFIG['min_fraction'] = CONFIDENCE_FRACTION_CONFIG_BASE['min_fraction']
            CONFIDENCE_FRACTION_CONFIG['max_fraction'] = CONFIDENCE_FRACTION_CONFIG_BASE['max_fraction']
    
    # 매수용 비중 계산 (레짐별 범위 적용 후)
    base_fraction_buy = normalize_confidence_to_fraction(confidence_buy, confidence_history_buy, **CONFIDENCE_FRACTION_CONFIG)
    
    # 매도용 비중 계산 (레짐별 범위 적용 후)
    # [수정] 매도용 신뢰도 히스토리를 별도로 유지하여 일관성 확보
    base_fraction_sell = normalize_confidence_to_fraction(confidence_sell, confidence_history_sell, **CONFIDENCE_FRACTION_CONFIG)
    
    # RL 에이전트와 혼합 (점진적 접근)
    if USE_RL_AGENT and rl_agent is not None and t >= 10:
        try:
            # RL 에이전트를 위한 관찰값 생성
            # observation: [vecm_pred, vecm_confidence, garch_vol, regime, position_ratio, capital_ratio]
            regime_map = {'bull': 1.0, 'bear': -1.0, 'sideways': 0.0, 'high_vol': 0.5}
            regime_value = regime_map.get(current_regime, 0.0)
            
            # 포지션 비율 계산 (actual_price는 아직 정의되지 않았으므로 test_data에서 직접 가져옴)
            current_price_for_rl = test_data[target_index].iloc[t]
            if position == 'long' and total_shares > 0:
                position_value = total_shares * current_price_for_rl
                temp_total_assets = capital + total_shares * current_price_for_rl
                position_ratio = position_value / temp_total_assets if temp_total_assets > 0 else 0.0
            else:
                position_ratio = 0.0
            
            # 자본 비율
            capital_ratio = capital / initial_capital if initial_capital > 0 else 1.0
            
            # [추가] RL 에이전트를 위한 매크로 모멘텀 지표 (거짓 양성 필터용)
            # CL=F (원유)와 ZB=F (미 국채)의 20일 모멘텀 계산
            try:
                # 20일 전 데이터가 있어야 하므로 t >= 20 조건 필요 (이 블록은 t >= 10이므로 히스토리 참조)
                cl_past = history['CL=F'].iloc[-21] if len(history) > 20 else history['CL=F'].iloc[0]
                cl_curr = history['CL=F'].iloc[-1]
                cl_momentum = (cl_curr - cl_past) / cl_past if cl_past > 0 else 0.0
                
                zb_past = history['ZB=F'].iloc[-21] if len(history) > 20 else history['ZB=F'].iloc[0]
                zb_curr = history['ZB=F'].iloc[-1]
                zb_momentum = (zb_curr - zb_past) / zb_past if zb_past > 0 else 0.0
            except Exception as e:
                # 에러 발생 시 (또는 종목이 없을 시) 0으로 처리
                cl_momentum = 0.0
                zb_momentum = 0.0
            
            observation = np.array([
                predicted_mean_buy,
                confidence_buy,
                garch_vol_mean,
                regime_value,
                position_ratio,
                capital_ratio,
                cl_momentum,   # [추가] 원유 20일 모멘텀 (인플레이션 지표)
                zb_momentum    # [추가] 국채 20일 모멘텀 (금리 지표)
            ], dtype=np.float32)
            
            # RL 에이전트 액션 예측
            rl_action = rl_agent.predict(observation, deterministic=True)
            rl_position_size = float(np.clip(rl_action[0], 0.0, 1.0))
            rl_signal = float(np.clip(rl_action[1], -1.0, 1.0))
            
            # RL 신호를 Fraction으로 변환
            # signal > 0.5: 매수 → fraction 증가
            # signal < -0.5: 매도 → fraction 감소
            if rl_signal > 0.5:
                # 매수 신호: RL position_size를 fraction으로 사용
                rl_fraction_buy = np.clip(rl_position_size, 
                                        CONFIDENCE_FRACTION_CONFIG['min_fraction'],
                                        CONFIDENCE_FRACTION_CONFIG['max_fraction'])
                rl_fraction_sell = base_fraction_sell  # 매도는 기본값
            elif rl_signal < -0.5:
                # 매도 신호: RL position_size를 fraction으로 사용 (매도용)
                rl_fraction_sell = np.clip(rl_position_size,
                                          CONFIDENCE_FRACTION_CONFIG['min_fraction'],
                                          CONFIDENCE_FRACTION_CONFIG['max_fraction'])
                rl_fraction_buy = base_fraction_buy  # 매수는 기본값
            else:
                # 보유: 기본 fraction 사용
                rl_fraction_buy = base_fraction_buy
                rl_fraction_sell = base_fraction_sell
            
            # RL fraction과 기본 fraction 혼합
            blend_factor = RL_BLEND_FACTOR
            fraction_buy = blend_factor * rl_fraction_buy + (1 - blend_factor) * base_fraction_buy
            fraction_sell = blend_factor * rl_fraction_sell + (1 - blend_factor) * base_fraction_sell
            
            # 범위 제한
            fraction_buy = np.clip(fraction_buy, 
                                  CONFIDENCE_FRACTION_CONFIG['min_fraction'],
                                  CONFIDENCE_FRACTION_CONFIG['max_fraction'])
            fraction_sell = np.clip(fraction_sell,
                                   CONFIDENCE_FRACTION_CONFIG['min_fraction'],
                                   CONFIDENCE_FRACTION_CONFIG['max_fraction'])
            
            if should_log:
                print(f"  [RL] Buy: {rl_fraction_buy:.4f} (blended: {fraction_buy:.4f}), "
                      f"Sell: {rl_fraction_sell:.4f} (blended: {fraction_sell:.4f}), "
                      f"Signal: {rl_signal:.2f}")
        except Exception as e:
            # Fallback: RL 실패 시 기본 계산 사용
            if should_log:
                print(f"  Warning: RL agent failed, using default calculation: {e}")
            fraction_buy = base_fraction_buy
            fraction_sell = base_fraction_sell
    else:
        # RL 비활성화 또는 데이터 부족: 기본 계산 사용
        fraction_buy = base_fraction_buy
        fraction_sell = base_fraction_sell
    
    # 현재 비중 (로깅 및 기록용, 매수용 비중 사용)
    fraction = fraction_buy
    fraction_history.append(fraction)
    
    # NaN 원인 파악용 디버깅 (GARCH 부분)
    if np.isnan(garch_mean_average):
        print(f"\n[DEBUG] NaN in garch_mean_average at t={t}, date={test_data.index[t]}")
        print(f"  garch_mean_values: {garch_mean_values}, shape: {garch_mean_values.shape}")
        print(f"  garch_mean_values has_nan: {np.isnan(garch_mean_values).any()}")
        print(f"  residuals length: {len(residuals)}, residuals has_nan: {np.isnan(residuals).any()}")
        print(f"  garch_forecast.variance shape: {garch_forecast.variance.shape if hasattr(garch_forecast, 'variance') else 'N/A'}")
    
    # 매수용 hybrid_yhat (매수 판단용)
    var_hat_buy = predicted_mean_buy + garch_mean_average
    if isinstance(var_hat_buy, np.ndarray):
        hybrid_yhat_buy = var_hat_buy.item()
    else:
        hybrid_yhat_buy = var_hat_buy
    
    # 매도용 hybrid_yhat (매도 판단용)
    var_hat_sell = predicted_mean_sell + garch_mean_average
    if isinstance(var_hat_sell, np.ndarray):
        hybrid_yhat_sell = var_hat_sell.item()
    else:
        hybrid_yhat_sell = var_hat_sell
    
    # 최종 hybrid_yhat NaN 체크
    if np.isnan(hybrid_yhat_buy) or np.isnan(hybrid_yhat_sell):
        print(f"\n[DEBUG] Final hybrid_yhat NaN at t={t}, date={test_data.index[t]}")
        print(f"  hybrid_yhat_buy: {hybrid_yhat_buy}, hybrid_yhat_sell: {hybrid_yhat_sell}")
        print(f"  predicted_mean_buy: {predicted_mean_buy}, predicted_mean_sell: {predicted_mean_sell}")
        print(f"  garch_mean_average: {garch_mean_average}")

    # Skip trade if prediction is negative (매수용 예측 기준)
    # 주의: 이 부분은 신뢰도 계산 전에 실행되므로, 신뢰도는 계산되지 않음
    """ if hybrid_yhat_buy < 0:
        # 주식 수량 기록 (거래가 없어도 기록)
        shares_history.append(total_shares)
        shares_dates.append(test_data.index[t])
        # 포트폴리오 가치 계산
        if position == 'long':
            total_assets = capital + total_shares * test_data[target_index].iloc[t]
        else:
            total_assets = capital
        results.append(total_assets)
        
        # 거래 없이 포트폴리오만 기록
        # 주의: negative prediction이면 신뢰도 계산 전이므로 None으로 기록
        date_str = test_data.index[t].strftime('%Y-%m-%d')
        close_price = ohlc_data['Close'][target_index].iloc[t]
        trade_history.append({
            'date': date_str,
            'confidence_buy': None,  # negative prediction이면 신뢰도 계산 안 함
            'confidence_sell': None,
            'fraction_buy': None,
            'fraction_sell': None,
            'fraction': None,
            'hybrid_yhat_buy': hybrid_yhat_buy,
            'hybrid_yhat_sell': hybrid_yhat_sell,
            'actual_price': close_price,
            'capital': capital,
            'total_shares': total_shares,
            'total_assets': total_assets,
            'position': position,
            'unrealized_pnl': (close_price - average_price) * total_shares if position == 'long' else 0,
            'note': 'Negative prediction'
        })
        continue """
    
    # ECT alpha 값을 기준으로 재추정 여부 결정
    current_alpha = get_alpha_value(model_fitted, target_index, history, method=ALPHA_METHOD)
    
    # 재추정 조건 체크
    # Alpha > 0이면 공적분 관계가 깨진 상태이므로, 거래를 건너뜀
    # initial_alpha < 0 체크는 불필요: 재조정 후에도 alpha > 0이면 여전히 문제
    should_reoptimize = False
    reoptimize_reason = ""
    
    # Alpha > 0이면 공적분 관계가 깨진 상태 → 재최적화
    if current_alpha > 0:
        should_reoptimize = True
        reoptimize_reason = f"Alpha is positive (convergence lost): {current_alpha:.6f} (initial: {initial_alpha:.6f})"
    
    # [신규 추가] Alpha가 50% 이상 약화되면 선제적으로 재최적화
    """ elif current_alpha > initial_alpha * 0.5:
        should_reoptimize = True
        reoptimize_reason = f"Alpha weakened by more than 50%: {current_alpha:.6f} (initial: {initial_alpha:.6f})" """

    # 재추정 실행
    if should_reoptimize:
        if should_log:
            print(f"\n[Re-optimization triggered by ECT alpha] {reoptimize_reason}")
            print(f"  ⚠️ 포지션은 유지하고 모델만 재최적화합니다 (전량 청산 없음)")
        
        # [NEW] Full Re-optimization Logic
        # 1. Lag order 재최적화
        lag_order_new = select_order(history, maxlags=15, deterministic="colo")
        k_ar_diff_opt = lag_order_new.aic
        
        # 2. Cointegration rank 재최적화
        coint_rank_test_new = select_coint_rank(history, det_order=1, k_ar_diff=k_ar_diff_opt, method='trace')
        coint_rank_opt = coint_rank_test_new.rank
        
        if should_log:
            print(f"  Re-optimized k_ar_diff_opt: {k_ar_diff_opt}")
            print(f"  Re-optimized coint_rank_opt: {coint_rank_opt}")
        
        # 3. 새로운 VECM 모델 피팅
        model = VECM(history, k_ar_diff=k_ar_diff_opt, coint_rank=coint_rank_opt, deterministic="colo")
        model_fitted = model.fit()
        
        # 4. GARCH 모델 재최적화
        residuals_new = model_fitted.resid[:, history.columns.get_loc(target_index)]
        best_aic_new, best_order_new, best_model_new = find_garch(residuals_new)
        p_opt, o_opt, q_opt = best_order_new
        
        if should_log:
            print(f"  Re-optimized GARCH order: {best_order_new} (AIC: {best_aic_new})")
        
        # 5. Update initial alpha with new fitted model
        initial_alpha = get_alpha_value(model_fitted, target_index, history, method=ALPHA_METHOD)
        
        # 6. 재추정 구간에서도 신뢰도 계산 (비중 업데이트는 하되 거래는 하지 않음)
        # [개선] 매수용과 매도용 신뢰도 각각 계산
        # 재최적화된 모델로 예측 다시 계산
        output_buy, lower_bound_buy, upper_bound_buy = model_fitted.predict(steps=forecast_steps_buy, alpha=0.5)
        lower_mean = lower_bound_buy.mean(axis=0)[history.columns.get_loc(target_index)]
        predicted_mean_buy = output_buy.mean(axis=0)[history.columns.get_loc(target_index)]
        upper_mean_buy = upper_bound_buy.mean(axis=0)[history.columns.get_loc(target_index)]
        
        output_sell, lower_bound_sell, upper_bound_sell = model_fitted.predict(steps=forecast_steps_sell, alpha=0.5)
        upper_mean = upper_bound_sell.mean(axis=0)[history.columns.get_loc(target_index)]
        predicted_mean_sell = output_sell.mean(axis=0)[history.columns.get_loc(target_index)]
        lower_mean_sell = lower_bound_sell.mean(axis=0)[history.columns.get_loc(target_index)]
        
        confidence_buy = get_vecm_confidence(model_fitted, target_index, history, 
                                             lower_mean, upper_mean_buy, predicted_mean_buy)
        confidence_sell = get_vecm_confidence(model_fitted, target_index, history, 
                                              lower_mean_sell, upper_mean, predicted_mean_sell)
        confidence_history_buy.append(confidence_buy)
        confidence_history_sell.append(confidence_sell)
        if len(confidence_history_buy) > CONFIDENCE_FRACTION_CONFIG['window_size'] * 2:
            confidence_history_buy = confidence_history_buy[-CONFIDENCE_FRACTION_CONFIG['window_size']:]
        if len(confidence_history_sell) > CONFIDENCE_FRACTION_CONFIG['window_size'] * 2:
            confidence_history_sell = confidence_history_sell[-CONFIDENCE_FRACTION_CONFIG['window_size']:]
        fraction_buy = normalize_confidence_to_fraction(confidence_buy, confidence_history_buy, **CONFIDENCE_FRACTION_CONFIG)
        fraction_sell = normalize_confidence_to_fraction(confidence_sell, confidence_history_sell, **CONFIDENCE_FRACTION_CONFIG)
        fraction = fraction_buy
        fraction_history.append(fraction)
        
        if should_log:
            print(f"  Updated initial_alpha: {initial_alpha:.6f} (method: {ALPHA_METHOD})")
            print(f"  Confidence (Buy): {confidence_buy:.6f}, Confidence (Sell): {confidence_sell:.6f}")
            print(f"  Fraction (Buy): {fraction_buy:.4f}, Fraction (Sell): {fraction_sell:.4f}")
            print(f"  ⚠️ Skipping trading at this step due to re-optimization\n")
        
        # 재추정이 필요한 구간에서는 거래를 하지 않음
        # 포트폴리오 가치만 업데이트하고 다음 루프로 이동
        date_str = test_data.index[t].strftime('%Y-%m-%d')
        actual_price = test_data[target_index].iloc[t]
        
        # 포트폴리오 가치 계산 (보유 포지션이 있다면)
        if position == 'long':
            total_assets = capital + total_shares * actual_price
        else:
            total_assets = capital
        
        results.append(total_assets)
        # 주식 수량 기록
        shares_history.append(total_shares)
        shares_dates.append(test_data.index[t])
        
        # 거래 없이 포트폴리오만 기록
        trade_history.append({
            'date': date_str,
            'confidence_buy': confidence_buy,
            'confidence_sell': confidence_sell,
            'fraction_buy': fraction_buy,
            'fraction_sell': fraction_sell,
            'fraction': fraction,
            'hybrid_yhat_buy': None,  # 재추정 중이므로 예측값 없음
            'hybrid_yhat_sell': None,  # 재추정 중이므로 예측값 없음
            'actual_price': actual_price,
            'capital': capital,
            'total_shares': total_shares,
            'total_assets': total_assets,
            'position': position,
            'unrealized_pnl': (actual_price - average_price) * total_shares if position == 'long' else 0,
            'note': 'Re-optimization',
            'regime': current_regime if USE_REGIME_DETECTION and regime_detector is not None else 'sideways'
        })
        
        if should_log:
            print(f"date: {date_str:>10} | [RE-OPTIMIZATION] | actual_price: {actual_price:>10.2f} | "
                  f"capital: {capital:>10.2f} | total_shares: {total_shares:>10.2f} | total_assets: {total_assets:>10.2f} | "
                  f"position: {str(position):>6}")
        
        continue  # 이번 루프에서는 거래하지 않고 다음으로

    # Get current actual price
    actual_price = test_data[target_index].iloc[t]
    lower_price = ohlc_data['Low'][target_index].iloc[t]
    upper_price = ohlc_data['High'][target_index].iloc[t]
    close_price = ohlc_data['Close'][target_index].iloc[t]

    # Initialize shares to sell and buy
    shares_to_sell = 0
    shares_to_buy = 0

    # 현재 레짐 확인 (상승장 여부)
    is_bull_regime = (current_regime == 'bull')

    # 롱 포지션 진입 - 정수 단위 주문으로 수정 (매수용 예측 사용)
    # [수정] 진입 조건은 모든 레짐에서 동일하게 지정가(lower_mean) 원칙을 유지
    entry_condition = (lower_price < lower_mean)
    
    # [개선] 기대 수익률 필터 추가
    expected_return = (hybrid_yhat_buy - actual_price) / actual_price if actual_price > 0 else 0
    has_edge = expected_return > ENTRY_EDGE_PCT

    if hybrid_yhat_buy > actual_price and capital > 0 and entry_condition and has_edge:
        # Set new long position
        if position != 'long':
            peak_price = actual_price  # 새로운 포지션 진입 시 초기화
        position = 'long'
        
        # 소수점 금액 계산 후 정수로 내림
        # 체결가는 하한 바운더리 평균(lower_mean)을 사용 (저가는 체결 불가능)
        # [개선] 매수용 비중 사용
        shares_to_buy_float = (capital * fraction_buy) / lower_mean
        shares_to_buy = int(shares_to_buy_float)  # 정수로 내림
        
        # 최소 주문 단위 확인 (0주는 주문 불가)
        if shares_to_buy >= 1 and shares_to_buy * lower_mean <= capital:
            total_value = (average_price * total_shares) + (lower_mean * shares_to_buy)
            total_shares += shares_to_buy
            average_price = total_value / total_shares
            commission = shares_to_buy * lower_mean * commission_rate
            cumulative_commission += commission
            capital -= shares_to_buy * lower_mean + commission
            
            if should_log:
                print(f"Buying {shares_to_buy} shares at {lower_mean:.2f} (wanted to buy {shares_to_buy_float:.2f}, lower_price: {lower_price:.2f})")

    # 트레일링 스탑 로직 (Profit Protection - 수익권에서만 작동)
    # [설명] 리스크 관리는 기본적으로 레짐 기반 비중 조절을 통해 수행됩니다. 
    # 본 섹션은 '수익 보존'을 위한 선택적 트레일링 스탑 기능을 제공합니다.
    # [현실적 손절선 설정 원칙]: 
    # 1. 'peak_price'는 오늘 이전까지 해당 포지션에서 관측된 "역대 최고가(High-Water Mark)"입니다. 
    # 2. 오늘 손절선은 이 'peak_price'를 기준으로 설정하며, 오늘 가격이 다시 전일 최고점에 도달할 것이라 가정하지 않습니다.
    # 3. 만약 오늘 저가가 이 손절선(Floor)을 건드리면 즉시 매도하며, 시가가 이미 손절선 아래라면 시가에 매도합니다 (Gap Down 대응).
    if USE_RISK_MANAGEMENT and position == 'long' and total_shares >= 1:
        # 현재 수익률 계산 (진입 시점 기준이 아닌 전일 종가 또는 오늘 시가 기준)
        current_return = (actual_price - average_price) / average_price if average_price > 0 else 0
        
        # 수익이 일정 수준(예: 15%) 이상일 때만 트레일링 스탑 활성화
        profit_protection_threshold = 0.15
        
        if current_return > profit_protection_threshold:
            # 1. 변동성 기반 동적 스탑로스 배수 조정 (수익권에서는 더 타이트하게)
            dynamic_stop_pct = np.clip(garch_volatility * (STOP_LOSS_MULT * 0.7), MIN_STOP_LOSS, MAX_STOP_LOSS)
            
            # 2. 트레일링 스탑 (수익 보존용)
            current_trailing_stop = TRAILING_STOP_BASE * 0.8 # 더 타이트하게
            trailing_stop_price = peak_price * (1 - current_trailing_stop)
            
            # 3. 고정 스탑로스 (최소 수익 보장: 5%)
            guaranteed_profit_price = average_price * 1.05
            
            # 최종 손절 트리거 가격 결정
            exit_trigger_price = max(trailing_stop_price, guaranteed_profit_price)
            
            # [현실성 강화] 오늘 저가(lower_price)가 트리거 가격을 하회하면 매도
            if lower_price < exit_trigger_price:
                # [정보 누수 방지] 만약 시가(actual_price)가 이미 트리거 가격보다 낮게 시작했다면(Gap Down),
                # 트리거 가격이 아닌 시가에 매도된 것으로 처리하여 수익률 과대계산을 방지합니다.
                sell_price = min(actual_price, exit_trigger_price)
                
                sell_value = total_shares * sell_price
                commission = sell_value * commission_rate
                cumulative_commission += commission
                capital += sell_value - commission
                print(f"!!! PROFIT PROTECTION TRIGGERED !!! Selling {total_shares} shares at {sell_price:.2f} on {date_str} (Return: {current_return:.2%})")
                total_shares = 0
                average_price = 0
                position = None
                peak_price = 0

    # 롱 포지션 청산 - 정수 단위 주문으로 수정 (매도용 예측 사용)
    should_sell = (upper_price > upper_mean)
    
    # 상승장(Bull) 특수 로직: 추세 추종 (Trend Following) 강화
    if is_bull_regime:
        # Bull regime에서는 모델이 유의미한 하락을 예측할 때만 매도 (추세 지속 시 보유)
        expected_sell_return = (hybrid_yhat_sell - actual_price) / actual_price
        should_sell = should_sell and (expected_sell_return < -0.01)

    if position == 'long' and should_sell:
        # [최적화] 상승장(Bull)에서는 이익을 길게 가져가기 위해 분할 매도(Partial Profit Taking) 적용
        if is_bull_regime:
            # 상승장에서는 매도 신호가 와도 물량의 최대 20%만 매도하여 추세 추종 (Let Profits Run)
            # fraction_sell이 높더라도 강제로 비율을 낮춤 (30% -> 20%로 더 축소하여 포지션 유지력 강화)
            sell_ratio = min(fraction_sell, 0.2)
            shares_to_sell_float = total_shares * sell_ratio
        else:
            # 횡보/하락장에서는 기존대로 신뢰도(fraction)에 따라 적극 매도 (Mean Reversion)
            shares_to_sell_float = total_shares * fraction_sell
            
        shares_to_sell = int(shares_to_sell_float)  # 정수로 내림
        
        # 최소 주문 단위 확인 (0주는 주문 불가)
        if shares_to_sell >= 1:
            # 매도 체결: 상한 바운더리 평균(upper_mean)으로 판매
            sell_value = shares_to_sell * upper_mean
            commission = shares_to_sell * upper_mean * commission_rate
            cumulative_commission += commission
            total_shares -= shares_to_sell
            # 매도 시 받는 자본: 판매 금액 - 수수료
            capital += sell_value - commission
            
            if should_log:
                print(f"Selling {shares_to_sell} shares at {upper_mean:.2f} (wanted to sell {shares_to_sell_float:.2f})")
            
            if total_shares <= 0:
                position = None
                total_shares = 0
                average_price = 0

    # Print key metrics for debugging and logging
    date_str = test_data.index[t].strftime('%Y-%m-%d')
    
    # Calculate total assets
    # [수정] 자산 가치 평가 시 오늘 시가(actual_price)가 아닌 종가(close_price)를 사용하여 현실성 강화
    if position == 'long':
        total_assets = capital + total_shares * close_price
    elif position == 'short':
        total_assets = capital + abs(total_shares) * average_price + (average_price - close_price) * total_shares
    else:
        total_assets = capital
    
    # 미실현 손익 계산 (참고용)
    if position == 'long':
        unrealized_pnl = (actual_price - average_price) * total_shares
    elif position == 'short':
        unrealized_pnl = (average_price - actual_price) * total_shares
    else:
        unrealized_pnl = 0
    results.append(total_assets)
    # 주식 수량 기록
    shares_history.append(total_shares)
    shares_dates.append(test_data.index[t])

    # Print metrics including cumulative commission
    if should_log:
        print(f"date: {date_str:>10} | Conf(Buy): {confidence_buy:>6.4f} | Conf(Sell): {confidence_sell:>6.4f} | "
              f"Frac(Buy): {fraction_buy:>5.3f} | Frac(Sell): {fraction_sell:>5.3f} | "
              f"hybrid_yhat_buy: {hybrid_yhat_buy:>10.2f} | hybrid_yhat_sell: {hybrid_yhat_sell:>10.2f} | actual_price: {actual_price:>10.2f} | "
              f"capital: {capital:>10.2f} | total_shares: {total_shares:>10.2f} | total_assets: {total_assets:>10.2f} | "
              f"position: {str(position):>6} | unrealized_pnl: {unrealized_pnl:>10.2f} | cumulative_commission: {cumulative_commission:>10.2f}")

    # [정보 누수 방지] 다음 날 시뮬레이션을 위해 최고가 업데이트 (오늘의 고가 반영)
    if position == 'long':
        peak_price = max(peak_price, upper_price)
    else:
        peak_price = 0

    # Add trade details to the trade history list
    trade_history.append({
        'date': date_str,
        'confidence_buy': confidence_buy,
        'confidence_sell': confidence_sell,
        'fraction_buy': fraction_buy,
        'fraction_sell': fraction_sell,
        'fraction': fraction,
        'hybrid_yhat_buy': hybrid_yhat_buy,
        'hybrid_yhat_sell': hybrid_yhat_sell,
        'actual_price': close_price,
        'capital': capital,
        'total_shares': total_shares,
        'total_assets': total_assets,
        'position': position,
        'unrealized_pnl': unrealized_pnl,
        'regime': current_regime  # 레짐 정보 추가
    })

# Convert trade history to DataFrame
trade_history_df = pd.DataFrame(trade_history)
# Save trade history to Excel file in results folder
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
trade_history_path = os.path.join(results_dir, f'trade_history_{timestamp}.xlsx')
trade_history_df.to_excel(trade_history_path, index=False)
print(f"Trade history saved to: {trade_history_path}")
# Calculate returns
returns = np.diff(results) / results[:-1]
# Convert daily returns to annual returns
trading_days_per_year = 252  # Typically 252 trading days in a year
annualized_return = np.mean(returns) * trading_days_per_year
annualized_std_return = np.std(returns) * np.sqrt(trading_days_per_year)
# Calculate Sharpe ratio (assuming risk-free rate is 0)
sharpe_ratio = annualized_return / annualized_std_return

# Calculate maximum drawdown
strategy_peak, strategy_max_drawdown = max_drawdown(results)
max_loss_streak = max_loss(results)

# Calculate total return over the entire period
total_return = (total_assets - initial_capital) / initial_capital
# Calculate simulation period in years
simulation_years = (simulation_end_date - simulation_start_date).days / 365.25
# Calculate annualized expected return
annualized_return = (1 + total_return) ** (1 / simulation_years) - 1

# Buy and Hold Performance Metrics (TQQQ)
initial_price = test_data[target_index].iloc[0]
final_price = test_data[target_index].iloc[-1]
bnh_total_return = (final_price - initial_price) / initial_price
bnh_annualized_return = (1 + bnh_total_return) ** (1 / simulation_years) - 1

# Calculate Buy and Hold returns for Sharpe ratio
bnh_returns = test_data[target_index].pct_change().dropna()
bnh_annualized_std = bnh_returns.std() * np.sqrt(trading_days_per_year)
bnh_sharpe_ratio = (bnh_annualized_return / bnh_annualized_std) if bnh_annualized_std > 0 else 0

# Calculate Buy and Hold drawdown
bnh_values = test_data[target_index].values
bnh_peak, bnh_max_drawdown = max_drawdown(bnh_values)

print("")
print(f"Simulation Performance: {simulation_start_date.date()} ~ {simulation_end_date.date()}")
print(f"=" * 80)

if USE_REGIME_DETECTION and 'regime' in trade_history_df.columns:
    regime_distribution = trade_history_df['regime'].value_counts(normalize=True) * 100
    regime_counts = trade_history_df['regime'].value_counts()
    print(f"MARKET REGIME DISTRIBUTION:")
    print(f"  Total Simulation Days: {len(trade_history_df)}")
    for regime, percentage in regime_distribution.items():
        count = regime_counts[regime]
        print(f"  - {regime.upper():<10}: {count:>5} days ({percentage:.2f}%)")
    print("-" * 80)

print(f"STRATEGY PERFORMANCE:")
print(f"  Final Capital: ${total_assets:,.2f} USD")
print(f"  Cumulative Return: {total_return:.2%}")
print(f"  Annualized Return: {annualized_return:.2%}")
print(f"  Return Standard Deviation: {annualized_std_return:.4f}")
print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"  Peak: ${strategy_peak:,.2f}")
print(f"  Maximum Drawdown: ${strategy_max_drawdown:,.2f}")
print(f"  Maximum Consecutive Loss Days: {max_loss_streak}")

print(f"\nBUY & HOLD ({target_index}) PERFORMANCE:")
print(f"  Final Price: ${final_price:,.2f} USD (Initial: ${initial_price:,.2f} USD)")
print(f"  Cumulative Return: {bnh_total_return:.2%}")
print(f"  Annualized Return: {bnh_annualized_return:.2%}")
print(f"  Return Standard Deviation: {bnh_annualized_std:.4f}")
print(f"  Sharpe Ratio: {bnh_sharpe_ratio:.4f}")
print(f"  Peak: ${bnh_peak:,.2f}")
print(f"  Maximum Drawdown: ${bnh_max_drawdown:,.2f}")

# Comparison
print(f"\nCOMPARISON:")
print(f"  Outperformance (Return): {(total_return - bnh_total_return):+.2%}")
print(f"  Outperformance (Annualized): {(annualized_return - bnh_annualized_return):+.2%}")
print(f"  Sharpe Ratio Difference: {sharpe_ratio - bnh_sharpe_ratio:+.4f}")
print(f"=" * 80)

# 차트 데이터 준비
chart_df = pd.DataFrame({
    'date': [test_data.index[t] for t in range(len(test_data))],
    'price': [test_data[target_index].iloc[t] for t in range(len(test_data))],
})

# shares_history를 날짜 기준으로 매칭
shares_dict = dict(zip(shares_dates, shares_history))
chart_df['shares'] = [shares_dict.get(test_data.index[t], 0) for t in range(len(test_data))]

# portfolio_value는 results를 날짜에 맞게 매칭
portfolio_value_list = []
results_idx = 0
for t in range(len(test_data)):
    current_date = test_data.index[t]
    if results_idx < len(results):
        portfolio_value_list.append(results[results_idx])
        if results_idx + 1 < len(results):
            results_idx += 1
    else:
        portfolio_value_list.append(results[-1] if results else initial_capital)
chart_df['portfolio_value'] = portfolio_value_list
chart_df.set_index('date', inplace=True)

# Add regime data from trade_history
regime_list = [item['regime'] for item in trade_history]
chart_df['regime'] = regime_list

# Buy and Hold 계산
initial_price = test_data[target_index].iloc[0]
buy_hold_shares = initial_capital / initial_price
buy_hold_commission_buy = buy_hold_shares * initial_price * commission_rate
buy_hold_actual_shares = (initial_capital - buy_hold_commission_buy) / initial_price
buy_hold_values = [buy_hold_actual_shares * price for price in chart_df['price']]
chart_df['buy_hold_value'] = buy_hold_values

# 시각화: 3패널 수직 레이아웃
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

# 패널 1: 가격 차트 + 레짐에 따른 배경색
ax1.set_ylabel('Price (USD)', fontsize=12)
ax1.set_title(f'{target_index} Price with Market Regime Background', fontsize=14, fontweight='bold')

# 레짐별 색상 정의
regime_colors = {
    'bull': 'lightgreen',
    'bear': 'lightcoral',
    'sideways': 'lightgray',
    'high_vol': 'lightgoldenrodyellow'
}

# 레짐에 따라 배경색 적용
for i in range(len(chart_df)):
    date = chart_df.index[i]
    regime = chart_df['regime'].iloc[i]
    color = regime_colors.get(regime, 'white')
    if i < len(chart_df) - 1:
        ax1.axvspan(date, chart_df.index[i+1], color=color, alpha=0.4, zorder=0)

# 가격 라인
price_line, = ax1.plot(chart_df.index, chart_df['price'], label=f'{target_index} Price', 
         linewidth=2, color='black', zorder=2)
ax1.grid(True, alpha=0.3, zorder=1)

# 레전드 추가 (배경색 + 가격 라인)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, edgecolor='gray', label=regime.capitalize()) for regime, color in regime_colors.items()]
legend_elements.append(price_line)
ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

# 패널 2: 보유 주식 수량 막대 차트
ax2.set_ylabel('Shares Held', fontsize=12)
ax2.set_title('Holdings Quantity', fontsize=14, fontweight='bold')
# 막대 그래프로 수량 표시
ax2.bar(chart_df.index, chart_df['shares'], width=1, alpha=0.6, color='orange', label='Shares Held')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend(loc='upper left', fontsize=10)

# 패널 3: 포트폴리오 가치 추이 + Buy & Hold 비교선
ax3.set_ylabel('Portfolio Value (USD)', fontsize=12)
ax3.set_xlabel('Date', fontsize=12)
ax3.set_title('Portfolio Value vs Buy & Hold', fontsize=14, fontweight='bold')
# 전략 포트폴리오 가치
ax3.plot(chart_df.index, chart_df['portfolio_value'], label='Strategy Portfolio', 
         linewidth=2, color='blue')
# Buy & Hold
ax3.plot(chart_df.index, chart_df['buy_hold_value'], label='Buy & Hold', 
         linewidth=2, color='red', linestyle='--')
# 초기 자본 기준선
ax3.axhline(y=initial_capital, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Initial Capital')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper left', fontsize=10)

# 성과 비교 텍스트 (우측 하단에 배치하여 범례와 겹치지 않도록)
strategy_return = (chart_df['portfolio_value'].iloc[-1] - initial_capital) / initial_capital * 100
buy_hold_return = (chart_df['buy_hold_value'].iloc[-1] - initial_capital) / initial_capital * 100
outperformance = strategy_return - buy_hold_return

ax3.text(0.98, 0.02, 
         f'Strategy: {strategy_return:.2f}%\nBuy & Hold: {buy_hold_return:.2f}%\nOutperformance: {outperformance:+.2f}%',
         transform=ax3.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
# Save chart to results folder
chart_path = os.path.join(results_dir, f'qqq_trading_simulation_chart_{timestamp}.png')
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
print(f"Chart saved to: {chart_path}")
plt.show()

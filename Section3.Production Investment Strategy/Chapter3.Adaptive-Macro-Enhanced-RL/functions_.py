import numpy as np
from arch import arch_model

def find_garch(residuals):
    """
    Search for the best GARCH (EGARCH) model based on AIC.
    """
    best_aic = np.inf
    best_order = None
    best_model = None

    for p in range(0, 2):
        for o in range(0, 2):
            for q in range(0, 2):
                try:
                    model = arch_model(residuals, vol='EGARCH', 
                                       p=p, o=o, q=q, rescale=True)
                    model_fit = model.fit(disp='off')
                    aic = model_fit.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, o, q)
                        best_model = model_fit
                except:
                    continue

    return best_aic, best_order, best_model

def max_drawdown(results):
    """
    Calculate maximum drawdown (amount based).
    """
    drawdowns = []
    peak = results[0]
    peak_before_max_drawdown = peak
    max_drawdown = 0

    for value in results:
        if value > peak:
            peak = value
        drawdown = peak - value
        drawdowns.append(drawdown)
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            peak_before_max_drawdown = peak
    
    return peak_before_max_drawdown, max_drawdown

def calculate_mdd_details(results, dates):
    """
    Identify independent drawdown segments and return Top 2 by percentage.
    Returns: (peak_before_max, max_dd_pct, max_dd_date, runner_up_dd_pct, runner_up_dd_date)
    """
    if results is None or dates is None or len(results) == 0:
        return 0, 0, None, 0, None

    drawdown_events = []
    current_peak = results[0]
    local_max_dd_pct = 0
    local_max_dd_date = dates[0]

    for i in range(len(results)):
        val = results[i]
        date = dates[i]

        if val > current_peak:
            # End of a potential drawdown event
            if local_max_dd_pct > 0:
                drawdown_events.append({
                    'peak': current_peak,
                    'pct': local_max_dd_pct,
                    'date': local_max_dd_date
                })
            current_peak = val
            local_max_dd_pct = 0
        else:
            if current_peak > 0:
                dd_pct = ((current_peak - val) / current_peak) * 100
                if dd_pct > local_max_dd_pct:
                    local_max_dd_pct = dd_pct
                    local_max_dd_date = date

    # Add the last ongoing drawdown
    if local_max_dd_pct > 0:
        drawdown_events.append({
            'peak': current_peak,
            'pct': local_max_dd_pct,
            'date': local_max_dd_date
        })

    # Sort to find top 2
    drawdown_events.sort(key=lambda x: x['pct'], reverse=True)

    top1 = drawdown_events[0] if len(drawdown_events) > 0 else {'peak': current_peak, 'pct': 0, 'date': None}
    top2 = drawdown_events[1] if len(drawdown_events) > 1 else {'pct': 0, 'date': None}

    return top1['peak'], top1['pct'], top1['date'], top2['pct'], top2['date']

def max_loss(results):
    """
    Calculate maximum consecutive days of loss.
    """
    loss_streaks = []
    current_streak = 0
    for i in range(1, len(results)):
        if results[i] < results[i-1]:
            current_streak += 1
        else:
            loss_streaks.append(current_streak)
            current_streak = 0
    loss_streaks.append(current_streak)
    return max(loss_streaks) if loss_streaks else 0

def get_alpha_value(model_fitted, target_index, history, method='weighted_mean'):
    """Extracts the Error Correction Speed (alpha) from the VECM.
    
    Alpha indicates convergence or divergence:
    - Alpha < 0: Convergence toward equilibrium ✓
    - Alpha > 0: Divergence away from equilibrium ✗
    - Alpha = 0: No error correction
    """
    try:
        alpha = model_fitted.alpha
        target_idx = history.columns.get_loc(target_index)
        
        if alpha.ndim == 2:
            if target_idx < alpha.shape[0] and alpha.shape[1] > 0:
                target_alphas = alpha[target_idx, :]
                negative_alphas = target_alphas[target_alphas < 0]
                
                if len(negative_alphas) > 0:
                    if method == 'weighted_mean':
                        abs_negative = np.abs(negative_alphas)
                        weights = abs_negative / np.sum(abs_negative)
                        weighted_alpha = np.sum(negative_alphas * weights)
                        return float(weighted_alpha)
                    elif method == 'sum_negative':
                        return float(np.sum(negative_alphas))
                    elif method == 'min_negative':
                        return float(np.min(negative_alphas))
                    else:
                        abs_negative = np.abs(negative_alphas)
                        weights = abs_negative / np.sum(abs_negative)
                        return float(np.sum(negative_alphas * weights))
                else:
                    return float(np.mean(target_alphas))
            else:
                return 0.0
        elif alpha.ndim == 1:
            if target_idx < len(alpha):
                return float(alpha[target_idx])
            else:
                return 0.0
        else:
            return 0.0
    except Exception as e:
        print(f"Warning: Error extracting alpha value: {e}")
        return 0.0

def get_vecm_confidence(model_fitted, target_index, history, lower_bound, upper_bound, predicted_mean):
    """Calculates the confidence level of the VECM model.
    
    Combines indicators:
    1. Interval Width: (upper - lower) / predicted_mean (Smaller is better).
    2. Residual StdDev: Lower indicates better historical fit.
    3. Alpha Absolute: Larger negative alpha indicates stronger error correction.
    """
    try:
        target_idx = history.columns.get_loc(target_index)
        
        # 1. Interval Width
        if predicted_mean > 0:
            interval_width = (upper_bound - lower_bound) / predicted_mean
        else:
            interval_width = 1.0
        
        # 2. Residual Standard Deviation
        residuals = model_fitted.resid
        if residuals.ndim == 2:
            target_residuals = residuals[:, target_idx]
        else:
            target_residuals = residuals
        
        residual_std = np.std(target_residuals)
        if predicted_mean > 0:
            normalized_residual_std = residual_std / predicted_mean
        else:
            normalized_residual_std = 1.0
        
        # 3. Absolute Alpha
        alpha_val = get_alpha_value(model_fitted, target_index, history)
        abs_alpha = abs(alpha_val) if alpha_val < 0 else 0.0
        
        # Simple weighted confidence score
        # Lower width and lower residual std = higher confidence
        # Higher absolute negative alpha = higher confidence
        raw_score = (1.0 / (1.0 + interval_width)) * 0.4 + \
                    (1.0 / (1.0 + normalized_residual_std)) * 0.4 + \
                    (min(1.0, abs_alpha * 10.0)) * 0.2
        
        return float(np.clip(raw_score, 0.1, 0.95))
    except Exception as e:
        print(f"Warning: Error calculating VECM confidence: {e}")
        return 0.5

import numpy as np
from arch import arch_model

def find_garch(residuals):
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

def max_loss(results):
    loss_streaks = []
    current_streak = 0
    for i in range(1, len(results)):
        if results[i] < results[i-1]:
            current_streak += 1
        else:
            loss_streaks.append(current_streak)
            current_streak = 0
    loss_streaks.append(current_streak)
    max_loss_streak = max(loss_streaks)
    return max_loss_streak

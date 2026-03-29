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

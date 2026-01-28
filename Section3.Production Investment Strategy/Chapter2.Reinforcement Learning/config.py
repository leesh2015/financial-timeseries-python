import os

def get_config():
    """Returns the configuration dictionary for the simulation."""
    return {
        # Data and Tickers
        'tickers': ['TQQQ', 'NQ=F', 'ZB=F', 'UNG', 'RB=F', 'BZ=F', 'CL=F'],
        'target_index': "TQQQ",
        'data_interval': '1d',
        'years_of_data': 10,
        'train_split_ratio': 0.7,

        # Simulation
        'initial_capital': 10000,
        'commission_rate': 0.0002,

        # VECM and GARCH
        'max_lags': 15,
        'alpha_method': 'weighted_mean',
        'forecast_steps_buy': 4,
        'forecast_steps_sell': 7,

        # Systems Activation
        'use_regime_detection': True,
        'use_rl_agent': True,
        'rl_blend_factor': float(os.environ.get('RL_BLEND_FACTOR', '0.6')),

        # Fraction and Regime
        'confidence_fraction_config_base': {
            'min_fraction': 0.2,
            'max_fraction': 0.8,
            'window_size': 60,
            'method': 'minmax',
            'absolute_threshold': None,
            'threshold_method': 'percentile',
            'threshold_percentile': 10
        },
        'regime_fraction_config': {
            'bull': {'min_fraction': 0.30, 'max_fraction': 0.99},
            'bear': {'min_fraction': 0.3, 'max_fraction': 0.6},
            'sideways': {'min_fraction': 0.2, 'max_fraction': 0.8},
            'high_vol': {'min_fraction': 0.25, 'max_fraction': 0.65}
        }
    }
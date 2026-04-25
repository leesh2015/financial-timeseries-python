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
        'alpha_change_threshold': 0.5,

        # Systems Activation
        'use_regime_detection': True,
        'use_rl_agent': True,
        'rl_blend_factor': float(os.environ.get('RL_BLEND_FACTOR', '0.9487')),

        # Risk Management (Selective Management)
        'risk_management': {
            'use_risk_management': False,
            'stop_loss_mult': 8.0,
            'trailing_stop_base': 0.25,
            'min_stop_loss': 0.15,
            'max_stop_loss': 0.50,
            'entry_edge_pct': 0.0
        },

        # RL Agent Policy Parameters
        'rl_policy_params': {
            'buy_confidence_threshold': 0.65,
            'sell_confidence_threshold': 0.55,
            'max_position_ratio': 0.90,
            'min_position_ratio_for_sell': 0.20,
            'max_position_size': 0.95
        },

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
            'bull': {
                'min_fraction': 0.30,         # Restored for 652% performance
                'max_fraction': 0.97,
                'sell_ratio_cap': 0.4414      # Bayesian Optimized for Bull Regime
            },
            'bear': {'min_fraction': 0.48, 'max_fraction': 0.77},
            'sideways': {'min_fraction': 0.43, 'max_fraction': 0.86},
            'high_vol': {'min_fraction': 0.47, 'max_fraction': 0.70}
        }
    }
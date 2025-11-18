"""
Configuration settings for Financial Time Series Analysis course
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class CourseConfig:
    """Configuration class for course examples"""
    
    # Default date settings
    default_start_date: Optional[str] = None  # None = 3 years ago
    default_end_date: Optional[str] = None  # None = today
    default_years_back: int = 3
    
    # Data settings
    default_interval: str = '1d'  # Daily interval
    default_frequency: str = 'D'  # Daily frequency (for asfreq)
    
    # Model settings
    default_max_lags: int = 15
    default_train_split: float = 0.7
    default_test_split: float = 0.3
    
    # Statistical test settings
    default_significance_level: float = 0.05
    default_max_lag_granger: int = 10
    
    # GARCH settings
    default_garch_p: int = 1
    default_garch_q: int = 1
    default_garch_max_p: int = 10
    default_garch_max_q: int = 10
    
    # Outlier detection
    default_outlier_threshold: float = 3.0  # Standard deviations
    
    def get_start_date(self) -> str:
        """Get start date string"""
        if self.default_start_date:
            return self.default_start_date
        return (datetime.today() - timedelta(days=365 * self.default_years_back)).strftime('%Y-%m-%d')
    
    def get_end_date(self) -> str:
        """Get end date string"""
        if self.default_end_date:
            return self.default_end_date
        return datetime.today().strftime('%Y-%m-%d')


# Global configuration instance
config = CourseConfig()


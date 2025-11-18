"""
Standardized error handling utilities
"""

import warnings
from functools import wraps
from typing import Callable, Any


def suppress_warnings(func: Callable) -> Callable:
    """
    Decorator to suppress warnings for a function.
    
    Parameters:
    -----------
    func : Callable
        Function to wrap
        
    Returns:
    --------
    Callable
        Wrapped function with warnings suppressed
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper


def handle_data_errors(func: Callable) -> Callable:
    """
    Decorator to handle common data-related errors.
    
    Parameters:
    -----------
    func : Callable
        Function to wrap
        
    Returns:
    --------
    Callable
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyError as e:
            raise ValueError(f"Required column not found: {e}")
        except ValueError as e:
            raise ValueError(f"Data validation error: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error in {func.__name__}: {str(e)}")
    return wrapper


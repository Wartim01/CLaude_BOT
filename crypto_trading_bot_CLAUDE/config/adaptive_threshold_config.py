"""
Configuration for adaptive threshold adjustment
"""
from typing import Dict, Any

# Base configuration for threshold adaptation
DEFAULT_ADAPTIVE_THRESHOLD_CONFIG = {
    # Enable/disable adaptive threshold adjustment
    "enabled": True,
    
    # Frequency of threshold checks (hours)
    "check_interval_hours": 6,
    
    # Maximum adjustment per step (in points)
    "adjustment_step": 2.0,
    
    # Maximum total adjustment from baseline (in points)
    "max_adjustment": 10.0,
    
    # Minimum allowed threshold value
    "min_threshold": 62.0,
    
    # Maximum allowed threshold value
    "max_threshold": 85.0,
    
    # Window for opportunity analysis (hours)
    "analysis_window_hours": 24,
    
    # Conditions for automatic adjustments
    "adjustment_conditions": {
        # Lower threshold if execution rate below this percentage
        "min_execution_rate": 0.2,
        
        # Lower threshold if this many near-misses occurred
        "min_near_misses": 3,
        
        # Raise threshold if execution rate above this percentage
        "max_execution_rate": 0.8,
        
        # Raise threshold if this many opportunities were executed
        "max_executions": 10
    },
    
    # Cooldown period after an adjustment (hours)
    "post_adjustment_cooldown_hours": 12
}

def get_adaptive_threshold_config(user_config: Dict = None) -> Dict[str, Any]:
    """
    Get the adaptive threshold configuration with user overrides
    
    Args:
        user_config: User configuration overrides
        
    Returns:
        Complete configuration dictionary
    """
    config = DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.copy()
    
    if user_config:
        # Override default values with user-provided values
        for key, value in user_config.items():
            if key in config:
                # For nested dictionaries, merge instead of replace
                if isinstance(config[key], dict) and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value
    
    return config

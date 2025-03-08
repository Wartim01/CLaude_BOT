
import numpy as np
from typing import Tuple, List, Union, Optional

def ensure_3d_shape(X: np.ndarray) -> np.ndarray:
    """
    Ensures that input data has 3 dimensions (batch_size, sequence_length, feature_dim)
    
    Args:
        X: Input array
        
    Returns:
        Array with 3 dimensions
    """
    if X is None:
        return None
    
    if len(X.shape) == 2:
        # If X is 2D (samples, features), reshape to 3D with sequence_length=1
        n_samples, n_features = X.shape
        return X.reshape(n_samples, 1, n_features)
    elif len(X.shape) == 1:
        # If X is 1D, reshape to 3D with sequence_length=1, feature_dim=1
        n_samples = X.shape[0]
        return X.reshape(n_samples, 1, 1)
    elif len(X.shape) == 3:
        # Already 3D, return as is
        return X
    else:
        raise ValueError(f"Cannot convert shape {X.shape} to 3D")

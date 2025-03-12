"""Evaluation metrics for model assessment.

Author: Tim Lin
Organization: DeepBioLab
License: MIT License
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def r2(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² score.
    
    Args:
        y: True values
        y_pred: Predicted values
        
    Returns:
        R² score
    """
    return round(r2_score(y.flatten(), y_pred.flatten()), 3)

def absolute_rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate absolute RMSE.
    
    Args:
        y: True values
        y_pred: Predicted values
        
    Returns:
        Absolute RMSE
    """
    return round(np.sqrt(mean_squared_error(y, y_pred)), 3)

def relative_rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate relative RMSE.
    
    Args:
        y: True values
        y_pred: Predicted values
        
    Returns:
        Relative RMSE
    """
    return round(np.sqrt(mean_squared_error(y, y_pred)) / np.std(np.array(y)), 3)
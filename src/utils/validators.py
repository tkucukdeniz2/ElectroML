"""
Input validation utilities for ElectroML.
"""

import os
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np


ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB


def allowed_file(filename: str) -> bool:
    """
    Check if file extension is allowed.
    
    Args:
        filename: Name of the file
        
    Returns:
        True if extension is allowed
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_file_size(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate file size.
    
    Args:
        file_path: Path to file
        
    Returns:
        (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return False, "File not found"
    
    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE:
        return False, f"File too large. Maximum size is {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
    
    if file_size == 0:
        return False, "File is empty"
    
    return True, None


def validate_data_format(data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate data format for electrochemical analysis.
    
    Args:
        data: DataFrame to validate
        
    Returns:
        (is_valid, error_message)
    """
    if data.empty:
        return False, "Data is empty"
    
    if data.shape[1] < 2:
        return False, "Data must have at least 2 columns (concentration and at least one measurement)"
    
    # Check first column is numeric (concentration)
    if not pd.api.types.is_numeric_dtype(data.iloc[:, 0]):
        return False, "First column must be numeric (concentration values)"
    
    # Check if all other columns are numeric (measurements)
    for col in data.columns[1:]:
        if not pd.api.types.is_numeric_dtype(data[col]):
            return False, f"Column '{col}' contains non-numeric values"
    
    # Check for NaN values
    if data.isnull().any().any():
        n_missing = data.isnull().sum().sum()
        return False, f"Data contains {n_missing} missing values"
    
    # Check for reasonable value ranges
    if (data.iloc[:, 0] < 0).any():
        return False, "Concentration values cannot be negative"
    
    return True, None


def validate_training_params(params: dict) -> Tuple[bool, Optional[str]]:
    """
    Validate training parameters.
    
    Args:
        params: Training parameters dictionary
        
    Returns:
        (is_valid, error_message)
    """
    # Validate cross-validation parameters
    if 'cv_strategy' in params:
        valid_strategies = ['kfold', 'stratified_kfold', 'loo', 'time_series', 'train_test_split']
        if params['cv_strategy'] not in valid_strategies:
            return False, f"Invalid CV strategy. Must be one of {valid_strategies}"
        
        if params['cv_strategy'] == 'kfold' and 'n_splits' in params:
            if not isinstance(params['n_splits'], int) or params['n_splits'] < 2:
                return False, "n_splits must be an integer >= 2"
        
        if params['cv_strategy'] == 'train_test_split' and 'test_size' in params:
            test_size = params['test_size']
            if not (0 < test_size < 1):
                return False, "test_size must be between 0 and 1"
    
    # Validate model parameters
    if 'models' in params:
        if not isinstance(params['models'], list) or len(params['models']) == 0:
            return False, "At least one model must be selected"
    
    # Validate random seed
    if 'random_state' in params:
        if not isinstance(params['random_state'], (int, type(None))):
            return False, "random_state must be an integer or None"
    
    return True, None


def validate_preprocessing_params(params: dict) -> Tuple[bool, Optional[str]]:
    """
    Validate preprocessing parameters.
    
    Args:
        params: Preprocessing parameters dictionary
        
    Returns:
        (is_valid, error_message)
    """
    # Validate outlier detection parameters
    if 'outlier_method' in params:
        valid_methods = ['iqr', 'zscore', 'isolation_forest', 'none']
        if params['outlier_method'] not in valid_methods:
            return False, f"Invalid outlier method. Must be one of {valid_methods}"
        
        if params['outlier_method'] == 'zscore' and 'zscore_threshold' in params:
            if not isinstance(params['zscore_threshold'], (int, float)) or params['zscore_threshold'] <= 0:
                return False, "Z-score threshold must be a positive number"
    
    # Validate normalization parameters
    if 'normalization_method' in params:
        valid_methods = ['minmax', 'zscore', 'robust', 'none']
        if params['normalization_method'] not in valid_methods:
            return False, f"Invalid normalization method. Must be one of {valid_methods}"
    
    # Validate filtering parameters
    if 'filter_type' in params:
        valid_filters = ['butterworth', 'chebyshev', 'median', 'savgol', 'none']
        if params['filter_type'] not in valid_filters:
            return False, f"Invalid filter type. Must be one of {valid_filters}"
        
        if params['filter_type'] in ['butterworth', 'chebyshev'] and 'cutoff_freq' in params:
            if not isinstance(params['cutoff_freq'], (int, float)) or params['cutoff_freq'] <= 0:
                return False, "Cutoff frequency must be a positive number"
    
    # Validate baseline correction parameters
    if 'baseline_method' in params:
        valid_methods = ['polynomial', 'als', 'moving_average', 'none']
        if params['baseline_method'] not in valid_methods:
            return False, f"Invalid baseline method. Must be one of {valid_methods}"
        
        if params['baseline_method'] == 'polynomial' and 'poly_order' in params:
            if not isinstance(params['poly_order'], int) or params['poly_order'] < 1:
                return False, "Polynomial order must be a positive integer"
    
    return True, None
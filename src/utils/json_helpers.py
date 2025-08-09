"""
JSON serialization helpers for handling special data types.
"""

import json
import math
import numpy as np
from typing import Any


def clean_nan_inf(obj: Any) -> Any:
    """
    Replace NaN and Inf values with None for JSON serialization.
    
    Args:
        obj: Object to clean
        
    Returns:
        Cleaned object safe for JSON serialization
    """
    if isinstance(obj, dict):
        return {k: clean_nan_inf(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_inf(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.floating):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return clean_nan_inf(obj.tolist())
    return obj


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    
    def default(self, obj):
        """Convert NumPy types to Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely serialize object to JSON string.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string
    """
    cleaned_obj = clean_nan_inf(obj)
    return json.dumps(cleaned_obj, cls=NumpyEncoder, **kwargs)
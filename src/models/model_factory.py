"""
Model factory for creating and configuring machine learning models.
"""

from typing import Dict, Any, Optional, List
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor, VotingRegressor, StackingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct
# Optional imports for advanced models
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except (ImportError, Exception):
    HAS_XGBOOST = False
    
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, Exception):
    HAS_LIGHTGBM = False
# TensorFlow imports (optional - commented out for compatibility)
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.regularizers import l1_l2
import logging

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating configured ML models."""
    
    # Model configurations with default parameters
    MODEL_CONFIGS = {
        'linear_regression': {
            'class': LinearRegression,
            'params': {},
            'param_space': {}
        },
        'ridge': {
            'class': Ridge,
            'params': {'alpha': 1.0},
            'param_space': {
                'alpha': {'type': 'float', 'low': 0.001, 'high': 100.0, 'log': True}
            }
        },
        'lasso': {
            'class': Lasso,
            'params': {'alpha': 1.0},
            'param_space': {
                'alpha': {'type': 'float', 'low': 0.001, 'high': 10.0, 'log': True}
            }
        },
        'elastic_net': {
            'class': ElasticNet,
            'params': {'alpha': 1.0, 'l1_ratio': 0.5},
            'param_space': {
                'alpha': {'type': 'float', 'low': 0.001, 'high': 10.0, 'log': True},
                'l1_ratio': {'type': 'float', 'low': 0.1, 'high': 0.9}
            }
        },
        'svm': {
            'class': SVR,
            'params': {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'},
            'param_space': {
                'C': {'type': 'float', 'low': 0.1, 'high': 100.0, 'log': True},
                'gamma': {'type': 'categorical', 'choices': ['scale', 'auto']},
                'kernel': {'type': 'categorical', 'choices': ['linear', 'rbf', 'poly', 'sigmoid']}
            }
        },
        'random_forest': {
            'class': RandomForestRegressor,
            'params': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            'param_space': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 5}
            }
        },
        'extra_trees': {
            'class': ExtraTreesRegressor,
            'params': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'random_state': 42
            },
            'param_space': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 10}
            }
        },
        'gradient_boosting': {
            'class': GradientBoostingRegressor,
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            },
            'param_space': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'max_depth': {'type': 'int', 'low': 2, 'high': 10}
            }
        },
        'xgboost': {
            'class': xgb.XGBRegressor if HAS_XGBOOST else None,
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'objective': 'reg:squarederror',
                'random_state': 42
            },
            'param_space': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0}
            }
        },
        'lightgbm': {
            'class': lgb.LGBMRegressor if HAS_LIGHTGBM else None,
            'params': {
                'n_estimators': 100,
                'num_leaves': 31,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbosity': -1
            },
            'param_space': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'num_leaves': {'type': 'int', 'low': 20, 'high': 100},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'min_child_samples': {'type': 'int', 'low': 5, 'high': 30}
            }
        },
        'adaboost': {
            'class': AdaBoostRegressor,
            'params': {
                'n_estimators': 50,
                'learning_rate': 1.0,
                'random_state': 42
            },
            'param_space': {
                'n_estimators': {'type': 'int', 'low': 30, 'high': 200},
                'learning_rate': {'type': 'float', 'low': 0.1, 'high': 2.0}
            }
        },
        'mlp': {
            'class': MLPRegressor,
            'params': {
                'hidden_layer_sizes': (50, 25),  # Smaller network for small datasets
                'activation': 'tanh',  # Often better for regression
                'solver': 'lbfgs',  # Better for small datasets than adam
                'alpha': 0.01,  # Higher regularization for small datasets
                'learning_rate': 'constant',
                'learning_rate_init': 0.001,
                'max_iter': 2000,  # More iterations
                'early_stopping': True,  # Prevent overfitting
                'validation_fraction': 0.1,
                'n_iter_no_change': 20,
                'random_state': 42,
                'tol': 0.0001
            },
            'param_space': {
                'hidden_layer_sizes': {'type': 'categorical', 
                                      'choices': [(20,), (50,), (30, 15), (50, 25)]},
                'activation': {'type': 'categorical', 'choices': ['tanh', 'relu', 'logistic']},
                'alpha': {'type': 'float', 'low': 0.001, 'high': 0.1, 'log': True},
                'learning_rate_init': {'type': 'float', 'low': 0.0001, 'high': 0.01, 'log': True}
            }
        },
        'neural_network': {  # Alias for MLP with different config
            'class': MLPRegressor,
            'params': {
                'hidden_layer_sizes': (64, 32, 16),  # Deeper but narrower
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'learning_rate': 'adaptive',
                'learning_rate_init': 0.001,
                'max_iter': 3000,
                'early_stopping': True,
                'validation_fraction': 0.15,
                'n_iter_no_change': 50,
                'random_state': 42,
                'batch_size': 'auto',
                'tol': 0.0001
            },
            'param_space': {
                'hidden_layer_sizes': {'type': 'categorical', 
                                      'choices': [(32,), (64, 32), (64, 32, 16), (128, 64, 32)]},
                'activation': {'type': 'categorical', 'choices': ['relu', 'tanh']},
                'alpha': {'type': 'float', 'low': 0.0001, 'high': 0.1, 'log': True},
                'learning_rate_init': {'type': 'float', 'low': 0.0001, 'high': 0.01, 'log': True}
            }
        },
        'gaussian_process': {
            'class': GaussianProcessRegressor,
            'params': {
                'kernel': RBF(length_scale=1.0),
                'alpha': 1e-10,
                'normalize_y': True,
                'random_state': 42
            },
            'param_space': {}  # GP hyperparameters are optimized internally
        }
    }
    
    
    @classmethod
    def create_model(cls, model_name: str, custom_params: Optional[Dict] = None) -> Any:
        """
        Create a model instance with specified parameters.
        
        Args:
            model_name: Name of the model to create
            custom_params: Optional custom parameters to override defaults
            
        Returns:
            Configured model instance
        """
        if model_name not in cls.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(cls.MODEL_CONFIGS.keys())}")
        
        config = cls.MODEL_CONFIGS[model_name]
        params = config['params'].copy()
        
        if custom_params:
            params.update(custom_params)
        
        if model_name == 'ann':
            return cls._create_keras_model(params)
        
        return config['class'](**params)
    
    @classmethod
    def get_param_space(cls, model_name: str) -> Dict:
        """Get hyperparameter search space for a model."""
        if model_name not in cls.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        
        return cls.MODEL_CONFIGS[model_name]['param_space']
    
    @classmethod
    def _create_keras_model(cls, params: Dict):
        """Create a Keras neural network model."""
        # Keras/TensorFlow support is optional
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.regularizers import l1_l2
            
            model = Sequential()
            
            # Input layer
            model.add(Dense(
                params.get('layer1_units', 64),
                activation='relu',
                input_shape=(params.get('input_dim', 10),),
                kernel_regularizer=l1_l2(l1=params.get('l1', 0.01), l2=params.get('l2', 0.01))
            ))
            
            if params.get('use_dropout', True):
                model.add(Dropout(params.get('dropout_rate', 0.2)))
            
            if params.get('use_batch_norm', True):
                model.add(BatchNormalization())
            
            # Hidden layers
            for i in range(params.get('n_hidden_layers', 2)):
                units = params.get(f'layer{i+2}_units', 32)
                model.add(Dense(
                    units,
                    activation='relu',
                    kernel_regularizer=l1_l2(l1=params.get('l1', 0.01), l2=params.get('l2', 0.01))
                ))
                
                if params.get('use_dropout', True):
                    model.add(Dropout(params.get('dropout_rate', 0.2)))
                
                if params.get('use_batch_norm', True):
                    model.add(BatchNormalization())
            
            # Output layer
            model.add(Dense(1))
            
            # Compile
            model.compile(
                optimizer=Adam(learning_rate=params.get('learning_rate', 0.001)),
                loss='mse',
                metrics=['mae', 'mse']
            )
            
            return model
        except ImportError:
            raise ImportError("TensorFlow is not installed. Please install tensorflow to use neural network models.")
    
    @classmethod
    def create_ensemble(cls, base_models: List[str], ensemble_type: str = 'voting') -> Any:
        """
        Create an ensemble model.
        
        Args:
            base_models: List of base model names
            ensemble_type: Type of ensemble ('voting', 'stacking')
            
        Returns:
            Ensemble model instance
        """
        estimators = []
        for model_name in base_models:
            model = cls.create_model(model_name)
            estimators.append((model_name, model))
        
        if ensemble_type == 'voting':
            return VotingRegressor(estimators=estimators)
        elif ensemble_type == 'stacking':
            return StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(),
                cv=5
            )
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model names."""
        available = []
        for name, config in cls.MODEL_CONFIGS.items():
            if config['class'] is not None:
                available.append(name)
        return available
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict:
        """Get information about a specific model."""
        if model_name not in cls.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = cls.MODEL_CONFIGS[model_name]
        return {
            'name': model_name,
            'class': config['class'].__name__,
            'default_params': config['params'],
            'tunable_params': list(config['param_space'].keys())
        }
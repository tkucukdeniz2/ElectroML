"""
Model training module with advanced ML algorithms and hyperparameter optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import (
    LeaveOneOut, KFold,
    cross_val_score, cross_validate
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    scaler_type: str = 'minmax'  # 'standard', 'minmax', 'robust'
    cv_strategy: str = 'loo'  # 'loo', 'kfold'
    n_splits: int = 5
    random_state: int = 42
    optimize_hyperparams: bool = True
    n_trials: int = 100
    n_jobs: int = -1
    verbose: bool = False


@dataclass
class TrainingResult:
    """Container for training results."""
    model: Any
    scaler_X: Any
    scaler_y: Any
    metrics: Dict[str, float]
    cv_scores: Dict[str, np.ndarray]
    predictions: np.ndarray
    actual_values: np.ndarray
    feature_importance: Optional[np.ndarray] = None
    best_params: Optional[Dict] = None
    optimization_history: Optional[pd.DataFrame] = None


class ModelTrainer:
    """Advanced model training with hyperparameter optimization."""
    
    def __init__(self, config: TrainingConfig = None):
        """Initialize model trainer with configuration."""
        self.config = config or TrainingConfig()
        self.models = {}
        self.results = {}
        
    def train_model(self, 
                   model_class: Any,
                   X: pd.DataFrame,
                   y: pd.Series,
                   model_name: str,
                   param_space: Optional[Dict] = None) -> TrainingResult:
        """
        Train a model with optional hyperparameter optimization.
        
        Args:
            model_class: Model class or factory function
            X: Feature matrix
            y: Target values
            model_name: Name identifier for the model
            param_space: Parameter space for optimization
            
        Returns:
            TrainingResult object with trained model and metrics
        """
        logger.info(f"Training {model_name}...")

        X_arr = X.values if hasattr(X, 'values') else X
        y_arr = y.values if hasattr(y, 'values') else y

        # Optimize hyperparameters if requested
        best_params = None
        optimization_history = None

        if self.config.optimize_hyperparams and param_space:
            best_params, optimization_history = self._optimize_hyperparameters(
                model_class, X_arr, y_arr, param_space, model_name
            )
            base_model = model_class(**best_params)
        else:
            base_model = model_class()

        # Create pipeline: scaling is performed INSIDE each CV fold
        scaler_X, scaler_y = self._create_scalers()
        pipeline = SkPipeline([('scaler', scaler_X), ('model', base_model)])

        # Cross-validation with pipeline (scaling per fold, no leakage)
        cv_scores = self._perform_cross_validation(pipeline, X_arr, y_arr)

        # Train final model on full dataset
        pipeline.fit(X_arr, y_arr)

        # Use out-of-fold predictions for unbiased evaluation
        from sklearn.model_selection import cross_val_predict
        cv_splitter = self._get_cv_splitter(len(X_arr))
        predictions = cross_val_predict(pipeline, X_arr, y_arr, cv=cv_splitter)

        # Calculate metrics from CV predictions (unbiased)
        metrics = self._calculate_metrics(y_arr, predictions)

        # Extract feature importance from the trained pipeline's model step
        trained_model = pipeline.named_steps['model']
        feature_importance = self._extract_feature_importance(
            trained_model, X.columns if hasattr(X, 'columns') else None
        )

        result = TrainingResult(
            model=pipeline,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            metrics=metrics,
            cv_scores=cv_scores,
            predictions=predictions,
            actual_values=y_arr,
            feature_importance=feature_importance,
            best_params=best_params,
            optimization_history=optimization_history
        )
        
        self.results[model_name] = result
        logger.info(f"Training completed. R2 Score: {metrics['r2']:.4f}")
        
        return result
    
    def _create_scalers(self) -> Tuple[Any, Any]:
        """Create data scalers based on configuration."""
        scaler_map = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler
        }
        
        scaler_class = scaler_map.get(self.config.scaler_type, MinMaxScaler)
        return scaler_class(), scaler_class()
    
    def _get_cv_splitter(self, n_samples: int):
        """Get cross-validation splitter based on strategy.

        Supports 'loo' (Leave-One-Out, default) and 'kfold' (K-Fold).
        """
        if self.config.cv_strategy == 'kfold':
            return KFold(n_splits=min(self.config.n_splits, n_samples),
                        shuffle=True,
                        random_state=self.config.random_state)
        else:  # Default to LOO
            return LeaveOneOut()
    
    def _perform_cross_validation(self, estimator: Any, X: np.ndarray, y: np.ndarray) -> Dict:
        """Perform cross-validation and return scores.

        The estimator should be a Pipeline so that scaling occurs within each fold.
        """
        cv = self._get_cv_splitter(len(X))

        scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']

        cv_results = cross_validate(
            estimator, X, y, cv=cv, scoring=scoring,
            n_jobs=self.config.n_jobs,
            return_train_score=True
        )

        return {
            'train_r2': cv_results['train_r2'],
            'test_r2': cv_results['test_r2'],
            'train_mse': -cv_results['train_neg_mean_squared_error'],
            'test_mse': -cv_results['test_neg_mean_squared_error'],
            'train_mae': -cv_results['train_neg_mean_absolute_error'],
            'test_mae': -cv_results['test_neg_mean_absolute_error']
        }
    
    def _optimize_hyperparameters(self,
                                 model_class: Any,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 param_space: Dict,
                                 model_name: str) -> Tuple[Dict, pd.DataFrame]:
        """Optimize hyperparameters using Optuna with Pipeline-based CV.

        Each trial wraps the model in a Pipeline with StandardScaler so that
        scaling occurs within each CV fold, preventing information leakage.
        """
        logger.info(f"Optimizing hyperparameters for {model_name}...")

        def objective(trial):
            # Sample parameters from space
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        step=param_config.get('step', 1)
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )

            # Create Pipeline with scaling inside CV
            model = model_class(**params)
            scaler = self._create_scalers()[0]
            pipe = SkPipeline([('scaler', scaler), ('model', model)])
            cv = self._get_cv_splitter(len(X))
            scores = cross_val_score(pipe, X, y, cv=cv, scoring='r2', n_jobs=1)

            return scores.mean()
        
        # Create study
        sampler = TPESampler(seed=self.config.random_state)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=f'{model_name}_optimization'
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            show_progress_bar=self.config.verbose
        )
        
        # Get optimization history
        history = study.trials_dataframe()
        
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best CV R2 score: {study.best_value:.4f}")
        
        return study.best_params, history
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = mean_absolute_percentage_error(y_true, y_pred) if not np.any(y_true == 0) else np.nan
        explained_var = explained_variance_score(y_true, y_pred)
        
        # Normalized metrics
        y_range = y_true.max() - y_true.min()
        nrmse = rmse / y_range if y_range > 0 else np.nan
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'explained_variance': explained_var,
            'nrmse': nrmse
        }
    
    def _extract_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[np.ndarray]:
        """Extract feature importance if available."""
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_)
        else:
            return None
    
    def save_model(self, model_name: str, filepath: str):
        """Save trained model to file."""
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        result = self.results[model_name]
        joblib.dump({
            'model': result.model,
            'scaler_X': result.scaler_X,
            'scaler_y': result.scaler_y,
            'metrics': result.metrics,
            'best_params': result.best_params,
            'feature_importance': result.feature_importance
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> Dict:
        """Load model from file."""
        return joblib.load(filepath)
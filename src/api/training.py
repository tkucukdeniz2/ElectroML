"""
Enhanced model training API with comprehensive configuration options.
"""

import numpy as np
import pandas as pd
from flask import request, jsonify
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut, TimeSeriesSplit, 
    train_test_split, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

from api import training_bp
from core.feature_extraction import FeatureExtractor
from core.model_training import ModelTrainer, TrainingConfig
from models.model_factory import ModelFactory
from utils.session_manager import session_manager
from utils.validators import validate_training_params
from utils.json_helpers import clean_nan_inf

logger = logging.getLogger(__name__)


class EnhancedTrainer:
    """Enhanced model trainer with configurable parameters."""
    
    @staticmethod
    def get_cv_splitter(cv_strategy: str, n_samples: int, **kwargs):
        """
        Get cross-validation splitter based on strategy.
        
        Args:
            cv_strategy: Type of CV strategy
            n_samples: Number of samples
            **kwargs: Strategy-specific parameters
        """
        if cv_strategy == 'kfold':
            n_splits = min(kwargs.get('n_splits', 5), n_samples)
            shuffle = kwargs.get('shuffle', True)
            random_state = kwargs.get('random_state', 42)
            return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        elif cv_strategy == 'stratified_kfold':
            n_splits = min(kwargs.get('n_splits', 5), n_samples)
            shuffle = kwargs.get('shuffle', True)
            random_state = kwargs.get('random_state', 42)
            return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        elif cv_strategy == 'loo':
            return LeaveOneOut()
        
        elif cv_strategy == 'time_series':
            n_splits = min(kwargs.get('n_splits', 5), n_samples - 1)
            return TimeSeriesSplit(n_splits=n_splits)
        
        else:  # Default to KFold
            return KFold(n_splits=min(5, n_samples), shuffle=True, random_state=42)
    
    @staticmethod
    def train_with_hyperparameter_tuning(model_name: str, X, y, param_grid: dict, 
                                        cv_splitter, search_type: str = 'grid'):
        """
        Train model with hyperparameter tuning.
        
        Args:
            model_name: Name of the model
            X: Features
            y: Target
            param_grid: Parameter grid for search
            cv_splitter: Cross-validation splitter
            search_type: 'grid' or 'random'
        """
        base_model = ModelFactory.create_model(model_name)
        
        if search_type == 'random':
            n_iter = min(20, np.prod([len(v) if isinstance(v, list) else 10 
                                     for v in param_grid.values()]))
            search = RandomizedSearchCV(
                base_model, param_grid, n_iter=n_iter, 
                cv=cv_splitter, scoring='r2', n_jobs=-1
            )
        else:
            search = GridSearchCV(
                base_model, param_grid, 
                cv=cv_splitter, scoring='r2', n_jobs=-1
            )
        
        search.fit(X, y)
        
        return {
            'best_model': search.best_estimator_,
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }


@training_bp.route('/extract_features', methods=['POST'])
def extract_features():
    """Extract features with configurable options."""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Invalid session'}), 400
        
        # Get data (preprocessed if available)
        df = session.get('preprocessed_data', session.get('data'))
        if df is None:
            return jsonify({'error': 'No data available'}), 400
        
        # Feature extraction parameters
        feature_config = {
            'smoothing': data.get('smoothing', True),
            'window_length': data.get('window_length', 11),
            'polyorder': data.get('polyorder', 3),
            'extract_statistical': data.get('extract_statistical', True),
            'extract_peak': data.get('extract_peak', True),
            'extract_frequency': data.get('extract_frequency', True),
            'extract_derivative': data.get('extract_derivative', True)
        }
        
        # Extract voltages from column names
        # Handle both numeric columns and string-formatted columns (e.g., 'V_neg1_000')
        voltages = []
        for col in df.columns[1:]:
            try:
                # Try direct float conversion first
                voltages.append(float(col))
            except ValueError:
                # Handle string format like 'V_neg1_000' -> -1.000
                if isinstance(col, str) and col.startswith('V_'):
                    # Remove 'V_' prefix and replace 'neg' with '-'
                    voltage_str = col[2:].replace('neg', '-').replace('_', '.')
                    try:
                        voltages.append(float(voltage_str))
                    except ValueError:
                        # If still can't convert, use index-based voltage range
                        logger.warning(f"Could not parse voltage from column name: {col}")
                        voltages.append(None)
                else:
                    voltages.append(None)
        
        # If we couldn't parse voltages, create a default range from -1 to 1
        if None in voltages or len(voltages) == 0:
            n_points = len(df.columns) - 1
            voltages = np.linspace(-1, 1, n_points)
            logger.info(f"Using default voltage range: -1V to 1V with {n_points} points")
        else:
            voltages = np.array(voltages)
        
        # Initialize feature extractor
        extractor = FeatureExtractor(
            smoothing=feature_config['smoothing'],
            window_length=feature_config['window_length'],
            polyorder=feature_config['polyorder']
        )
        
        # Extract features
        features_df = extractor.extract_features(df, voltages)
        
        # Feature selection if requested
        if data.get('feature_selection', False):
            n_features = data.get('n_features_to_select', 20)
            from sklearn.feature_selection import SelectKBest, f_regression
            
            X = features_df.drop(columns=['Concentration'])
            y = features_df['Concentration']
            
            selector = SelectKBest(f_regression, k=min(n_features, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = X.columns[selector.get_support()].tolist()
            features_df = pd.DataFrame(X_selected, columns=selected_features)
            features_df['Concentration'] = y.values
        
        # Store features
        session_manager.update_session(session_id, 'features', features_df)
        session_manager.update_session(session_id, 'feature_config', feature_config)
        
        # Calculate feature importance
        from sklearn.ensemble import RandomForestRegressor
        X = features_df.drop(columns=['Concentration'])
        y = features_df['Concentration']
        
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Log voltage parsing info
        logger.info(f"Parsed {len(voltages)} voltage points from {voltages[0]:.3f}V to {voltages[-1]:.3f}V")
        
        response = {
            'session_id': session_id,
            'n_features': len(X.columns),
            'n_samples': len(features_df),
            'feature_names': X.columns.tolist(),
            'feature_importance': feature_importance.head(20).to_dict('records'),
            'top_features': feature_importance.head(10).to_dict('records'),  # For tooltips
            'feature_statistics': {
                'mean': X.mean().to_dict(),
                'std': X.std().to_dict()
            },
            'voltage_range': {
                'min': float(voltages[0]),
                'max': float(voltages[-1]),
                'n_points': len(voltages)
            },
            'educational_note': f'Extracted {len(X.columns)} features from {len(voltages)} voltage measurements. Feature importance shows which characteristics best predict concentration.'
        }
        
        return jsonify(clean_nan_inf(response)), 200
        
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return jsonify({'error': str(e)}), 500


@training_bp.route('/train', methods=['POST'])
def train_models():
    """Train models with comprehensive configuration options."""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        # Validate parameters
        is_valid, error_msg = validate_training_params(data)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        session = session_manager.get_session(session_id)
        if not session or session['features'] is None:
            return jsonify({'error': 'Invalid session or features not extracted'}), 400
        
        features_df = session['features']
        X = features_df.drop(columns=['Concentration'])
        y = features_df['Concentration']
        
        # Get training configuration
        models_to_train = data.get('models', ['linear_regression'])
        cv_strategy = data.get('cv_strategy', 'loo')  # LOO is default for electrochemical data
        hyperparameter_tuning = data.get('hyperparameter_tuning', False)
        custom_params = data.get('custom_params', {})
        
        # Handle train/test split if requested
        if cv_strategy == 'train_test_split':
            test_size = data.get('test_size', 0.2)
            random_state = data.get('random_state', 42)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = None, None
        
        # Get CV splitter
        trainer = EnhancedTrainer()
        # Pass CV parameters without duplicating cv_strategy
        cv_params = {k: v for k, v in data.items() 
                    if k not in ['cv_strategy', 'session_id', 'models', 'hyperparameter_tuning', 
                                 'custom_params', 'test_size']}
        cv_splitter = trainer.get_cv_splitter(cv_strategy, len(X_train), **cv_params)
        
        # Store training configuration
        session_manager.update_session(session_id, 'training_config', data)
        
        results = {}
        
        for model_name in models_to_train:
            try:
                logger.info(f"Training {model_name}...")
                
                if hyperparameter_tuning and model_name in custom_params:
                    # Train with hyperparameter tuning
                    param_grid = custom_params[model_name]
                    search_type = data.get('search_type', 'grid')
                    
                    tuning_result = trainer.train_with_hyperparameter_tuning(
                        model_name, X_train, y_train, param_grid, cv_splitter, search_type
                    )
                    
                    model = tuning_result['best_model']
                    best_params = tuning_result['best_params']
                    cv_score = tuning_result['best_score']
                    
                else:
                    # Train with default or custom parameters
                    model_params = custom_params.get(model_name, {})
                    
                    # For neural networks, ensure data is scaled
                    if model_name in ['mlp', 'neural_network']:
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.pipeline import Pipeline
                        
                        # Create pipeline with scaling
                        model = Pipeline([
                            ('scaler', StandardScaler()),
                            ('model', ModelFactory.create_model(model_name, model_params))
                        ])
                    else:
                        model = ModelFactory.create_model(model_name, model_params)
                    
                    # Perform cross-validation
                    from sklearn.model_selection import cross_val_score
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, scoring='r2')
                    cv_score = cv_scores.mean()
                    best_params = model_params
                    
                    # Train final model
                    model.fit(X_train, y_train)
                
                # Calculate metrics
                if X_test is not None:
                    # Use test set
                    y_pred = model.predict(X_test)
                    metrics = {
                        'r2': r2_score(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'mae': mean_absolute_error(y_test, y_pred)
                    }
                    test_predictions = y_pred
                else:
                    # Use training set
                    y_pred = model.predict(X_train)
                    metrics = {
                        'r2': r2_score(y_train, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_train, y_pred)),
                        'mae': mean_absolute_error(y_train, y_pred)
                    }
                    test_predictions = None
                
                # Store results with actual and predicted values
                results[model_name] = {
                    'metrics': metrics,
                    'cv_score': float(cv_score),
                    'best_params': best_params,
                    'training_size': len(X_train),
                    'test_size': len(X_test) if X_test is not None else 0,
                    'actual_values': (y_test.tolist() if X_test is not None else y_train.tolist()),
                    'predicted_values': y_pred.tolist()
                }
                
                # Store model
                if 'models' not in session:
                    session['models'] = {}
                session['models'][model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred,
                    'test_predictions': test_predictions
                }
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        # Update session
        session_manager.update_session(session_id, 'models', session['models'])
        
        # Add educational context for chemists
        model_descriptions = {
            'linear_regression': 'Linear calibration model - equivalent to traditional calibration curves',
            'ridge': 'Regularized linear model - prevents overfitting by penalizing large coefficients',
            'lasso': 'Feature-selecting linear model - automatically identifies important features',
            'elastic_net': 'Combined Ridge+Lasso - balances feature selection and coefficient shrinkage',
            'random_forest': 'Ensemble of decision trees - excellent for non-linear voltammetric relationships',
            'gradient_boosting': 'Sequential tree ensemble - highest accuracy but slower training',
            'xgboost': 'Optimized gradient boosting - industry standard for competitions',
            'lightgbm': 'Fast gradient boosting - efficient for large datasets',
            'svm': 'Support Vector Machine - finds optimal separation hyperplane',
            'neural_network': 'Deep learning model - universal function approximator'
        }
        
        cv_descriptions = {
            'kfold': 'K-Fold: Splits data into k parts, trains on k-1, validates on 1',
            'loo': 'Leave-One-Out: Each sample tested individually - gold standard for small datasets',
            'stratified_kfold': 'Stratified K-Fold: Maintains concentration distribution in each fold',
            'time_series': 'Time Series Split: Respects temporal order for degradation studies',
            'train_test_split': 'Simple split: Fast but less robust validation'
        }
        
        # Find best model
        best_model = None
        best_r2 = -float('inf')
        for name, result in results.items():
            if 'metrics' in result and result['metrics']['r2'] > best_r2:
                best_r2 = result['metrics']['r2']
                best_model = name
        
        # Interpret R² score
        if best_r2 > 0.95:
            performance_interpretation = 'Excellent - comparable to standard analytical methods'
        elif best_r2 > 0.90:
            performance_interpretation = 'Very good - suitable for quantitative analysis'
        elif best_r2 > 0.80:
            performance_interpretation = 'Good - acceptable for semi-quantitative work'
        else:
            performance_interpretation = 'Moderate - consider data quality or feature engineering'
        
        response = {
            'session_id': session_id,
            'results': results,
            'cv_strategy': cv_strategy,
            'n_models_trained': len([r for r in results.values() if 'error' not in r]),
            'best_model': best_model,
            'best_model_description': model_descriptions.get(best_model, ''),
            'cv_description': cv_descriptions.get(cv_strategy, ''),
            'performance_interpretation': performance_interpretation,
            'educational_note': f'Cross-validation ensures your model generalizes to new samples. {cv_descriptions.get(cv_strategy, "")} is being used.'
        }
        
        return jsonify(clean_nan_inf(response)), 200
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@training_bp.route('/ensemble', methods=['POST'])
def create_ensemble():
    """Create ensemble model from trained models."""
    try:
        data = request.json
        session_id = data.get('session_id')
        base_models = data.get('base_models', [])
        ensemble_type = data.get('ensemble_type', 'voting')
        
        session = session_manager.get_session(session_id)
        if not session or 'models' not in session:
            return jsonify({'error': 'No trained models available'}), 400
        
        # Create ensemble
        ensemble = ModelFactory.create_ensemble(base_models, ensemble_type)
        
        # Get data for training
        features_df = session['features']
        X = features_df.drop(columns=['Concentration'])
        y = features_df['Concentration']
        
        # Train ensemble
        ensemble.fit(X, y)
        
        # Calculate metrics
        y_pred = ensemble.predict(X)
        metrics = {
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred)
        }
        
        # Store ensemble
        session['models'][f'ensemble_{ensemble_type}'] = {
            'model': ensemble,
            'metrics': metrics,
            'base_models': base_models
        }
        
        session_manager.update_session(session_id, 'models', session['models'])
        
        response = {
            'session_id': session_id,
            'ensemble_type': ensemble_type,
            'base_models': base_models,
            'metrics': metrics
        }
        
        return jsonify(clean_nan_inf(response)), 200
        
    except Exception as e:
        logger.error(f"Ensemble creation error: {e}")
        return jsonify({'error': str(e)}), 500
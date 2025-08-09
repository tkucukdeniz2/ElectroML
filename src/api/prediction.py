"""
Enhanced prediction API with flexible test dataset handling.
"""

import os
import pandas as pd
import numpy as np
from flask import request, jsonify, current_app
from werkzeug.utils import secure_filename
import logging

from api import prediction_bp
from utils.session_manager import session_manager
from utils.validators import allowed_file, validate_data_format
from utils.json_helpers import clean_nan_inf
from core.feature_extraction import FeatureExtractor

logger = logging.getLogger(__name__)


@prediction_bp.route('/predict', methods=['POST'])
def make_predictions():
    """Make predictions with flexible input options."""
    try:
        session_id = request.form.get('session_id') or request.json.get('session_id')
        model_name = request.form.get('model_name') or request.json.get('model_name')
        prediction_mode = request.form.get('mode', 'file')  # 'file', 'manual', 'validation'
        
        session = session_manager.get_session(session_id)
        if not session or 'models' not in session:
            return jsonify({'error': 'No trained models available'}), 400
        
        if model_name not in session['models']:
            return jsonify({'error': f'Model {model_name} not found'}), 404
        
        model_data = session['models'][model_name]
        model = model_data['model']
        
        # Get test data based on mode
        if prediction_mode == 'file':
            # Upload new test file
            if 'file' not in request.files:
                return jsonify({'error': 'No test file provided'}), 400
            
            file = request.files['file']
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            
            # Save and load test file
            filename = secure_filename(file.filename)
            upload_folder = current_app.config.get('UPLOAD_FOLDER', 
                                                  os.path.join(os.path.dirname(__file__), '..', 'uploads'))
            test_path = os.path.join(upload_folder, f"test_{session_id}_{filename}")
            file.save(test_path)
            
            # Load test data
            if filename.endswith('.csv'):
                test_data = pd.read_csv(test_path)
            else:
                test_data = pd.read_excel(test_path)
            
            # Convert column names to strings
            test_data.columns = [str(col) for col in test_data.columns]
            
            # Validate format
            is_valid, error_msg = validate_data_format(test_data)
            if not is_valid:
                os.remove(test_path)
                return jsonify({'error': error_msg}), 400
            
            # Apply same preprocessing as training data
            if 'preprocessing_config' in session:
                test_data = apply_preprocessing(test_data, session['preprocessing_config'])
            
            # Extract features using same configuration
            feature_config = session.get('feature_config', {})
            voltages = np.array([float(col) for col in test_data.columns[1:]])
            
            extractor = FeatureExtractor(
                smoothing=feature_config.get('smoothing', True),
                window_length=feature_config.get('window_length', 11),
                polyorder=feature_config.get('polyorder', 3)
            )
            
            features_df = extractor.extract_features(test_data, voltages)
            X_test = features_df.drop(columns=['Concentration'])
            y_test = features_df['Concentration'] if 'Concentration' in features_df else None
            
        elif prediction_mode == 'manual':
            # Manual data entry
            manual_data = request.json.get('manual_data')
            if not manual_data:
                return jsonify({'error': 'No manual data provided'}), 400
            
            # Convert to DataFrame
            X_test = pd.DataFrame(manual_data)
            y_test = X_test.pop('Concentration') if 'Concentration' in X_test.columns else None
            
        elif prediction_mode == 'validation':
            # Use validation split from training
            if 'test_predictions' not in model_data or model_data['test_predictions'] is None:
                return jsonify({'error': 'No validation data available for this model'}), 400
            
            # Return stored validation predictions
            response = {
                'session_id': session_id,
                'model_name': model_name,
                'predictions': model_data['test_predictions'].tolist(),
                'mode': 'validation',
                'n_samples': len(model_data['test_predictions'])
            }
            return jsonify(clean_nan_inf(response)), 200
        
        else:
            return jsonify({'error': f'Invalid prediction mode: {prediction_mode}'}), 400
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate confidence intervals (using bootstrap if possible)
        confidence_intervals = calculate_confidence_intervals(model, X_test)
        
        # Calculate metrics if ground truth is available
        metrics = None
        if y_test is not None:
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            metrics = {
                'r2': r2_score(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'mae': mean_absolute_error(y_test, predictions)
            }
        
        # Prepare response
        response = {
            'session_id': session_id,
            'model_name': model_name,
            'predictions': predictions.tolist(),
            'confidence_intervals': confidence_intervals,
            'mode': prediction_mode,
            'n_samples': len(predictions),
            'metrics': metrics,
            'feature_names': X_test.columns.tolist()
        }
        
        # Store predictions for download
        session_manager.update_session(session_id, 'last_predictions', {
            'predictions': predictions,
            'confidence_intervals': confidence_intervals,
            'test_data': X_test,
            'ground_truth': y_test
        })
        
        return jsonify(clean_nan_inf(response)), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@prediction_bp.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Make predictions on multiple datasets."""
    try:
        data = request.json
        session_id = data.get('session_id')
        model_name = data.get('model_name')
        batch_data = data.get('batch_data')  # List of datasets
        
        session = session_manager.get_session(session_id)
        if not session or 'models' not in session:
            return jsonify({'error': 'No trained models available'}), 400
        
        model = session['models'][model_name]['model']
        
        batch_results = []
        for i, dataset in enumerate(batch_data):
            try:
                # Process each dataset
                X = pd.DataFrame(dataset)
                predictions = model.predict(X)
                
                batch_results.append({
                    'batch_index': i,
                    'predictions': predictions.tolist(),
                    'n_samples': len(predictions)
                })
            except Exception as e:
                batch_results.append({
                    'batch_index': i,
                    'error': str(e)
                })
        
        response = {
            'session_id': session_id,
            'model_name': model_name,
            'batch_results': batch_results,
            'n_batches': len(batch_data),
            'n_successful': len([r for r in batch_results if 'error' not in r])
        }
        
        return jsonify(clean_nan_inf(response)), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@prediction_bp.route('/download_predictions/<session_id>', methods=['GET'])
def download_predictions(session_id):
    """Download predictions as CSV."""
    try:
        session = session_manager.get_session(session_id)
        if not session or 'last_predictions' not in session:
            return jsonify({'error': 'No predictions available'}), 404
        
        pred_data = session['last_predictions']
        predictions = pred_data['predictions']
        confidence_intervals = pred_data.get('confidence_intervals', {})
        ground_truth = pred_data.get('ground_truth')
        
        # Create DataFrame
        results_df = pd.DataFrame({
            'Prediction': predictions
        })
        
        if 'lower' in confidence_intervals and 'upper' in confidence_intervals:
            results_df['CI_Lower'] = confidence_intervals['lower']
            results_df['CI_Upper'] = confidence_intervals['upper']
        
        if ground_truth is not None:
            results_df['Ground_Truth'] = ground_truth
            results_df['Error'] = results_df['Prediction'] - results_df['Ground_Truth']
            results_df['Relative_Error_%'] = (results_df['Error'] / results_df['Ground_Truth'] * 100)
        
        # Create CSV response
        from io import StringIO
        from flask import make_response
        
        output = StringIO()
        results_df.to_csv(output, index=False)
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = f'attachment; filename=predictions_{session_id[:8]}.csv'
        response.headers['Content-Type'] = 'text/csv'
        
        return response
        
    except Exception as e:
        logger.error(f"Error downloading predictions: {e}")
        return jsonify({'error': str(e)}), 500


def apply_preprocessing(data, preprocessing_config):
    """Apply preprocessing steps to test data."""
    from api.preprocessing import PreprocessingPipeline
    
    pipeline = PreprocessingPipeline()
    processed_data = data.copy()
    
    # Apply outlier removal
    if preprocessing_config.get('remove_outliers', False):
        method = preprocessing_config.get('outlier_method', 'iqr')
        processed_data = pipeline.remove_outliers(processed_data, method, **preprocessing_config)
    
    measurements = processed_data.iloc[:, 1:].values
    
    # Apply baseline correction
    if preprocessing_config.get('baseline_correction', False):
        method = preprocessing_config.get('baseline_method', 'polynomial')
        measurements = pipeline.baseline_correction(measurements, method, **preprocessing_config)
    
    # Apply filtering
    if preprocessing_config.get('apply_filter', False):
        filter_type = preprocessing_config.get('filter_type', 'savgol')
        measurements = pipeline.apply_filter(measurements, filter_type, **preprocessing_config)
    
    # Apply normalization
    if preprocessing_config.get('normalize', False):
        method = preprocessing_config.get('normalization_method', 'minmax')
        measurements, _ = pipeline.normalize_data(measurements, method, **preprocessing_config)
    
    processed_data.iloc[:, 1:] = measurements
    
    return processed_data


def calculate_confidence_intervals(model, X, confidence=0.95, n_bootstrap=100):
    """Calculate confidence intervals using bootstrap."""
    try:
        # Check if model supports prediction intervals
        if hasattr(model, 'predict_proba'):
            # For probabilistic models
            predictions = []
            for _ in range(n_bootstrap):
                # Bootstrap sampling
                indices = np.random.choice(len(X), len(X), replace=True)
                X_bootstrap = X.iloc[indices]
                pred = model.predict(X_bootstrap)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            lower = np.percentile(predictions, (1 - confidence) * 100 / 2, axis=0)
            upper = np.percentile(predictions, (1 + confidence) * 100 / 2, axis=0)
            
            return {
                'lower': lower.tolist(),
                'upper': upper.tolist(),
                'confidence_level': confidence
            }
        else:
            # For deterministic models, return empty
            return {}
    except:
        return {}


@prediction_bp.route('/compare_models', methods=['POST'])
def compare_model_predictions():
    """Compare predictions from multiple models."""
    try:
        data = request.json
        session_id = data.get('session_id')
        model_names = data.get('models', [])
        test_data = data.get('test_data')
        
        session = session_manager.get_session(session_id)
        if not session or 'models' not in session:
            return jsonify({'error': 'No trained models available'}), 400
        
        comparison_results = {}
        
        for model_name in model_names:
            if model_name not in session['models']:
                comparison_results[model_name] = {'error': 'Model not found'}
                continue
            
            model = session['models'][model_name]['model']
            X_test = pd.DataFrame(test_data)
            
            predictions = model.predict(X_test)
            comparison_results[model_name] = {
                'predictions': predictions.tolist(),
                'mean': float(predictions.mean()),
                'std': float(predictions.std()),
                'min': float(predictions.min()),
                'max': float(predictions.max())
            }
        
        response = {
            'session_id': session_id,
            'comparison': comparison_results,
            'n_models': len(model_names),
            'n_samples': len(test_data) if test_data else 0
        }
        
        return jsonify(clean_nan_inf(response)), 200
        
    except Exception as e:
        logger.error(f"Model comparison error: {e}")
        return jsonify({'error': str(e)}), 500
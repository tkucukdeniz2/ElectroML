#!/usr/bin/env python3
"""
ElectroML - Web Application for Electrochemical Data Analysis
Main Flask application
"""

import os
import sys
from pathlib import Path
import json
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import plotly.graph_objs as go
import plotly.utils
import logging
import io
import base64
import math

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from core.feature_extraction import FeatureExtractor
from core.model_training import ModelTrainer, TrainingConfig
from models.model_factory import ModelFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'electroml-secret-key-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

CORS(app)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Global storage for session data (in production, use Redis or database)
session_data = {}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_session_id():
    """Generate unique session ID."""
    return str(uuid.uuid4())


def clean_nan_inf(obj):
    """Replace NaN and Inf values with None for JSON serialization."""
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


@app.route('/')
def index():
    """Render main application page."""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and initial data processing."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload CSV or Excel file.'}), 400
        
        # Generate session ID
        session_id = generate_session_id()
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] / f"{session_id}_{filename}"
        file.save(filepath)
        
        # Load data
        if filename.endswith('.csv'):
            data = pd.read_csv(filepath)
        else:
            data = pd.read_excel(filepath)
        
        # Convert all column names to strings to avoid mixed type issues
        data.columns = [str(col) for col in data.columns]
        
        # Store in session
        session_data[session_id] = {
            'data': data,
            'filename': filename,
            'upload_time': datetime.now().isoformat()
        }
        
        # Generate preview
        preview = {
            'session_id': session_id,
            'filename': filename,
            'shape': data.shape,
            'columns': list(data.columns)[:20],  # First 20 columns (already strings)
            'sample_data': data.head(10).to_dict('records'),  # Column names are already strings
            'concentration_range': {
                'min': float(data.iloc[:, 0].min()),
                'max': float(data.iloc[:, 0].max()),
                'unique': int(data.iloc[:, 0].nunique())
            }
        }
        
        # Create initial plot
        plot_data = create_voltammogram_plot(data)
        preview['plot'] = plot_data
        
        return jsonify(preview), 200
        
    except Exception as e:
        import traceback
        logger.error(f"Upload error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/extract_features', methods=['POST'])
def extract_features():
    """Extract features from uploaded data."""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id or session_id not in session_data:
            return jsonify({'error': 'Invalid session'}), 400
        
        # Get data
        df = session_data[session_id]['data']
        
        # Extract features
        smoothing = data.get('smoothing', True)
        window_length = data.get('window_length', 11)
        polyorder = data.get('polyorder', 3)
        
        # Convert string column names to float (they were converted to strings during upload)
        voltages = np.array([float(col) for col in df.columns[1:]])
        
        extractor = FeatureExtractor(
            smoothing=smoothing,
            window_length=window_length,
            polyorder=polyorder
        )
        
        features_df = extractor.extract_features(df, voltages)
        
        # Store features
        session_data[session_id]['features'] = features_df
        
        # Calculate feature importance
        from sklearn.ensemble import RandomForestRegressor
        X = features_df.drop(columns=['Concentration'])
        y = features_df['Concentration']
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Create feature importance plot
        importance_plot = create_feature_importance_plot(importance)
        
        response = {
            'features': features_df.describe().to_dict(),
            'feature_names': list(X.columns),
            'n_features': len(X.columns),
            'n_samples': len(features_df),
            'importance_plot': importance_plot,
            'top_features': importance.head(10).to_dict('records')
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def train_models():
    """Train machine learning models."""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id or session_id not in session_data:
            return jsonify({'error': 'Invalid session'}), 400
        
        if 'features' not in session_data[session_id]:
            return jsonify({'error': 'Please extract features first'}), 400
        
        # Get features
        features_df = session_data[session_id]['features']
        X = features_df.drop(columns=['Concentration'])
        y = features_df['Concentration']
        
        # Get training parameters
        models_to_train = data.get('models', ['linear_regression', 'random_forest'])
        cv_strategy = data.get('cv_strategy', 'loo')
        optimize = data.get('optimize_hyperparameters', False)
        
        # Configure training
        n_samples = len(X)
        if n_samples < 5:
            cv_strategy = 'loo'  # Force LOO for small datasets
        
        config = TrainingConfig(
            cv_strategy=cv_strategy,
            n_splits=min(5, n_samples),
            optimize_hyperparams=optimize,
            n_trials=50 if optimize else 0
        )
        
        trainer = ModelTrainer(config)
        results = {}
        
        # Train each model
        for model_name in models_to_train:
            try:
                logger.info(f"Training {model_name}...")
                
                # Create model
                model_class = lambda **kwargs: ModelFactory.create_model(model_name, kwargs)
                param_space = ModelFactory.get_param_space(model_name) if optimize else None
                
                # Train
                result = trainer.train_model(model_class, X, y, model_name, param_space)
                
                # Store result with NaN/Inf handling
                results[model_name] = {
                    'metrics': clean_nan_inf(result.metrics),
                    'cv_scores': {
                        'mean': float(np.mean(result.cv_scores['test_r2'])) if result.cv_scores else 0.0,
                        'std': float(np.std(result.cv_scores['test_r2'])) if result.cv_scores else 0.0
                    }
                }
                # Clean the cv_scores too
                results[model_name]['cv_scores'] = clean_nan_inf(results[model_name]['cv_scores'])
                
                # Store model for predictions
                if 'models' not in session_data[session_id]:
                    session_data[session_id]['models'] = {}
                session_data[session_id]['models'][model_name] = result
                
            except Exception as e:
                import traceback
                logger.error(f"Failed to train {model_name}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                results[model_name] = {'error': str(e)}
        
        # Create comparison plot
        comparison_plot = create_model_comparison_plot(results)
        
        # Create prediction plots for best model (filter out models with errors)
        valid_models = {k: v for k, v in results.items() if 'error' not in v and 'metrics' in v}
        
        if not valid_models:
            return jsonify({'error': 'All models failed to train'}), 500
        
        best_model_name = max(valid_models.keys(), 
                            key=lambda k: valid_models[k].get('metrics', {}).get('r2', -float('inf')))
        best_model_result = session_data[session_id]['models'][best_model_name]
        
        prediction_plot = create_prediction_plot(
            best_model_result.actual_values,
            best_model_result.predictions
        )
        
        response = {
            'results': results,
            'comparison_plot': comparison_plot,
            'prediction_plot': prediction_plot,
            'best_model': best_model_name
        }
        
        # Clean any remaining NaN/Inf values
        response = clean_nan_inf(response)
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions on new data."""
    try:
        session_id = request.form.get('session_id')
        model_name = request.form.get('model_name')
        
        if not session_id or session_id not in session_data:
            return jsonify({'error': 'Invalid session'}), 400
        
        if 'models' not in session_data[session_id]:
            return jsonify({'error': 'No trained models available'}), 400
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Load new data
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] / f"pred_{session_id}_{filename}"
        file.save(filepath)
        
        if filename.endswith('.csv'):
            new_data = pd.read_csv(filepath)
        else:
            new_data = pd.read_excel(filepath)
        
        # Extract features
        voltages = np.array(new_data.columns[1:].astype(float))
        extractor = FeatureExtractor()
        features_df = extractor.extract_features(new_data, voltages)
        
        # Get model
        if model_name not in session_data[session_id]['models']:
            return jsonify({'error': f'Model {model_name} not found'}), 400
        
        model_result = session_data[session_id]['models'][model_name]
        
        # Prepare features
        X = features_df.drop(columns=['Concentration'])
        X_scaled = model_result.scaler_X.transform(X)
        
        # Predict
        predictions_scaled = model_result.model.predict(X_scaled)
        predictions = model_result.scaler_y.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).ravel()
        
        # Create results
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                'sample': i + 1,
                'prediction': float(pred),
                'confidence_lower': float(pred * 0.95),  # Placeholder
                'confidence_upper': float(pred * 1.05)   # Placeholder
            })
        
        # Create visualization
        pred_plot = create_prediction_results_plot(results)
        
        response = {
            'predictions': results,
            'plot': pred_plot,
            'model_used': model_name,
            'n_samples': len(predictions)
        }
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/<session_id>')
def export_results(session_id):
    """Export analysis results."""
    try:
        if session_id not in session_data:
            return jsonify({'error': 'Invalid session'}), 400
        
        # Create Excel file with results
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write features if available
            if 'features' in session_data[session_id]:
                session_data[session_id]['features'].to_excel(
                    writer, sheet_name='Features', index=False
                )
            
            # Write model results if available
            if 'models' in session_data[session_id]:
                results_data = []
                for model_name, result in session_data[session_id]['models'].items():
                    results_data.append({
                        'Model': model_name,
                        'R2': result.metrics['r2'],
                        'RMSE': result.metrics['rmse'],
                        'MAE': result.metrics['mae']
                    })
                
                pd.DataFrame(results_data).to_excel(
                    writer, sheet_name='Model Results', index=False
                )
        
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'electroml_results_{session_id[:8]}.xlsx'
        )
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({'error': str(e)}), 500


# Plotting functions
def create_voltammogram_plot(data):
    """Create voltammogram plot."""
    # Convert string column names to float (they were converted to strings during upload)
    voltages = np.array([float(col) for col in data.columns[1:]])
    
    traces = []
    n_samples = min(10, len(data))  # Plot up to 10 samples
    
    for i in range(n_samples):
        current = data.iloc[i, 1:].values
        concentration = data.iloc[i, 0]
        
        traces.append(go.Scatter(
            x=list(voltages),
            y=list(current),
            mode='lines',
            name=f'C={concentration:.2f} μM',
            line=dict(width=2)
        ))
    
    layout = go.Layout(
        title='Voltammogram',
        xaxis=dict(title='Voltage (V)'),
        yaxis=dict(title='Current (A)'),
        hovermode='closest',
        showlegend=True
    )
    
    fig = go.Figure(data=traces, layout=layout)
    return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))


def create_feature_importance_plot(importance_df):
    """Create feature importance bar plot."""
    top_features = importance_df.head(15)
    
    trace = go.Bar(
        x=list(top_features['importance']),
        y=list(top_features['feature']),
        orientation='h',
        marker=dict(color='rgb(33, 150, 243)')
    )
    
    layout = go.Layout(
        title='Feature Importance',
        xaxis=dict(title='Importance'),
        yaxis=dict(title='Feature'),
        margin=dict(l=150)
    )
    
    fig = go.Figure(data=[trace], layout=layout)
    return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))


def create_model_comparison_plot(results):
    """Create model comparison bar plot."""
    models = []
    r2_scores = []
    
    for model_name, result in results.items():
        if 'metrics' in result:
            models.append(model_name)
            r2_scores.append(result['metrics']['r2'])
    
    trace = go.Bar(
        x=models,
        y=r2_scores,
        marker=dict(color='rgb(76, 175, 80)')
    )
    
    layout = go.Layout(
        title='Model Comparison',
        xaxis=dict(title='Model'),
        yaxis=dict(title='R² Score'),
        showlegend=False
    )
    
    fig = go.Figure(data=[trace], layout=layout)
    return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))


def create_prediction_plot(actual, predicted):
    """Create actual vs predicted scatter plot."""
    trace = go.Scatter(
        x=list(actual),
        y=list(predicted),
        mode='markers',
        marker=dict(size=10, color='rgb(255, 152, 0)'),
        name='Predictions'
    )
    
    # Add ideal line
    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))
    
    ideal_trace = go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(dash='dash', color='red'),
        name='Ideal'
    )
    
    layout = go.Layout(
        title='Actual vs Predicted',
        xaxis=dict(title='Actual Values'),
        yaxis=dict(title='Predicted Values'),
        hovermode='closest'
    )
    
    fig = go.Figure(data=[trace, ideal_trace], layout=layout)
    return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))


def create_prediction_results_plot(results):
    """Create prediction results bar plot."""
    samples = [r['sample'] for r in results]
    predictions = [r['prediction'] for r in results]
    
    trace = go.Bar(
        x=samples,
        y=predictions,
        marker=dict(color='rgb(156, 39, 176)')
    )
    
    layout = go.Layout(
        title='Predicted Concentrations',
        xaxis=dict(title='Sample'),
        yaxis=dict(title='Concentration (μM)'),
        showlegend=False
    )
    
    fig = go.Figure(data=[trace], layout=layout)
    return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))


if __name__ == '__main__':
    # Run the application
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
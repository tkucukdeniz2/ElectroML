"""
Data upload and management API endpoints.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from flask import request, jsonify, current_app
from werkzeug.utils import secure_filename
import logging

from api import data_bp
from utils.validators import allowed_file, validate_file_size, validate_data_format
from utils.session_manager import session_manager
from utils.json_helpers import clean_nan_inf

logger = logging.getLogger(__name__)


@data_bp.route('/upload', methods=['POST'])
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
        
        # Create new session
        session_id = session_manager.create_session()
        
        # Save file
        filename = secure_filename(file.filename)
        upload_folder = current_app.config.get('UPLOAD_FOLDER', Path(__file__).parent.parent / 'uploads')
        upload_folder.mkdir(exist_ok=True)
        filepath = upload_folder / f"{session_id}_{filename}"
        file.save(filepath)
        
        # Validate file size
        is_valid, error_msg = validate_file_size(str(filepath))
        if not is_valid:
            os.remove(filepath)
            return jsonify({'error': error_msg}), 400
        
        # Load data
        if filename.endswith('.csv'):
            data = pd.read_csv(filepath)
        else:
            data = pd.read_excel(filepath)
        
        # Convert all column names to strings to avoid mixed type issues
        data.columns = [str(col) for col in data.columns]
        
        # Validate data format
        is_valid, error_msg = validate_data_format(data)
        if not is_valid:
            os.remove(filepath)
            session_manager.delete_session(session_id)
            return jsonify({'error': error_msg}), 400
        
        # Store data in session
        session_manager.update_session(session_id, 'data', data)
        session_manager.update_session(session_id, 'filename', filename)
        session_manager.update_session(session_id, 'filepath', str(filepath))
        
        # Generate preview
        preview = {
            'session_id': session_id,
            'filename': filename,
            'shape': data.shape,
            'columns': list(data.columns)[:20],
            'sample_data': data.head(10).to_dict('records'),
            'concentration_range': {
                'min': float(data.iloc[:, 0].min()),
                'max': float(data.iloc[:, 0].max()),
                'unique': int(data.iloc[:, 0].nunique())
            },
            'data_statistics': {
                'mean': float(data.iloc[:, 1:].mean().mean()),
                'std': float(data.iloc[:, 1:].std().mean()),
                'median': float(data.iloc[:, 1:].median().median())
            }
        }
        
        # Clean any NaN/Inf values
        preview = clean_nan_inf(preview)
        
        logger.info(f"File uploaded successfully: {filename} (session: {session_id})")
        return jsonify(preview), 200
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500


@data_bp.route('/info/<session_id>', methods=['GET'])
def get_data_info(session_id):
    """Get information about uploaded data."""
    try:
        session = session_manager.get_session(session_id)
        if not session or session['data'] is None:
            return jsonify({'error': 'Session not found or no data uploaded'}), 404
        
        data = session['data']
        
        info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'memory_usage': float(data.memory_usage(deep=True).sum() / 1024),  # KB
            'has_missing': bool(data.isnull().any().any()),
            'concentration_stats': {
                'min': float(data.iloc[:, 0].min()),
                'max': float(data.iloc[:, 0].max()),
                'mean': float(data.iloc[:, 0].mean()),
                'std': float(data.iloc[:, 0].std()),
                'unique': int(data.iloc[:, 0].nunique())
            }
        }
        
        return jsonify(clean_nan_inf(info)), 200
        
    except Exception as e:
        logger.error(f"Error getting data info: {e}")
        return jsonify({'error': str(e)}), 500


@data_bp.route('/download/<session_id>', methods=['GET'])
def download_processed_data(session_id):
    """Download processed data."""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Get the most recent data (either preprocessed or original)
        data = session.get('preprocessed_data', session.get('data'))
        if data is None:
            return jsonify({'error': 'No data available'}), 404
        
        # Create CSV response
        from io import StringIO
        from flask import make_response
        
        output = StringIO()
        data.to_csv(output, index=False)
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = f'attachment; filename=processed_data_{session_id[:8]}.csv'
        response.headers['Content-Type'] = 'text/csv'
        
        return response
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return jsonify({'error': str(e)}), 500


@data_bp.route('/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions."""
    try:
        sessions = session_manager.get_all_sessions()
        session_list = []
        
        for sid, session_data in sessions.items():
            session_info = {
                'session_id': sid,
                'created_at': session_data['created_at'],
                'has_data': session_data['data'] is not None,
                'has_features': session_data.get('features') is not None,
                'n_models': len(session_data.get('models', {})),
                'filename': session_data.get('filename', 'N/A')
            }
            session_list.append(session_info)
        
        return jsonify({'sessions': session_list}), 200
        
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return jsonify({'error': str(e)}), 500


@data_bp.route('/cleanup', methods=['POST'])
def cleanup_old_sessions():
    """Clean up old sessions."""
    try:
        max_age = request.json.get('max_age_hours', 24)
        session_manager.cleanup_old_sessions(max_age)
        return jsonify({'message': 'Cleanup completed'}), 200
        
    except Exception as e:
        logger.error(f"Error cleaning up sessions: {e}")
        return jsonify({'error': str(e)}), 500
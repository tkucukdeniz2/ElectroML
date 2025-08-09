"""
Data preprocessing API endpoints with comprehensive signal processing.
"""

import numpy as np
import pandas as pd
from flask import request, jsonify
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import median_filter
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
import logging

from api import preprocessing_bp
from utils.session_manager import session_manager
from utils.validators import validate_preprocessing_params
from utils.json_helpers import clean_nan_inf

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Comprehensive preprocessing pipeline for electrochemical data."""
    
    @staticmethod
    def remove_outliers(data: pd.DataFrame, method: str = 'iqr', **kwargs) -> pd.DataFrame:
        """
        Remove outliers from data.
        
        Args:
            data: Input DataFrame
            method: 'iqr', 'zscore', 'isolation_forest'
            **kwargs: Method-specific parameters
        """
        data_clean = data.copy()
        
        if method == 'iqr':
            Q1 = data_clean.iloc[:, 1:].quantile(0.25)
            Q3 = data_clean.iloc[:, 1:].quantile(0.75)
            IQR = Q3 - Q1
            factor = kwargs.get('iqr_factor', 1.5)
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            mask = ((data_clean.iloc[:, 1:] >= lower_bound) & 
                   (data_clean.iloc[:, 1:] <= upper_bound)).all(axis=1)
            data_clean = data_clean[mask]
            
        elif method == 'zscore':
            threshold = kwargs.get('zscore_threshold', 3)
            z_scores = np.abs((data_clean.iloc[:, 1:] - data_clean.iloc[:, 1:].mean()) / 
                            data_clean.iloc[:, 1:].std())
            mask = (z_scores < threshold).all(axis=1)
            data_clean = data_clean[mask]
            
        elif method == 'isolation_forest':
            contamination = kwargs.get('contamination', 0.1)
            clf = IsolationForest(contamination=contamination, random_state=42)
            outliers = clf.fit_predict(data_clean.iloc[:, 1:])
            data_clean = data_clean[outliers == 1]
        
        return data_clean
    
    @staticmethod
    def baseline_correction(data: np.ndarray, method: str = 'polynomial', **kwargs) -> np.ndarray:
        """
        Apply baseline correction to signals.
        
        Args:
            data: Signal data (each row is a signal)
            method: 'polynomial', 'als', 'moving_average'
            **kwargs: Method-specific parameters
        """
        corrected = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            signal_data = data[i, :]
            
            if method == 'polynomial':
                poly_order = kwargs.get('poly_order', 3)
                x = np.arange(len(signal_data))
                coeffs = np.polyfit(x, signal_data, poly_order)
                baseline = np.polyval(coeffs, x)
                corrected[i, :] = signal_data - baseline
                
            elif method == 'als':
                # Asymmetric Least Squares
                lam = kwargs.get('lambda', 1e6)
                p = kwargs.get('p', 0.01)
                baseline = PreprocessingPipeline._als_baseline(signal_data, lam, p)
                corrected[i, :] = signal_data - baseline
                
            elif method == 'moving_average':
                window = kwargs.get('window_size', 50)
                baseline = pd.Series(signal_data).rolling(window, center=True, min_periods=1).mean().values
                corrected[i, :] = signal_data - baseline
            
            else:
                corrected[i, :] = signal_data
        
        return corrected
    
    @staticmethod
    def _als_baseline(y, lam, p, niter=10):
        """Asymmetric Least Squares baseline."""
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        D = lam * D.dot(D.transpose())
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        
        for i in range(niter):
            W.setdiag(w)
            Z = W + D
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        
        return z
    
    @staticmethod
    def apply_filter(data: np.ndarray, filter_type: str = 'savgol', **kwargs) -> np.ndarray:
        """
        Apply signal filtering.
        
        Args:
            data: Signal data
            filter_type: 'butterworth', 'chebyshev', 'median', 'savgol'
            **kwargs: Filter-specific parameters
        """
        filtered = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            signal_data = data[i, :]
            
            if filter_type == 'butterworth':
                order = kwargs.get('order', 4)
                cutoff = kwargs.get('cutoff_freq', 0.1)
                b, a = signal.butter(order, cutoff)
                filtered[i, :] = signal.filtfilt(b, a, signal_data)
                
            elif filter_type == 'chebyshev':
                order = kwargs.get('order', 4)
                ripple = kwargs.get('ripple', 0.1)
                cutoff = kwargs.get('cutoff_freq', 0.1)
                b, a = signal.cheby1(order, ripple, cutoff)
                filtered[i, :] = signal.filtfilt(b, a, signal_data)
                
            elif filter_type == 'median':
                kernel_size = kwargs.get('kernel_size', 5)
                filtered[i, :] = median_filter(signal_data, size=kernel_size)
                
            elif filter_type == 'savgol':
                window_length = kwargs.get('window_length', 11)
                polyorder = kwargs.get('polyorder', 3)
                filtered[i, :] = signal.savgol_filter(signal_data, window_length, polyorder)
            
            else:
                filtered[i, :] = signal_data
        
        return filtered
    
    @staticmethod
    def normalize_data(data: np.ndarray, method: str = 'minmax', **kwargs) -> tuple:
        """
        Normalize data.
        
        Args:
            data: Data to normalize
            method: 'minmax', 'zscore', 'robust'
            
        Returns:
            (normalized_data, scaler)
        """
        if method == 'minmax':
            scaler = MinMaxScaler(feature_range=kwargs.get('feature_range', (0, 1)))
        elif method == 'zscore':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            return data, None
        
        normalized = scaler.fit_transform(data.T).T
        return normalized, scaler


@preprocessing_bp.route('/process', methods=['POST'])
def preprocess_data():
    """Apply comprehensive preprocessing to uploaded data."""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        # Validate session
        session = session_manager.get_session(session_id)
        if not session or session['data'] is None:
            return jsonify({'error': 'Invalid session or no data uploaded'}), 400
        
        # Validate preprocessing parameters
        is_valid, error_msg = validate_preprocessing_params(data)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Get original data
        df = session['data'].copy()
        original_shape = df.shape
        
        # Store preprocessing configuration
        session_manager.update_session(session_id, 'preprocessing_config', data)
        
        pipeline = PreprocessingPipeline()
        steps_applied = []
        
        # Step 1: Outlier removal
        if data.get('remove_outliers', False):
            method = data.get('outlier_method', 'iqr')
            # Pass outlier-specific parameters without duplicating method
            outlier_params = {k: v for k, v in data.items() 
                            if k not in ['outlier_method', 'session_id', 'remove_outliers']}
            df = pipeline.remove_outliers(df, method, **outlier_params)
            steps_applied.append(f"Outlier removal ({method})")
            logger.info(f"Removed {original_shape[0] - df.shape[0]} outliers")
        
        # Extract measurement data for signal processing
        measurements = df.iloc[:, 1:].values
        
        # Step 2: Baseline correction
        if data.get('baseline_correction', False):
            method = data.get('baseline_method', 'polynomial')
            # Pass baseline-specific parameters without duplicating method
            baseline_params = {k: v for k, v in data.items() 
                             if k not in ['baseline_method', 'session_id', 'baseline_correction']}
            measurements = pipeline.baseline_correction(measurements, method, **baseline_params)
            steps_applied.append(f"Baseline correction ({method})")
        
        # Step 3: Signal filtering
        if data.get('apply_filter', False):
            filter_type = data.get('filter_type', 'savgol')
            # Pass filter-specific parameters without duplicating filter_type
            filter_params = {k: v for k, v in data.items() 
                           if k not in ['filter_type', 'session_id', 'apply_filter']}
            measurements = pipeline.apply_filter(measurements, filter_type, **filter_params)
            steps_applied.append(f"Signal filtering ({filter_type})")
        
        # Step 4: Normalization
        if data.get('normalize', False):
            method = data.get('normalization_method', 'minmax')
            # Pass normalization-specific parameters without duplicating method
            norm_params = {k: v for k, v in data.items() 
                         if k not in ['normalization_method', 'session_id', 'normalize']}
            measurements, scaler = pipeline.normalize_data(measurements, method, **norm_params)
            steps_applied.append(f"Normalization ({method})")
            session_manager.update_session(session_id, 'normalization_scaler', scaler)
        
        # Step 5: Derivative computation
        if data.get('compute_derivative', False):
            derivative_order = data.get('derivative_order', 1)
            derivatives = np.gradient(measurements, axis=1)
            if derivative_order == 2:
                derivatives = np.gradient(derivatives, axis=1)
            
            # Add derivatives as additional features
            derivative_cols = [f"d{derivative_order}_col_{i}" for i in range(derivatives.shape[1])]
            derivative_df = pd.DataFrame(derivatives, columns=derivative_cols)
            steps_applied.append(f"{derivative_order}st order derivative")
        
        # Update DataFrame with processed measurements
        df.iloc[:, 1:] = measurements
        
        # Store preprocessed data
        session_manager.update_session(session_id, 'preprocessed_data', df)
        
        # Add educational context for chemists
        educational_context = {
            'outlier_removal': {
                'iqr': 'Interquartile Range - removes data outside Q1-1.5×IQR to Q3+1.5×IQR, like removing spurious peaks in chromatography',
                'zscore': 'Z-score method - removes data >3 standard deviations from mean, useful for normally distributed noise',
                'isolation_forest': 'ML-based outlier detection - identifies anomalous voltammograms using pattern recognition'
            },
            'baseline_correction': {
                'polynomial': 'Fits polynomial to baseline drift - similar to baseline correction in spectroscopy',
                'als': 'Asymmetric Least Squares - handles curved baselines common in CV data',
                'moving_average': 'Rolling average subtraction - simple but effective for linear drift'
            },
            'filtering': {
                'savgol': 'Savitzky-Golay filter - preserves peak shape while reducing noise, standard in analytical chemistry',
                'median': 'Median filter - removes spike noise without affecting peak shape',
                'butterworth': 'Frequency-based filtering - removes high-frequency noise'
            },
            'normalization': {
                'minmax': 'Scales to [0,1] range - essential for SVM and neural networks',
                'standard': 'Z-score normalization - centers data with unit variance',
                'robust': 'Median-based scaling - resistant to outliers'
            }
        }
        
        # Add relevant educational notes based on steps applied
        educational_notes = []
        for step in steps_applied:
            for category, methods in educational_context.items():
                for method, description in methods.items():
                    if method in step.lower():
                        educational_notes.append(description)
                        break
        
        # Generate response
        response = {
            'session_id': session_id,
            'original_shape': original_shape,
            'processed_shape': df.shape,
            'steps_applied': steps_applied,
            'statistics': {
                'mean': float(measurements.mean()),
                'std': float(measurements.std()),
                'min': float(measurements.min()),
                'max': float(measurements.max())
            },
            'preview': df.head(5).to_dict('records'),
            'educational_notes': educational_notes,
            'preprocessing_tip': 'Preprocessing improves ML accuracy by 20-40%. Always check signal quality after each step!'
        }
        
        return jsonify(clean_nan_inf(response)), 200
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return jsonify({'error': str(e)}), 500


@preprocessing_bp.route('/peak_detection', methods=['POST'])
def detect_peaks():
    """Detect peaks in voltammetric signals."""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Invalid session'}), 400
        
        # Get data (preprocessed if available, otherwise original)
        df = session.get('preprocessed_data', session.get('data'))
        if df is None:
            return jsonify({'error': 'No data available'}), 400
        
        measurements = df.iloc[:, 1:].values
        
        # Peak detection parameters
        height = data.get('min_height', None)
        distance = data.get('min_distance', 10)
        prominence = data.get('min_prominence', None)
        
        all_peaks = []
        for i in range(measurements.shape[0]):
            peaks, properties = signal.find_peaks(
                measurements[i, :],
                height=height,
                distance=distance,
                prominence=prominence
            )
            
            peak_info = {
                'sample_index': i,
                'concentration': float(df.iloc[i, 0]),
                'n_peaks': len(peaks),
                'peak_positions': peaks.tolist(),
                'peak_heights': properties.get('peak_heights', []).tolist() if 'peak_heights' in properties else []
            }
            all_peaks.append(peak_info)
        
        response = {
            'session_id': session_id,
            'peaks': all_peaks,
            'total_peaks': sum(p['n_peaks'] for p in all_peaks),
            'average_peaks_per_sample': np.mean([p['n_peaks'] for p in all_peaks])
        }
        
        return jsonify(clean_nan_inf(response)), 200
        
    except Exception as e:
        logger.error(f"Peak detection error: {e}")
        return jsonify({'error': str(e)}), 500


@preprocessing_bp.route('/voltage_range', methods=['POST'])
def select_voltage_range():
    """Select specific voltage range for analysis."""
    try:
        data = request.json
        session_id = data.get('session_id')
        min_voltage = data.get('min_voltage')
        max_voltage = data.get('max_voltage')
        
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Invalid session'}), 400
        
        df = session.get('preprocessed_data', session.get('data'))
        if df is None:
            return jsonify({'error': 'No data available'}), 400
        
        # Get voltage values from column names
        voltages = np.array([float(col) for col in df.columns[1:]])
        
        # Find indices within range
        mask = (voltages >= min_voltage) & (voltages <= max_voltage)
        selected_columns = [df.columns[0]] + [df.columns[i+1] for i in range(len(mask)) if mask[i]]
        
        # Create new DataFrame with selected range
        df_filtered = df[selected_columns].copy()
        
        # Update session
        session_manager.update_session(session_id, 'preprocessed_data', df_filtered)
        
        response = {
            'session_id': session_id,
            'original_voltage_range': [float(voltages.min()), float(voltages.max())],
            'selected_voltage_range': [min_voltage, max_voltage],
            'original_n_points': len(voltages),
            'selected_n_points': sum(mask),
            'new_shape': df_filtered.shape
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Voltage range selection error: {e}")
        return jsonify({'error': str(e)}), 500
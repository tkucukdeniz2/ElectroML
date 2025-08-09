"""
Feature extraction module for electrochemical data analysis.
Implements sophisticated feature engineering for voltammetric signals.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.stats import skew, kurtosis, entropy
from scipy.integrate import simpson
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Advanced feature extraction for voltammetric data."""
    
    def __init__(self, smoothing: bool = True, window_length: int = 11, polyorder: int = 3):
        """
        Initialize feature extractor.
        
        Args:
            smoothing: Apply Savitzky-Golay filter for noise reduction
            window_length: Window length for smoothing filter
            polyorder: Polynomial order for smoothing filter
        """
        self.smoothing = smoothing
        self.window_length = window_length
        self.polyorder = polyorder
        self.feature_names = []
        
    def extract_features(self, data: pd.DataFrame, voltages: np.ndarray) -> pd.DataFrame:
        """
        Extract comprehensive features from voltammetric data.
        
        Args:
            data: DataFrame with concentration in first column, currents in rest
            voltages: Array of voltage values
            
        Returns:
            DataFrame with extracted features
        """
        features_list = []
        current_data = data.iloc[:, 1:]
        
        for idx in range(len(data)):
            current_values = current_data.iloc[idx].values
            
            if self.smoothing and len(current_values) > self.window_length:
                current_values = savgol_filter(current_values, self.window_length, self.polyorder)
            
            features = self._extract_single_sample_features(current_values, voltages)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        features_df['Concentration'] = data.iloc[:, 0].values
        
        self.feature_names = [col for col in features_df.columns if col != 'Concentration']
        logger.info(f"Extracted {len(self.feature_names)} features from {len(data)} samples")
        
        return features_df
    
    def _extract_single_sample_features(self, current: np.ndarray, voltages: np.ndarray) -> Dict:
        """Extract features from a single voltammetric scan."""
        features = {}
        
        # Basic statistical features
        features.update(self._extract_statistical_features(current))
        
        # Peak-related features
        features.update(self._extract_peak_features(current, voltages))
        
        # Derivative features
        features.update(self._extract_derivative_features(current, voltages))
        
        # Integral features
        features.update(self._extract_integral_features(current, voltages))
        
        # Symmetry and shape features
        features.update(self._extract_shape_features(current))
        
        # Frequency domain features (optional, for AC voltammetry)
        features.update(self._extract_frequency_features(current))
        
        return features
    
    def _extract_statistical_features(self, current: np.ndarray) -> Dict:
        """Extract statistical features."""
        return {
            'mean_current': np.mean(current),
            'median_current': np.median(current),
            'std_current': np.std(current),
            'var_current': np.var(current),
            'max_current': np.max(current),
            'min_current': np.min(current),
            'range_current': np.ptp(current),
            'skewness': skew(current),
            'kurtosis': kurtosis(current),
            'entropy': entropy(np.abs(current) + 1e-10),
            'q25': np.percentile(current, 25),
            'q75': np.percentile(current, 75),
            'iqr': np.percentile(current, 75) - np.percentile(current, 25)
        }
    
    def _extract_peak_features(self, current: np.ndarray, voltages: np.ndarray) -> Dict:
        """Extract peak-related features."""
        features = {}
        
        # Find peaks
        peaks, properties = find_peaks(current, prominence=0.01, distance=5)
        
        features['num_peaks'] = len(peaks)
        
        if len(peaks) > 0:
            features['max_peak_height'] = np.max(current[peaks])
            features['mean_peak_height'] = np.mean(current[peaks])
            features['peak_voltage_span'] = voltages[peaks[-1]] - voltages[peaks[0]] if len(peaks) > 1 else 0
            
            # Peak widths
            widths, width_heights, left_ips, right_ips = peak_widths(current, peaks, rel_height=0.5)
            features['mean_peak_width'] = np.mean(widths) if len(widths) > 0 else 0
            features['max_peak_width'] = np.max(widths) if len(widths) > 0 else 0
            
            # Peak prominences
            if 'peak_prominences' in properties:
                features['mean_prominence'] = np.mean(properties['peak_prominences'])
                features['max_prominence'] = np.max(properties['peak_prominences'])
        else:
            features.update({
                'max_peak_height': 0,
                'mean_peak_height': 0,
                'peak_voltage_span': 0,
                'mean_peak_width': 0,
                'max_peak_width': 0,
                'mean_prominence': 0,
                'max_prominence': 0
            })
        
        # Find valleys (negative peaks)
        valleys, _ = find_peaks(-current, prominence=0.01, distance=5)
        features['num_valleys'] = len(valleys)
        
        return features
    
    def _extract_derivative_features(self, current: np.ndarray, voltages: np.ndarray) -> Dict:
        """Extract derivative-based features."""
        # First derivative (di/dV)
        first_deriv = np.gradient(current, voltages)
        
        # Second derivative (d²i/dV²)
        second_deriv = np.gradient(first_deriv, voltages)
        
        return {
            'mean_first_deriv': np.mean(first_deriv),
            'std_first_deriv': np.std(first_deriv),
            'max_first_deriv': np.max(first_deriv),
            'min_first_deriv': np.min(first_deriv),
            'mean_second_deriv': np.mean(second_deriv),
            'std_second_deriv': np.std(second_deriv),
            'max_second_deriv': np.max(second_deriv),
            'min_second_deriv': np.min(second_deriv),
            'rising_rate': np.max(first_deriv[first_deriv > 0]) if np.any(first_deriv > 0) else 0,
            'falling_rate': np.min(first_deriv[first_deriv < 0]) if np.any(first_deriv < 0) else 0
        }
    
    def _extract_integral_features(self, current: np.ndarray, voltages: np.ndarray) -> Dict:
        """Extract integral-based features."""
        # Total charge (area under curve)
        total_charge = simpson(current, voltages)
        
        # Positive and negative charges
        positive_current = np.where(current > 0, current, 0)
        negative_current = np.where(current < 0, current, 0)
        
        positive_charge = simpson(positive_current, voltages)
        negative_charge = simpson(negative_current, voltages)
        
        return {
            'total_charge': total_charge,
            'positive_charge': positive_charge,
            'negative_charge': abs(negative_charge),
            'charge_ratio': positive_charge / (abs(negative_charge) + 1e-10),
            'net_charge': positive_charge + negative_charge
        }
    
    def _extract_shape_features(self, current: np.ndarray) -> Dict:
        """Extract shape and symmetry features."""
        midpoint = len(current) // 2
        first_half = current[:midpoint]
        second_half = current[midpoint:]
        
        # Reverse second half for symmetry comparison
        if len(second_half) > len(first_half):
            second_half = second_half[:len(first_half)]
        elif len(first_half) > len(second_half):
            first_half = first_half[:len(second_half)]
        
        second_half_reversed = second_half[::-1]
        
        # Symmetry measures
        symmetry = 1 - np.mean(np.abs(first_half - second_half_reversed)) / (np.mean(np.abs(current)) + 1e-10)
        
        # Shape factors
        rms_current = np.sqrt(np.mean(current**2))
        form_factor = rms_current / (np.mean(np.abs(current)) + 1e-10)
        crest_factor = np.max(np.abs(current)) / (rms_current + 1e-10)
        
        return {
            'symmetry': symmetry,
            'form_factor': form_factor,
            'crest_factor': crest_factor,
            'initial_current': current[0],
            'final_current': current[-1],
            'midpoint_current': current[midpoint] if midpoint < len(current) else current[-1]
        }
    
    def _extract_frequency_features(self, current: np.ndarray) -> Dict:
        """Extract frequency domain features using FFT."""
        # Apply FFT
        fft_values = np.fft.fft(current)
        fft_magnitude = np.abs(fft_values)
        
        # Get dominant frequencies
        n = len(current)
        dominant_freq_idx = np.argmax(fft_magnitude[1:n//2]) + 1
        
        return {
            'dominant_frequency': dominant_freq_idx,
            'mean_fft_magnitude': np.mean(fft_magnitude[1:n//2]),
            'max_fft_magnitude': np.max(fft_magnitude[1:n//2]),
            'spectral_energy': np.sum(fft_magnitude[1:n//2]**2)
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of extracted feature names."""
        return self.feature_names
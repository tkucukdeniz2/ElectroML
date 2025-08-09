# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ElectroML is an open-source machine learning platform for analyzing electrochemical sensor data (Cyclic Voltammetry and Differential Pulse Voltammetry). It predicts analyte concentrations in chemical samples for environmental monitoring, medical diagnostics, and industrial quality control.

## Development Commands

### Running the Application
```bash
# Start the Streamlit web interface
streamlit run src/app.py

# The application will be available at http://localhost:8501
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Testing with Sample Data
The repository includes `src/sensor_data.xlsx` as sample data. The expected format:
- Column 1: "Con" (Analyte Concentration in μM)
- Columns 2+: Current measurements at voltages from -1V to +1V (0.005V increments)

## Architecture Overview

### Core Application Structure

The entire application is contained in `src/app.py` (706 lines) with a modular architecture:

1. **Feature Extraction Engine** (`extract_features`, lines 86-140)
   - Extracts 16 features from voltammetric signals
   - Uses scipy for signal processing and peak detection
   - Calculates statistical (mean, median, std), geometric (peaks, slopes), and advanced features (skewness, kurtosis)

2. **Model Training Pipeline** (lines 143-183)
   - Six ML models: LinearRegression, SVM, RandomForest, XGBoost, LightGBM, ANN
   - Each model has dedicated training function with optimized hyperparameters
   - ANN uses 3-layer architecture (64-32-1 neurons) with Adam optimizer

3. **Data Normalization** (`normalize_data`, lines 76-84)
   - MinMaxScaler for both features (X) and targets (y)
   - Maintains separate scalers for inverse transformation during prediction

4. **Model Evaluation** (`evaluate_model`, lines 185-203)
   - Handles different model types (Keras, XGBoost, sklearn)
   - Computes MSE, MAE, RMSE, R² metrics
   - Performs inverse scaling for interpretable results

### Streamlit Interface Pages

The application uses session state to maintain data flow between pages:

1. **Data Preprocessing** (lines 206-292)
   - File upload → Feature extraction → Feature importance analysis
   - Stores `processed_features` in session state
   - Uses RandomForest for feature importance ranking

2. **Model Training** (lines 294-429)
   - Reads from `processed_features` session state
   - Implements Leave-One-Out Cross-Validation
   - Parallel progress tracking for multiple models
   - Stores `training_results` (models + metrics) in session state

3. **Results Visualization** (lines 431-576)
   - Interactive Plotly charts for model comparison
   - Three visualization types: Actual vs Predicted, Residuals, Error Distribution
   - Excel export with multiple sheets for complete analysis

4. **Prediction** (lines 578-705)
   - Uses trained models from session state
   - Applies same feature extraction pipeline to new data
   - Handles model-specific prediction methods (XGBoost DMatrix, Keras predict)
   - Inverse transforms predictions using stored scalers

### Key Technical Patterns

**Session State Management**: Critical for multi-page workflow
- `st.session_state['processed_features']`: Extracted features DataFrame
- `st.session_state['training_results']`: List of dicts containing models, scalers, and metrics

**Model Persistence**: Each training result contains:
```python
{
    'Model': str,           # Model name
    'trained_model': object,# Trained model instance
    'scaler_y': object,     # Target scaler for inverse transformation
    'MSE', 'MAE', 'RMSE', 'R2': float,  # Metrics
    'Actual', 'Predicted': list         # LOO-CV results
}
```

**Data Flow**: 
1. Raw voltammetric data → Feature extraction (16 features)
2. Features → Normalized → Model training with LOO-CV
3. New data → Same feature extraction → Normalized → Prediction → Inverse transform

## Scientific Context

This software analyzes electrochemical sensor signals to detect trace concentrations of analytes (chemicals of interest). Key applications:
- Environmental pollutant detection (e.g., PFOA at 10⁻⁸ M levels)
- Heavy metal monitoring in water
- Biomarker analysis for medical diagnostics
- Quality control in pharmaceutical manufacturing

The ML approach improves prediction accuracy by 25-30% over traditional electrochemical analysis methods.

## Published Research

Associated with the article "ElectroML: An Open-Source Software for Machine Learning-Based Analyte Concentration Prediction in Electrochemical Sensing" in SoftwareX journal. The software has been validated on nitro-substituted PANI nanocomposite sensors for dopamine detection.

## Contact Information

- Canan Hazal Akarsu: hazalakarsu@iuc.edu.tr
- Tarık Küçükdeniz: tkdeniz@iuc.edu.tr
import streamlit as st
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import io
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration and theme
st.set_page_config(
    page_title="DPV Analysis Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #0066cc;
        color: white;
    }
    .stProgress > div > div > div {
        background-color: #0066cc;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stAlert {
        background-color: #e3f2fd;
        color: #1565c0;
        border: none;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Main title with description
st.title("🔬 DPV Analysis Platform")
st.markdown("""
    Welcome to the Differential Pulse Voltammetry (DPV) Analysis Platform. This tool helps you analyze 
    voltammetric sensor data using advanced machine learning techniques.
    
    *Developed based on research in nitro-substituted PANI nanocomposite sensors.*
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Analysis Stage",
    ["📊 Data Preprocessing", "🧠 Model Training", "📈 Results Visualization", "🔮 Forecasting"]
)

def normalize_data(X_train, X_test, y_train, y_test):
    """Normalize training and test data"""
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y

def extract_features(data, voltages):
    """Extract features from voltammetric data"""
    features_df = pd.DataFrame()
    current_data = data.iloc[:, 1:]

    with st.expander("📌 Feature Extraction Process"):
        st.markdown("""
            The following features are being extracted from your voltammetric data:
            - **Peak Analysis**: Maximum/minimum currents and number of peaks
            - **Statistical Measures**: Mean, median, standard deviation
            - **Signal Characteristics**: Slope, area under curve, symmetry
            - **Advanced Statistics**: Skewness, kurtosis
        """)
        
        progress_bar = st.progress(0)
        progress_text = st.empty()

    # Basic statistical features
    progress_text.text("Calculating basic statistical features...")
    features_df['Imax'] = current_data.max(axis=1)
    features_df['Imin'] = current_data.min(axis=1)
    features_df['Imean'] = current_data.mean(axis=1)
    features_df['Imedian'] = current_data.median(axis=1)
    features_df['Istd'] = current_data.std(axis=1)
    progress_bar.progress(0.25)

    # Slope and peaks
    progress_text.text("Analyzing signal characteristics...")
    slopes = np.gradient(current_data.values, voltages, axis=1)
    features_df['Slope'] = slopes.mean(axis=1)
    features_df['Number_of_Peaks'] = current_data.apply(
        lambda row: len(find_peaks(row)[0]), axis=1
    )
    progress_bar.progress(0.50)

    # Additional features
    progress_text.text("Computing advanced features...")
    features_df['Positive_Area'] = current_data.apply(
        lambda row: np.trapz(row, voltages), axis=1
    )
    features_df['Symmetry'] = features_df['Imax'] - features_df['Imin']
    features_df['Skewness'] = current_data.apply(lambda row: skew(row), axis=1)
    features_df['Kurtosis'] = current_data.apply(lambda row: kurtosis(row), axis=1)
    progress_bar.progress(0.75)

    # Rate features
    progress_text.text("Finalizing feature extraction...")
    features_df['Initial_Current'] = current_data.iloc[:, 0]
    features_df['Final_Current'] = current_data.iloc[:, -1]
    features_df['Rising_Rate'] = current_data.diff(axis=1).max(axis=1)
    features_df['Falling_Rate'] = current_data.diff(axis=1).min(axis=1)
    progress_bar.progress(1.0)
    progress_text.text("Feature extraction completed!")

    return features_df

# Model training functions
def train_linear_regression(X_train, y_train):
    """Train Linear Regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train.ravel())
    return model

def train_svm(X_train, y_train):
    """Train SVM model"""
    model = SVR(C=1.0, gamma=0.1)
    model.fit(X_train, y_train.ravel())
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train.ravel())
    return model

def train_xgboost(X_train, y_train):
    """Train XGBoost model"""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {'objective': 'reg:squarederror', 'max_depth': 6, 'eta': 0.1}
    model = xgb.train(params, dtrain, num_boost_round=100)
    return model

def train_lightgbm(X_train, y_train):
    """Train LightGBM model"""
    model = lgb.LGBMRegressor(learning_rate=0.1, num_leaves=31, random_state=42)
    model.fit(X_train, y_train.ravel())
    return model

def train_ann(X_train, y_train):
    """Train ANN model"""
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    model.fit(X_train, y_train, epochs=50, verbose=0)
    return model

def evaluate_model(model, X_test, y_test, scaler_y, is_keras=False, is_xgboost=False):
    """Evaluate model performance"""
    if is_keras:
        y_pred_scaled = model.predict(X_test)
    elif is_xgboost:
        dtest = xgb.DMatrix(X_test)
        y_pred_scaled = model.predict(dtest)
    else:
        y_pred_scaled = model.predict(X_test)

    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    mse = mean_squared_error(y_test_original, y_pred)
    mae = mean_absolute_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred)
    
    return mse, mae, rmse, r2, y_test_original, y_pred

# Data Preprocessing Page
if page == "📊 Data Preprocessing":
    st.header("Data Preprocessing")
    
    st.info("""
    To facilitate hands-on testing of ElectroML, we provide a [sample dataset](https://github.com/tkucukdeniz2/ElectroML/sensor_data.xlsx) that demonstrates the required data format. The Excel file contains voltammetric measurements organized in a matrix structure where:
    
    - The first column 'Con' represents different analyte concentrations (1.25, 2.5, and 5.0 μM in the example)
    - Subsequent columns contain the measured current responses at different voltage points, ranging from -1.0V to +1.0V in 0.005V increments
    - Each row represents a complete voltammetric scan for a specific concentration
    - The dataset includes multiple measurements to demonstrate the framework's ability to handle real experimental data variability
    """)
    
    uploaded_file = st.file_uploader(
        "Upload your sensor data (Excel file)",
        type=['xlsx'],
        help="Upload an Excel file containing your voltammetric measurements"
    )
    
    if uploaded_file is not None:
        with st.spinner('Processing your data...'):
            data = pd.read_excel(uploaded_file)
            
            st.success("File uploaded successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Data Preview")
                st.dataframe(data.head(), use_container_width=True)
            
            with col2:
                st.subheader("Dataset Information")
                st.write(f"Total Samples: {len(data)}")
                st.write(f"Number of Features: {data.shape[1]}")
        
        concentration = data.iloc[:, 0]
        current_data = data.iloc[:, 1:]
        voltages = np.array(current_data.columns).astype(float)
        
        # Feature extraction
        with st.spinner('Extracting features...'):
            features_df = extract_features(data, voltages)
            features_df['Concentration'] = concentration
        
        st.subheader("Extracted Features")
        st.dataframe(features_df.head(), use_container_width=True)
        
        # Feature importance analysis
        X = features_df.drop(columns=['Concentration'])
        y = features_df['Concentration']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        st.subheader("Feature Importance Analysis")
        fig = px.bar(importance_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title='Feature Importance Analysis',
                    labels={'Importance': 'Relative Importance', 'Feature': 'Feature Name'},
                    color='Importance',
                    color_continuous_scale='viridis')
        
        fig.update_layout(
            showlegend=False,
            height=500,
            yaxis={'categoryorder':'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Save processed features
        st.session_state['processed_features'] = features_df
        
        # Download processed data
        st.download_button(
            label="📥 Download Processed Features",
            data=features_df.to_csv(index=False).encode('utf-8'),
            file_name="processed_features.csv",
            mime="text/csv",
            help="Download the extracted features as a CSV file"
        )

# Model Training Page
elif page == "🧠 Model Training":
    st.header("Model Training")
    
    if 'processed_features' not in st.session_state:
        st.warning("⚠️ No processed data available. Please preprocess your data first.")
        st.info("Go to the Data Preprocessing section to prepare your data.")
    else:
        features_df = st.session_state['processed_features']
        X = features_df.drop(columns=['Concentration'])
        y = features_df['Concentration']
        
        st.success("Features loaded successfully!")
        
        # Model selection
        st.subheader("Select Models to Train")
        model_descriptions = {
            'Linear Regression': 'Simple and interpretable baseline model',
            'SVM': 'Effective for non-linear relationships',
            'Random Forest': 'Ensemble method with good feature importance analysis',
            'XGBoost': 'Advanced gradient boosting for high performance',
            'LightGBM': 'Light and fast gradient boosting framework',
            'ANN': 'Deep learning approach for complex patterns'
        }
        
        model_functions = {
            'Linear Regression': train_linear_regression,
            'SVM': train_svm,
            'Random Forest': train_random_forest,
            'XGBoost': train_xgboost,
            'LightGBM': train_lightgbm,
            'ANN': train_ann
        }
        
        selected_models = []
        col1, col2 = st.columns(2)
        
        with col1:
            for model in list(model_descriptions.keys())[:3]:
                if st.checkbox(f"✓ {model}", help=model_descriptions[model]):
                    selected_models.append(model)
                    
        with col2:
            for model in list(model_descriptions.keys())[3:]:
                if st.checkbox(f"✓ {model}", help=model_descriptions[model]):
                    selected_models.append(model)
        
        if st.button("🚀 Train Selected Models"):
            if not selected_models:
                st.warning("Please select at least one model to train.")
            else:
                results = []
                loo = LeaveOneOut()
                
                # Create columns for parallel progress tracking
                cols = st.columns(len(selected_models))
                progress_bars = {model: cols[i].progress(0) for i, model in enumerate(selected_models)}
                status_texts = {model: cols[i].empty() for i, model in enumerate(selected_models)}
                
                for model_name in selected_models:
                    status_texts[model_name].text(f"Training {model_name}...")
                    
                    # Train final model on all data
                    X_all_scaled = MinMaxScaler().fit_transform(X)
                    y_all_scaled = MinMaxScaler().fit_transform(y.values.reshape(-1, 1))
                    final_model = model_functions[model_name](X_all_scaled, y_all_scaled)
                    
                    # Perform LOO CV for evaluation
                    actual_values, predicted_values = [], []
                    for i, (train_idx, test_idx) in enumerate(loo.split(X)):
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                        
                        # Normalize data
                        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y = normalize_data(
                            X_train, X_test, y_train, y_test
                        )
                        
                        # Train and evaluate
                        model = model_functions[model_name](X_train_scaled, y_train_scaled)
                        _, _, _, _, y_test_orig, y_pred = evaluate_model(
                            model, X_test_scaled, y_test_scaled, scaler_y,
                            is_keras=(model_name == 'ANN'),
                            is_xgboost=(model_name == 'XGBoost')
                        )
                        
                        actual_values.extend(y_test_orig)
                        predicted_values.extend(y_pred)
                        
                        # Update progress
                        progress = (i + 1) / len(X)
                        progress_bars[model_name].progress(progress)
                    
                    # Calculate final metrics
                    mse = mean_squared_error(actual_values, predicted_values)
                    mae = mean_absolute_error(actual_values, predicted_values)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(actual_values, predicted_values)
                    
                    # Store results
                    results.append({
                        'Model': model_name,
                        'MSE': mse,
                        'MAE': mae,
                        'RMSE': rmse,
                        'R2': r2,
                        'Actual': actual_values,
                        'Predicted': predicted_values,
                        'trained_model': final_model,
                        'scaler_y': scaler_y
                    })
                    
                    status_texts[model_name].success(f"{model_name} Complete!")
                
                # Save results in session state
                st.session_state['training_results'] = results
                
                # Display summary metrics
                st.subheader("📊 Model Performance Summary")
                metrics_df = pd.DataFrame([
                    {
                        'Model': r['Model'],
                        'MSE': round(r['MSE'], 4),
                        'MAE': round(r['MAE'], 4),
                        'RMSE': round(r['RMSE'], 4),
                        'R²': round(r['R2'], 4)
                    }
                    for r in results
                ])
                
                st.dataframe(
                    metrics_df.style.background_gradient(subset=['R²'], cmap='viridis'),
                    use_container_width=True
                )
                
                st.success("🎉 Training completed! Go to Results Visualization or Forecasting for detailed analysis.")

# Forecasting Page
elif page == "🔮 Forecasting":
    st.header("Concentration Forecasting")
    
    st.info("""
    Upload new DPV sensor data to predict analyte concentrations using trained models. 
    The input data should follow the same format as training data:
    - Current responses for each voltage point
    - No concentration column needed for prediction
    
    Download the [sample format](https://github.com/tkucukdeniz2/ElectroML/sensor_data.xlsx) for reference.
    """)
    
    if 'training_results' not in st.session_state:
        st.warning("⚠️ No trained models available. Please train models first.")
        st.info("Go to the Model Training section to train models before making predictions.")
    else:
        # File upload for new data
        uploaded_file = st.file_uploader(
            "Upload new DPV measurements (Excel file)",
            type=['xlsx'],
            help="Upload an Excel file containing new voltammetric measurements"
        )
        
        if uploaded_file is not None:
            with st.spinner('Processing new data...'):
                try:
                    # Load and process new data
                    new_data = pd.read_excel(uploaded_file)
                    
                    # Check if data format is correct
                    if new_data.shape[1] < 2:
                        st.error("Invalid data format. Please ensure your file contains voltage and current measurements.")
                        st.stop()
                    
                    voltages = np.array(new_data.columns[1:]).astype(float)
                    
                    # Extract features from new data
                    features_df = extract_features(new_data, voltages)
                    
                    # Model selection
                    available_models = [r['Model'] for r in st.session_state['training_results']]
                    selected_model = st.selectbox(
                        "Select Model for Prediction",
                        available_models,
                        help="Choose the model to use for concentration prediction"
                    )
                    
                    if st.button("🔮 Predict Concentration"):
                        with st.spinner('Making predictions...'):
                            # Get the selected model results
                            model_result = next(r for r in st.session_state['training_results'] 
                                             if r['Model'] == selected_model)
                            
                            # Normalize features
                            X = features_df
                            scaler_X = MinMaxScaler()
                            X_scaled = scaler_X.fit_transform(X)
                            
                            # Make predictions
                            model = model_result['trained_model']
                            if selected_model == 'XGBoost':
                                dtest = xgb.DMatrix(X_scaled)
                                predictions_scaled = model.predict(dtest)
                            elif selected_model == 'ANN':
                                predictions_scaled = model.predict(X_scaled)
                            else:
                                predictions_scaled = model.predict(X_scaled)
                            
                            # Inverse transform predictions
                            predictions = model_result['scaler_y'].inverse_transform(
                                predictions_scaled.reshape(-1, 1)
                            ).flatten()
                            
                            # Display results
                            st.subheader("Prediction Results")
                            
                            results_df = pd.DataFrame({
                                'Sample': range(1, len(predictions) + 1),
                                'Predicted Concentration (μM)': predictions
                            })
                            
                            st.dataframe(results_df.style.format({
                                'Predicted Concentration (μM)': '{:.3f}'
                            }))
                            
                            # Visualization
                            fig = px.scatter(
                                results_df,
                                x='Sample',
                                y='Predicted Concentration (μM)',
                                title='Predicted Concentrations by Sample',
                                labels={'Sample': 'Sample Number'}
                            )
                            fig.update_traces(marker=dict(size=10))
                            fig.update_layout(
                                height=500,
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Download predictions
                            st.download_button(
                                label="📥 Download Predictions",
                                data=results_df.to_csv(index=False).encode('utf-8'),
                                file_name="concentration_predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Model performance metrics
                            st.subheader("Model Performance Metrics")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("R² Score", f"{model_result['R2']:.3f}")
                            with col2:
                                st.metric("RMSE", f"{model_result['RMSE']:.3f}")
                            with col3:
                                st.metric("MAE", f"{model_result['MAE']:.3f}")
                            with col4:
                                st.metric("MSE", f"{model_result['MSE']:.3f}")
                            
                            st.info("""
                            Note: These metrics are based on the model's performance during training. 
                            Actual prediction accuracy may vary depending on the similarity between 
                            training data and new measurements.
                            """)
                            
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
                    st.info("Please ensure your input data follows the required format.")

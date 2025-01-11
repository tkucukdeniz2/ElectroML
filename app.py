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
    ["📊 Data Preprocessing", "🧠 Model Training", "📈 Results Visualization"]
)

def normalize_data(X_train, X_test, y_train, y_test):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y

def extract_features(data, voltages):
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
@st.cache_data
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train.ravel())
    return model

@st.cache_data
def train_svm(X_train, y_train):
    model = SVR(C=1.0, gamma=0.1)
    model.fit(X_train, y_train.ravel())
    return model

@st.cache_data
def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train.ravel())
    return model

@st.cache_data
def train_xgboost(X_train, y_train):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {'objective': 'reg:squarederror', 'max_depth': 6, 'eta': 0.1}
    model = xgb.train(params, dtrain, num_boost_round=100)
    return model

@st.cache_data
def train_lightgbm(X_train, y_train):
    model = lgb.LGBMRegressor(learning_rate=0.1, num_leaves=31, random_state=42)
    model.fit(X_train, y_train.ravel())
    return model

@st.cache_data
def train_ann(X_train, y_train):
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    model.fit(X_train, y_train, epochs=50, verbose=0)
    return model

def evaluate_model(model, X_test, y_test, scaler_y, is_keras=False, is_xgboost=False):
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
To facilitate hands-on testing of ElectroML, we provide a sample dataset (available at https://github.com/tkucukdeniz2/ElectroML/blob/9add814fab2edbb9b22b45d360df56b977732a4c/sensor_data.xlsx) that demonstrates the required data format. The Excel file contains voltammetric measurements organized in a matrix structure where: The first column 'Con' represents different analyte concentrations (1.25, 2.5, and 5.0 μM in the example)
Subsequent columns contain the measured current responses at different voltage points, ranging from -1.0V to +1.0V in 0.005V increments
Each row represents a complete voltammetric scan for a specific concentration
The dataset includes multiple measurements to demonstrate the framework's ability to handle real experimental data variability
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
        
        # Feature extraction with progress tracking
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
        
        # Create interactive feature importance plot
        fig = px.bar(importance_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title='Feature Importance in Dopamine Detection',
                    labels={'Importance': 'Relative Importance', 'Feature': 'Feature Name'},
                    color='Importance',
                    color_continuous_scale='viridis')
        
        fig.update_layout(
            showlegend=False,
            height=500,
            yaxis={'categoryorder':'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
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
    
    st.info("""
        This section allows you to train various machine learning models on your preprocessed data. 
        Select the models you want to evaluate and compare their performance in predicting dopamine concentrations.
    """)
    
    uploaded_file = st.file_uploader(
        "Upload processed features (CSV file)",
        type=['csv'],
        help="Upload the CSV file containing the processed features from the previous step"
    )
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        X = data.drop(columns=['Concentration'])
        y = data['Concentration']
        
        st.success("Features loaded successfully!")
        
        # Model selection with descriptions
        st.subheader("Select Models to Train")
        model_descriptions = {
            'Linear Regression': 'Simple and interpretable baseline model',
            'SVM': 'Effective for non-linear relationships',
            'Random Forest': 'Ensemble method with good feature importance analysis',
            'XGBoost': 'Advanced gradient boosting for high performance',
            'LightGBM': 'Light and fast gradient boosting framework',
            'ANN': 'Deep learning approach for complex patterns'
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
        
        if st.button("🚀 Train Selected Models", help="Start training the selected models"):
            if not selected_models:
                st.warning("Please select at least one model to train.")
            else:
                models = {
                    'Linear Regression': train_linear_regression,
                    'SVM': train_svm,
                    'Random Forest': train_random_forest,
                    'XGBoost': train_xgboost,
                    'LightGBM': train_lightgbm,
                    'ANN': train_ann
                }
                
                results = []
                loo = LeaveOneOut()
                
                # Create columns for parallel progress tracking
                cols = st.columns(len(selected_models))
                progress_bars = {model: cols[i].progress(0) for i, model in enumerate(selected_models)}
                status_texts = {model: cols[i].empty() for i, model in enumerate(selected_models)}
                
                for model_name in selected_models:
                    if model_name in models:
                        actual_values, predicted_values = [], []
                        train_func = models[model_name]
                        
                        status_texts[model_name].text(f"Training {model_name}...")
                        
                        for i, (train_index, test_index) in enumerate(loo.split(X)):
                            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                            
                            X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y = normalize_data(
                                X_train, X_test, y_train, y_test
                            )
                            
                            model = train_func(X_train_scaled, y_train_scaled)
                            _, _, _, _, y_test_original, y_pred = evaluate_model(
                                model, X_test_scaled, y_test_scaled, scaler_y,
                                is_keras=(model_name == 'ANN'),
                                is_xgboost=(model_name == 'XGBoost')
                            )
                            
                            actual_values.append(y_test_original[0])
                            predicted_values.append(y_pred[0])
                            
                            # Update progress
                            progress = (i + 1) / len(X)
                            progress_bars[model_name].progress(progress)
                        
                        # Calculate metrics
                        mse = mean_squared_error(actual_values, predicted_values)
                        mae = mean_absolute_error(actual_values, predicted_values)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(actual_values, predicted_values)
                        
                        results.append({
                            'Model': model_name,
                            'MSE': mse,
                            'MAE': mae,
                            'RMSE': rmse,
                            'R2': r2,
                            'Actual': actual_values,
                            'Predicted': predicted_values
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
                
                # Create a styled metrics table
                st.dataframe(
                    metrics_df.style.background_gradient(subset=['R²'], cmap='viridis'),
                    use_container_width=True
                )
                
                st.success("🎉 Training completed! Go to Results Visualization for detailed analysis.")

# Results Visualization Page
elif page == "📈 Results Visualization":
    st.header("Results Visualization")
    
    if 'training_results' not in st.session_state:
        st.warning("⚠️ No training results available. Please train models first.")
    else:
        results = st.session_state['training_results']
        
        # Model Performance Overview
        st.subheader("Model Performance Overview")
        metrics_df = pd.DataFrame([
            {
                'Model': r['Model'],
                'MSE': r['MSE'],
                'MAE': r['MAE'],
                'RMSE': r['RMSE'],
                'R²': r['R2']
            }
            for r in results
        ])
        
        # Interactive metric comparison
        col1, col2 = st.columns([1, 2])
        
        with col1:
            comparison_metric = st.selectbox(
                "Select Performance Metric",
                ["R²", "RMSE", "MSE", "MAE"],
                help="Choose the metric to compare across models"
            )
        
        with col2:
            fig = px.bar(
                metrics_df,
                x='Model',
                y=comparison_metric,
                title=f'Model Comparison by {comparison_metric}',
                color=comparison_metric,
                color_continuous_scale='viridis'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Model Analysis
        st.subheader("Detailed Model Analysis")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            selected_model = st.selectbox(
                "Select Model for Analysis",
                [r['Model'] for r in results]
            )
            
            viz_type = st.selectbox(
                "Select Visualization",
                ["Actual vs Predicted", "Residual Plot", "Error Distribution"]
            )
        
        with col2:
            model_data = next(r for r in results if r['Model'] == selected_model)
            
            if viz_type == "Actual vs Predicted":
                fig = px.scatter(
                    x=model_data['Actual'],
                    y=model_data['Predicted'],
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                    title=f'{selected_model} - Actual vs Predicted'
                )
                
                # Add ideal line
                min_val = min(min(model_data['Actual']), min(model_data['Predicted']))
                max_val = max(max(model_data['Actual']), max(model_data['Predicted']))
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Ideal',
                        line=dict(dash='dash', color='red')
                    )
                )
                
            elif viz_type == "Residual Plot":
                residuals = np.array(model_data['Predicted']) - np.array(model_data['Actual'])
                fig = px.scatter(
                    x=model_data['Predicted'],
                    y=residuals,
                    labels={'x': 'Predicted Values', 'y': 'Residuals'},
                    title=f'{selected_model} - Residual Plot'
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                
            else:  # Error Distribution
                residuals = np.array(model_data['Predicted']) - np.array(model_data['Actual'])
                fig = px.histogram(
                    x=residuals,
                    nbins=20,
                    title=f'{selected_model} - Error Distribution',
                    labels={'x': 'Prediction Error', 'y': 'Count'}
                )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Export Results
        st.subheader("Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download individual model results
            results_df = pd.DataFrame({
                'Actual': model_data['Actual'],
                'Predicted': model_data['Predicted']
            })
            
            st.download_button(
                label="📥 Download Selected Model Results",
                data=results_df.to_csv(index=False).encode('utf-8'),
                file_name=f"{selected_model.lower().replace(' ', '_')}_results.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export all results
            if st.button("📊 Export Complete Analysis"):
                buffer = io.BytesIO()
                
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    metrics_df.to_excel(writer, sheet_name='Metrics_Summary', index=False)
                    
                    for result in results:
                        model_name = result['Model']
                        pd.DataFrame({
                            'Actual': result['Actual'],
                            'Predicted': result['Predicted']
                        }).to_excel(writer, sheet_name=f'{model_name[:31]}_Results', index=False)
                
                buffer.seek(0)
                st.download_button(
                    label="📥 Download Complete Analysis",
                    data=buffer,
                    file_name="dpv_analysis_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

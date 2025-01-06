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

# Set page configuration
st.set_page_config(page_title="ML Model Training Pipeline", layout="wide")
st.title("Machine Learning Model Training Pipeline")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Select a Stage",
    ["Data Preprocessing", "Model Training", "Results Visualization"]
)

# Function definitions
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
    current_data = data.iloc[:, 1:]  # Current values

    # Basic statistical features
    features_df['Imax'] = current_data.max(axis=1)
    features_df['Imin'] = current_data.min(axis=1)
    features_df['Imean'] = current_data.mean(axis=1)
    features_df['Imedian'] = current_data.median(axis=1)
    features_df['Istd'] = current_data.std(axis=1)

    # Slope of current with respect to voltage
    slopes = np.gradient(current_data.values, voltages, axis=1)
    features_df['Slope'] = slopes.mean(axis=1)

    # Number of peaks
    features_df['Number_of_Peaks'] = current_data.apply(lambda row: len(find_peaks(row)[0]), axis=1)

    # Additional features
    features_df['Positive_Area'] = current_data.apply(lambda row: np.trapz(row, voltages), axis=1)
    features_df['Symmetry'] = features_df['Imax'] - features_df['Imin']
    features_df['Skewness'] = current_data.apply(lambda row: skew(row), axis=1)
    features_df['Kurtosis'] = current_data.apply(lambda row: kurtosis(row), axis=1)
    features_df['Initial_Current'] = current_data.iloc[:, 0]
    features_df['Final_Current'] = current_data.iloc[:, -1]
    features_df['Rising_Rate'] = current_data.diff(axis=1).max(axis=1)
    features_df['Falling_Rate'] = current_data.diff(axis=1).min(axis=1)

    return features_df

# Model training functions
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train.ravel())
    return model

def train_svm(X_train, y_train):
    model = SVR(C=1.0, gamma=0.1)
    model.fit(X_train, y_train.ravel())
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train.ravel())
    return model

def train_xgboost(X_train, y_train):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {'objective': 'reg:squarederror', 'max_depth': 6, 'eta': 0.1}
    model = xgb.train(params, dtrain, num_boost_round=100)
    return model

def train_lightgbm(X_train, y_train):
    model = lgb.LGBMRegressor(learning_rate=0.1, num_leaves=31, random_state=42)
    model.fit(X_train, y_train.ravel())
    return model

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
if page == "Data Preprocessing":
    st.header("Data Preprocessing")
    
    uploaded_file = st.file_uploader("Upload sensor data (Excel file)", type=['xlsx'])
    
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        st.write("Data Preview:", data.head())
        
        concentration = data.iloc[:, 0]
        current_data = data.iloc[:, 1:]
        voltages = np.array(current_data.columns).astype(float)
        
        # Feature extraction
        features_df = extract_features(data, voltages)
        features_df['Concentration'] = concentration
        
        st.write("Extracted Features Preview:", features_df.head())
        
        # Feature importance analysis
        X = features_df.drop(columns=['Concentration'])
        y = features_df['Concentration']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Plot feature importances
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        st.subheader("Feature Importance Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        st.pyplot(fig)
        
        # Save processed data
        st.download_button(
            label="Download Processed Features",
            data=features_df.to_csv(index=False).encode('utf-8'),
            file_name="output_features.csv",
            mime="text/csv"
        )

# Model Training Page
elif page == "Model Training":
    st.header("Model Training")
    
    uploaded_file = st.file_uploader("Upload processed features (CSV file)", type=['csv'])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        X = data.drop(columns=['Concentration'])
        y = data['Concentration']

        # Model selection
        selected_models = st.multiselect(
            "Select models to train",
            ['Linear Regression', 'SVM', 'Random Forest', 'XGBoost', 'LightGBM', 'ANN'],
            default=['Linear Regression', 'Random Forest']
        )
        
        if st.button("Train Selected Models"):
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
            
            for model_name in selected_models:
                if model_name in models:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    actual_values, predicted_values = [], []
                    train_func = models[model_name]
                    
                    status_text.text(f"Training {model_name}...")
                    
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
                        progress_bar.progress(progress)
                    
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
                    
                    status_text.text(f"{model_name} training completed!")
            
            # Save results
            if results:
                st.session_state['training_results'] = results
                st.success("Training completed! Go to Results Visualization to see the results.")

# Results Visualization Page
elif page == "Results Visualization":
    st.header("Results Visualization")
    
    if 'training_results' not in st.session_state:
        st.warning("No training results available. Please train models first.")
    else:
        results = st.session_state['training_results']
        
        # Metrics comparison
        st.subheader("Model Performance Metrics")
        metrics_df = pd.DataFrame([
            {
                'Model': r['Model'],
                'MSE': r['MSE'],
                'MAE': r['MAE'],
                'RMSE': r['RMSE'],
                'R2': r['R2']
            }
            for r in results
        ])
        st.write(metrics_df)
        
        # Visualization selection
        viz_type = st.selectbox(
            "Select visualization",
            ["Actual vs Predicted", "Residual Plot", "Error Distribution"]
        )
        
        selected_model = st.selectbox(
            "Select model",
            [r['Model'] for r in results]
        )
        
        # Get data for selected model
        model_data = next(r for r in results if r['Model'] == selected_model)
        
        if viz_type == "Actual vs Predicted":
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(model_data['Actual'], model_data['Predicted'], alpha=0.5)
            ax.plot([min(model_data['Actual']), max(model_data['Actual'])],
                   [min(model_data['Actual']), max(model_data['Actual'])],
                   'r--', label='Ideal')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{selected_model} - Actual vs Predicted')
            ax.legend()
            st.pyplot(fig)
            
        elif viz_type == "Residual Plot":
            residuals = np.array(model_data['Predicted']) - np.array(model_data['Actual'])
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(model_data['Predicted'], residuals, alpha=0.5)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title(f'{selected_model} - Residual Plot')
            st.pyplot(fig)
            
        else:  # Error Distribution
            residuals = np.array(model_data['Predicted']) - np.array(model_data['Actual'])
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(residuals, kde=True, ax=ax)
            ax.set_xlabel('Prediction Error')
            ax.set_ylabel('Count')
            ax.set_title(f'{selected_model} - Error Distribution')
            st.pyplot(fig)

        # Download results
        results_df = pd.DataFrame({
            'Actual': model_data['Actual'],
            'Predicted': model_data['Predicted']
        })
        
        st.download_button(
            label="Download Results",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name=f"{selected_model.lower().replace(' ', '_')}_results.csv",
            mime="text/csv"
        )

        # Compare all models
        st.subheader("Model Comparison")
        comparison_metric = st.selectbox(
            "Select metric for comparison",
            ["R2", "RMSE", "MSE", "MAE"]
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=metrics_df, x='Model', y=comparison_metric)
        plt.xticks(rotation=45)
        plt.title(f'Model Comparison by {comparison_metric}')
        st.pyplot(fig)

        # Export all results
        if st.button("Export All Results"):
            # Create a BytesIO object to store the Excel file
            buffer = io.BytesIO()
            
            # Create Excel writer object
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # Write metrics summary
                metrics_df.to_excel(writer, sheet_name='Metrics_Summary', index=False)
                
                # Write detailed results for each model
                for result in results:
                    model_name = result['Model']
                    pd.DataFrame({
                        'Actual': result['Actual'],
                        'Predicted': result['Predicted']
                    }).to_excel(writer, sheet_name=f'{model_name[:31]}_Results', index=False)
            
            # Prepare download button
            buffer.seek(0)
            st.download_button(
                label="Download Complete Results",
                data=buffer,
                file_name="all_model_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
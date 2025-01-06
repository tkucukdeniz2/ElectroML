import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit App Title
st.title("Sensor Data Analysis and Prediction App")

# Step 1: Data Upload
st.header("Step 1: Upload Sensor Data")
uploaded_file = st.file_uploader("Upload your sensor_data.xlsx file", type=["xlsx"])

if uploaded_file:
    # Load Data
    data = pd.read_excel(uploaded_file)
    st.write("Dataset Preview:", data.head())

    # Extract Concentration and Current Data
    concentration = data.iloc[:, 0]  # Target variable
    current_data = data.iloc[:, 1:]  # Current values
    voltages = np.array(current_data.columns).astype(float)  # Voltage values

    # Step 2: Feature Extraction
    st.header("Step 2: Feature Extraction")
    if st.button("Extract Features"):
        features_df = pd.DataFrame()

        # Statistical Features
        features_df['Imax'] = current_data.max(axis=1)
        features_df['Imin'] = current_data.min(axis=1)
        features_df['Imean'] = current_data.mean(axis=1)
        features_df['Imedian'] = current_data.median(axis=1)
        features_df['Istd'] = current_data.std(axis=1)

        # Slope of Current
        slopes = np.gradient(current_data.values, voltages, axis=1)
        features_df['Slope'] = slopes.mean(axis=1)

        # Number of Peaks
        features_df['Number_of_Peaks'] = current_data.apply(lambda row: len(find_peaks(row)[0]), axis=1)

        # Positive Area Under Curve
        features_df['Positive_Area'] = current_data.apply(lambda row: np.trapz(row, voltages), axis=1)

        # Symmetry
        features_df['Symmetry'] = features_df['Imax'] - features_df['Imin']

        # Skewness and Kurtosis
        features_df['Skewness'] = current_data.apply(lambda row: skew(row), axis=1)
        features_df['Kurtosis'] = current_data.apply(lambda row: kurtosis(row), axis=1)

        # Initial and Final Current
        features_df['Initial_Current'] = current_data.iloc[:, 0]
        features_df['Final_Current'] = current_data.iloc[:, -1]

        # Rising and Falling Rates
        features_df['Rising_Rate'] = current_data.diff(axis=1).max(axis=1)
        features_df['Falling_Rate'] = current_data.diff(axis=1).min(axis=1)

        # Add Target Variable
        features_df['Concentration'] = concentration
        st.write("Extracted Features:", features_df.head())

    # Step 3: Model Selection and Training
    st.header("Step 3: Model Training")
    model_choice = st.selectbox("Select a Model for Training", ["Linear Regression", "SVM", "Random Forest", "XGBoost", "LightGBM", "CatBoost", "ANN"])
    if st.button("Train Model"):
        # Prepare Data
        X = features_df.drop(columns=['Concentration'])
        y = features_df['Concentration']

        def normalize_data(X_train, X_test, y_train, y_test):
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
            y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
            return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y

        # Train Model Function
        def train_model(X_train, y_train):
            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "SVM":
                model = SVR()
            elif model_choice == "Random Forest":
                model = RandomForestRegressor()
            elif model_choice == "XGBoost":
                model = xgb.XGBRegressor(objective="reg:squarederror")
            elif model_choice == "LightGBM":
                model = lgb.LGBMRegressor()
            elif model_choice == "CatBoost":
                model = CatBoostRegressor(verbose=0)
            elif model_choice == "ANN":
                model = Sequential([
                    Dense(64, activation='relu', input_dim=X_train.shape[1]),
                    Dense(32, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
                model.fit(X_train, y_train, epochs=50, verbose=0)
                return model
            model.fit(X_train, y_train)
            return model

        # Leave-One-Out Cross-Validation
        loo = LeaveOneOut()
        actual_values, predicted_values = [], []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y = normalize_data(X_train, X_test, y_train, y_test)
            model = train_model(X_train_scaled, y_train_scaled)
            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
            actual_values.append(y_test_original[0])
            predicted_values.append(y_pred[0])

        # Evaluate Model
        mse = mean_squared_error(actual_values, predicted_values)
        mae = mean_absolute_error(actual_values, predicted_values)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_values, predicted_values)

        st.write(f"Model: {model_choice}")
        st.write(f"MSE: {mse}, MAE: {mae}, RMSE: {rmse}, R2: {r2}")

    # Step 4: Visualization
    st.header("Step 4: Visualization")
    if st.button("Visualize Predictions"):
        plt.figure(figsize=(10, 6))
        plt.scatter(actual_values, predicted_values, alpha=0.7)
        plt.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], 'k--', lw=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Actual vs Predicted for {model_choice}")
        st.pyplot(plt)

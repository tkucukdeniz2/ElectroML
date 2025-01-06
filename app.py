import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Title and Sidebar
st.title("Sensor Data Analysis and Feature Extraction")
st.sidebar.header("Options")

# Upload File
uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])

# Define Functions
def extract_features(data, voltages):
    features_df = pd.DataFrame()

    # Basic statistical features
    features_df['Imax'] = data.max(axis=1)
    features_df['Imin'] = data.min(axis=1)
    features_df['Imean'] = data.mean(axis=1)
    features_df['Imedian'] = data.median(axis=1)
    features_df['Istd'] = data.std(axis=1)

    # Slope of current with respect to voltage
    slopes = np.gradient(data.values, voltages, axis=1)
    features_df['Slope'] = slopes.mean(axis=1)

    # Number of peaks
    features_df['Number_of_Peaks'] = data.apply(lambda row: len(find_peaks(row)[0]), axis=1)

    # Positive area under the curve
    features_df['Positive_Area'] = data.apply(lambda row: np.trapz(row, voltages), axis=1)

    # Symmetry (difference between max and min current)
    features_df['Symmetry'] = features_df['Imax'] - features_df['Imin']

    # Skewness and kurtosis
    features_df['Skewness'] = data.apply(lambda row: skew(row), axis=1)
    features_df['Kurtosis'] = data.apply(lambda row: kurtosis(row), axis=1)

    # Initial and final current values
    features_df['Initial_Current'] = data.iloc[:, 0]
    features_df['Final_Current'] = data.iloc[:, -1]

    # Current change rates (rising and falling)
    features_df['Rising_Rate'] = data.diff(axis=1).max(axis=1)
    features_df['Falling_Rate'] = data.diff(axis=1).min(axis=1)

    return features_df

def train_model(features, target):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, target)
    feature_importances = pd.DataFrame({
        'Feature': features.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    return feature_importances

def plot_feature_importance(importance_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    st.pyplot(plt)

def plot_correlation(correlation_df):
    plt.figure(figsize=(10, 6))
    colors = ['red' if c > 0 else 'blue' for c in correlation_df['Correlation']]
    sns.barplot(x='Correlation', y='Feature', data=correlation_df, palette=colors)
    plt.title('Feature Correlation with Concentration')
    st.pyplot(plt)

# App Logic
if uploaded_file:
    # Load the dataset
    data = pd.read_excel(uploaded_file)
    concentration = data.iloc[:, 0]
    current_data = data.iloc[:, 1:]
    voltages = np.array(current_data.columns).astype(float)

    # Feature Extraction
    features_df = extract_features(current_data, voltages)
    features_df['Concentration'] = concentration
    st.write("Extracted Features", features_df)

    # Train Model and Feature Importance
    X = features_df.drop(columns=['Concentration'])
    y = features_df['Concentration']
    feature_importances = train_model(X, y)
    st.write("Feature Importances", feature_importances)
    plot_feature_importance(feature_importances)

    # Feature Correlation with Target
    correlations = X.corrwith(y)
    correlation_df = pd.DataFrame({
        'Feature': X.columns,
        'Correlation': correlations
    }).sort_values(by='Correlation', ascending=False)
    st.write("Feature Correlations", correlation_df)
    plot_correlation(correlation_df)

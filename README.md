# ElectroML

An Open-Source Software for Machine Learning-Based Analyte Concentration Prediction in Electrochemical Sensing

ElectroML is an open-source machine learning (ML) software designed to automate the processing, analysis, and prediction of analyte concentrations from cyclic voltammetry (CV) and differential pulse voltammetry (DPV) data. The software integrates six state-of-the-art ML models including Artificial Neural Networks, Support Vector Machines, and gradient boosting methods, providing a streamlined and accessible platform for researchers across various scientific domains.

This software is associated with the article titled **"ElectroML: An Open-Source Software for Machine Learning-Based Analyte Concentration Prediction in Electrochemical Sensing"** submitted to SoftwareX.

## Features
- **Data Preprocessing Module:** Advanced feature extraction from voltammetric data.
- **Model Training Module:** Training multiple ML models with LOOCV validation.
- **Visualization Module:** Publication-quality plots (Actual vs Predicted, Residuals, Error Distribution).
- **Prediction Module:** Rapid analyte concentration prediction from new measurements.
- **Streamlit Web Interface:** Intuitive and interactive usage without coding.
- **Fully Open-Source and Modular Design:** Easily extensible and reproducible.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/tkucukdeniz2/ElectroML.git
cd ElectroML
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Once the application launches, use the sidebar to navigate through the modules:
- **Data Preprocessing:** Upload your voltammetric dataset and extract features.
- **Model Training:** Train various machine learning models.
- **Results Visualization:** Analyze model performances with detailed plots.
- **Prediction:** Predict analyte concentrations from new sensor data.

## Example Data

A sample dataset (`sensor_data.xlsx`) is provided in the repository. It demonstrates the required input format:
- First column: "Con" (Analyte Concentration)
- Following columns: Current measurements at different voltages (-1V to +1V)

## License

This project is licensed under the GNU General Public License v3.0. See the `LICENSE.txt` file for details.

## Contact

For questions, suggestions, or support, please contact:
- Canan Hazal Akarsu: hazalakarsu@iuc.edu.tr
- Tarık Küçükdeniz: tkdeniz@iuc.edu.tr
# ElectroML

An Open-Source Web Platform for Machine Learning-Based Analyte Concentration Prediction in Electrochemical Sensing

ElectroML is a sophisticated web-based platform designed to automate the processing, analysis, and prediction of analyte concentrations from cyclic voltammetry (CV) and differential pulse voltammetry (DPV) data. The application integrates state-of-the-art machine learning models with an intuitive web interface, providing a professional platform for electrochemical data analysis that works with any modern web browser.

## 🌟 Key Features

### Advanced Data Processing
- **Comprehensive Feature Extraction:** Extracts 40+ features including statistical, peak-related, derivative, integral, and frequency domain features
- **Signal Processing:** Savitzky-Golay filtering for noise reduction with customizable parameters
- **Drag-and-Drop Interface:** Intuitive file upload with support for Excel and CSV formats
- **Real-time Visualization:** Interactive plots using Plotly.js

### Machine Learning Models
- **Multiple Algorithms:** Linear Regression, Ridge, Lasso, ElasticNet, SVM, Random Forest, Gradient Boosting, XGBoost, LightGBM
- **Hyperparameter Optimization:** Automated tuning using Optuna with Bayesian optimization
- **Cross-Validation:** Leave-One-Out, K-Fold strategies with automatic selection for small datasets
- **Model Comparison:** Side-by-side performance evaluation

### Web-Based Interface
- **No Installation Required:** Runs in any modern web browser
- **Responsive Design:** Works on desktop, tablet, and mobile devices
- **Real-time Updates:** Live progress tracking during processing
- **Interactive Visualizations:** Zoomable, exportable plots

### Professional Features
- **Session Management:** Multiple users can work simultaneously
- **Export Capabilities:** Download results in Excel, CSV formats
- **Publication-Ready Plots:** High-quality visualizations for journals
- **Batch Processing:** Handle multiple datasets efficiently

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher (works with Python 3.13!)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/tkucukdeniz2/ElectroML.git
cd ElectroML
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python run.py
```

The application will automatically open in your default web browser at `http://localhost:5000`

## 📊 Usage

### Workflow

1. **Upload Data**
   - Drag and drop your voltammetric data file
   - Supported formats: Excel (.xlsx, .xls) and CSV
   - Data format: First column = Concentration, remaining columns = Current measurements

2. **Extract Features**
   - Configure preprocessing options (smoothing, window length)
   - Automatic feature extraction and importance ranking
   - Visual feature importance analysis

3. **Train Models**
   - Select one or multiple ML algorithms
   - Optional hyperparameter optimization
   - Real-time training progress
   - Comprehensive performance metrics

4. **Make Predictions**
   - Upload new measurements
   - Select trained model
   - Get concentration predictions with confidence intervals
   - Export results for further analysis

### Example Data

A sample dataset (`src/sensor_data.xlsx`) is provided demonstrating the required format:
- Column 1: "Con" - Analyte concentration (μM)
- Columns 2+: Current measurements from -1V to +1V (0.005V increments)

## 🏗️ Architecture

```
ElectroML/
├── src/
│   ├── app.py                    # Flask application
│   ├── templates/                # HTML templates
│   │   └── index.html           # Main web interface
│   ├── static/                   # Static assets
│   │   ├── css/style.css        # Custom styles
│   │   └── js/app.js            # Frontend JavaScript
│   ├── core/                     # Core algorithms
│   │   ├── feature_extraction.py # Feature engineering
│   │   └── model_training.py     # Training pipeline
│   ├── models/                   # ML models
│   │   └── model_factory.py      # Model creation
│   └── api/                      # API endpoints
├── run.py                        # Application launcher
├── requirements.txt              # Python dependencies
├── LICENSE.txt                   # GNU GPL v3.0
└── README.md                     # This file
```

## 🔬 Scientific Applications

ElectroML has been validated for:
- **Environmental Monitoring:** Detection of pollutants (PFOA at 10⁻⁸ M levels)
- **Heavy Metal Analysis:** Simultaneous detection in water samples
- **Biomedical Diagnostics:** Biomarker quantification (dopamine, tyrosine, uric acid)
- **Quality Control:** Industrial process monitoring
- **Materials Research:** Sensor development and optimization

## 📈 Performance

- **Prediction Accuracy:** 25-30% improvement over traditional methods
- **Processing Speed:** Feature extraction < 1 second for 100 samples
- **Scalability:** Handles datasets with 10,000+ samples
- **Small Dataset Support:** Automatic LOO-CV for < 5 samples

## 🌐 Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 🔧 Advanced Configuration

### Running on Different Port
```bash
PORT=8080 python run.py
```

### Production Deployment
For production deployment, use a WSGI server:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 src.app:app
```

## 📝 Publication

This software is associated with the article:
**"ElectroML: An Open-Source Web Platform for Machine Learning-Based Analyte Concentration Prediction in Electrochemical Sensing"** 
Submitted to SoftwareX journal.

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.txt](LICENSE.txt) file for details.

## 🤝 Contributing

We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📧 Support

For questions, bug reports, or feature requests:
- Open an issue on [GitHub](https://github.com/tkucukdeniz2/ElectroML/issues)
- Contact the developers (see below)

## 👥 Authors

- **Canan Hazal Akarsu** - *Lead Developer* - hazalakarsu@iuc.edu.tr
- **Tarık Küçükdeniz** - *Co-Developer* - tkdeniz@iuc.edu.tr
- **Elif Tüzün** 
- **Selcan Karakuş**

## 🙏 Acknowledgments

- Corporate Data Management Office, Istanbul University-Cerrahpaşa
- Department of Industrial Engineering, Istanbul University-Cerrahpaşa
- Department of Chemistry, Istanbul University-Cerrahpaşa
- Health Biotechnology Joint Research and Application Center of Excellence, Istanbul University-Cerrahpaşa

## 📚 Citation

If you use ElectroML in your research, please cite:

```bibtex
@article{akarsu2025electroml,
  title={ElectroML: An Open-Source Web Platform for Machine Learning-Based Analyte Concentration Prediction in Electrochemical Sensing},
  author={Akarsu, Canan Hazal and Küçükdeniz, Tarık and Tüzün, Elif and Karakuş, Selcan},
  journal={SoftwareX},
  year={2025},
  publisher={Elsevier}
}
```

## 🚀 What's New in v2.1

- **Web-Based Architecture:** No desktop GUI dependencies, works with Python 3.13+
- **Modern Interface:** Responsive design with Bootstrap 5
- **Interactive Visualizations:** Plotly.js for dynamic, exportable plots
- **Session Management:** Multi-user support
- **Enhanced Feature Extraction:** 40+ features with automatic importance ranking
- **Improved Model Training:** Better handling of small datasets
- **Real-time Progress:** Live updates during processing
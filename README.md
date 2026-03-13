# ElectroML

An Open-Source Web Platform for Machine Learning-Based Analyte Concentration Prediction in Electrochemical Sensing

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

ElectroML is a web-based platform for automated preprocessing, feature extraction, machine learning model training, and concentration prediction from cyclic voltammetry (CV) and differential pulse voltammetry (DPV) data. The application integrates 14 regression algorithms with an intuitive browser interface, providing a professional platform for electrochemical data analysis.

## Key Features

### Data Processing
- **47 Electrochemical Features:** Statistical, peak-related, derivative, integral, shape, and frequency-domain features extracted automatically
- **Signal Processing:** Savitzky-Golay filtering with customizable parameters and in-app parameter guidance
- **Drag-and-Drop Interface:** Support for Excel and CSV formats
- **Interactive Visualization:** Plotly.js-based zoomable, exportable plots

### Machine Learning
- **14 Algorithms:** Linear Regression, Ridge, Lasso, ElasticNet, SVM, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, AdaBoost, MLP, Neural Network, Gaussian Process
- **Hyperparameter Optimization:** Optuna with TPE sampler (100 trials default)
- **Cross-Validation:** Leave-One-Out (recommended for small datasets) and K-Fold
- **Leakage Prevention:** All models wrapped in sklearn Pipeline — StandardScaler fitted only on training folds
- **Honest Metrics:** All reported R², RMSE, MAE computed from out-of-fold predictions via `cross_val_predict`

### Web Interface
- **No Installation Required:** Runs in any modern web browser
- **Responsive Design:** Desktop, tablet, and mobile support
- **Session Management:** Multiple users can work simultaneously
- **Publication-Ready Exports:** PNG, SVG, PDF, and interactive HTML

## Quick Start

### Prerequisites
- Python 3.8+ (tested with Python 3.13)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

```bash
git clone https://github.com/tkucukdeniz2/ElectroML.git
cd ElectroML
pip install -r requirements.txt
python run.py
```

The application opens automatically at `http://localhost:5005`.

## Usage

### Workflow

1. **Upload Data** — Drag and drop your voltammetric dataset (Excel/CSV). Format: first column = concentration, remaining columns = current measurements with voltage values as headers.
2. **Preprocess** — Configure outlier removal, baseline correction, signal filtering, and normalization.
3. **Extract Features** — 47 features computed per sample with automatic importance ranking.
4. **Train Models** — Select algorithms, choose CV strategy, optionally enable Optuna tuning. All metrics are cross-validated.
5. **Predict** — Upload new measurements, select a trained model, get predictions with confidence intervals.
6. **Visualize** — Generate publication-ready voltammograms, scatter plots, and model comparison figures.

### Sample Datasets

| File | Type | Samples | Concentrations | Voltage Range |
|------|------|---------|----------------|---------------|
| `data/data_electroml.xlsx` | DPV | 9 | 0.1–0.85 mM | 0.009–0.799 V |
| `src/sensor_data.xlsx` | CV | 4 | 0.625–5.00 μM | -1.0–1.0 V |

### Benchmark Results (DPV dataset, LOO-CV)

| Model | R² (CV) | RMSE (mM) | MAE (mM) |
|-------|---------|-----------|----------|
| Ridge | 0.9815 | 0.0322 | 0.0254 |
| Linear Regression | 0.9752 | 0.0373 | 0.0292 |
| MLP | 0.9572 | 0.0491 | 0.0375 |
| Extra Trees | 0.9359 | 0.0600 | 0.0394 |
| Gaussian Process | 0.9009 | 0.0746 | 0.0437 |
| Gradient Boosting | 0.8667 | 0.0865 | 0.0687 |

All metrics computed from out-of-fold predictions using sklearn Pipeline with StandardScaler inside CV folds.

## Architecture

```
ElectroML/
├── run.py                           # Application launcher
├── requirements.txt                 # Python dependencies
├── src/
│   ├── app_refactored.py            # Flask app factory
│   ├── api/                         # REST API (Flask Blueprints)
│   │   ├── data_handler.py          # File upload & sessions
│   │   ├── preprocessing.py         # Signal processing
│   │   ├── training.py              # Feature extraction & model training
│   │   ├── prediction.py            # Inference & confidence intervals
│   │   └── visualization.py         # Publication-ready plots
│   ├── core/                        # Business logic
│   │   ├── feature_extraction.py    # 47 features in 6 categories
│   │   └── model_training.py        # Training pipeline with Optuna
│   ├── models/
│   │   └── model_factory.py         # 14 algorithms via factory pattern
│   ├── utils/                       # Session management, validation, JSON helpers
│   ├── templates/
│   │   └── index.html               # Bootstrap 5 frontend
│   └── static/                      # CSS, JS assets
├── data/
│   └── data_electroml.xlsx          # DPV sample dataset
└── LICENSE.txt                      # GNU GPL v3.0
```

## Scientific Applications

- **Environmental Monitoring:** Detection of pollutants at trace levels
- **Heavy Metal Analysis:** Simultaneous detection in water samples
- **Biomedical Diagnostics:** Biomarker quantification (dopamine, uric acid)
- **Quality Control:** Industrial process monitoring
- **Materials Research:** Sensor development and optimization

## Advanced Configuration

### Running on a Different Port
```bash
PORT=8080 python run.py
```

### Production Deployment
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5005 "src.app_refactored:create_app()"
```

## Publication

This software is associated with the article:

**"ElectroML: An Open-Source Web Platform for Machine Learning-Based Analyte Concentration Prediction in Electrochemical Sensing"**
Submitted to *SoftwareX* (Elsevier).

## License

GNU General Public License v3.0 — see [LICENSE.txt](LICENSE.txt).

## Authors

- **Canan Hazal Akarsu** — hazalakarsu@iuc.edu.tr
- **Tarik Kucukdeniz** — tkdeniz@iuc.edu.tr
- **Elif Tuzun**
- **Selcan Karakus**

Istanbul University-Cerrahpasa

## Citation

```bibtex
@article{akarsu2025electroml,
  title={ElectroML: An Open-Source Web Platform for Machine Learning-Based Analyte Concentration Prediction in Electrochemical Sensing},
  author={Akarsu, Canan Hazal and K{\"u}{\c{c}}{\"u}kdeniz, Tar{\i}k and T{\"u}z{\"u}n, Elif and Karaku{\c{s}}, Selcan},
  journal={SoftwareX},
  year={2025},
  publisher={Elsevier}
}
```

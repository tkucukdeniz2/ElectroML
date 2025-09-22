// ElectroML Web Application JavaScript

let sessionId = null;
let currentData = null;
let trainedModels = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
});

function initializeEventListeners() {
    // File input
    const fileInput = document.getElementById('fileInput');
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    const dropZone = document.getElementById('dropZone');
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);
    
    // Prediction file input
    const predFileInput = document.getElementById('predFileInput');
    if (predFileInput) {
        predFileInput.addEventListener('change', handlePredictionFileSelect);
    }
}

// File handling
function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        uploadFile(file);
    }
}

function uploadFile(file) {
    // Validate file type
    const validTypes = ['csv', 'xlsx', 'xls'];
    const fileType = file.name.split('.').pop().toLowerCase();
    
    if (!validTypes.includes(fileType)) {
        showAlert('Invalid file type. Please upload CSV or Excel file.', 'danger');
        return;
    }
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    // Show loading
    showLoading('Uploading and processing file...');
    
    // Upload file
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.error) {
            showAlert(data.error, 'danger');
            return;
        }
        
        // Store session ID
        sessionId = data.session_id;
        currentData = data;
        
        // Display data preview
        displayDataPreview(data);
        
        // Update workflow
        updateWorkflowStep(1, 'completed');
        updateWorkflowStep(2, 'active');
        
        // Enable features tab
        document.getElementById('features-tab').disabled = false;
        
        showAlert('File uploaded successfully!', 'success');
    })
    .catch(error => {
        hideLoading();
        showAlert('Error uploading file: ' + error, 'danger');
    });
}

function displayDataPreview(data) {
    // Show preview section
    document.getElementById('dataPreview').style.display = 'block';
    
    // Update info
    document.getElementById('fileName').textContent = data.filename;
    document.getElementById('nSamples').textContent = data.shape[0];
    document.getElementById('nFeatures').textContent = data.shape[1];
    document.getElementById('concRange').textContent = 
        `${data.concentration_range.min.toFixed(2)} - ${data.concentration_range.max.toFixed(2)} μM`;
    
    // Display plot
    if (data.plot) {
        Plotly.newPlot('voltammogramPlot', data.plot.data, data.plot.layout);
    }
    
    // Display table preview
    displayDataTable(data.sample_data, data.columns);
}

function displayDataTable(data, columns) {
    const table = document.getElementById('dataTable');
    const thead = table.querySelector('thead');
    const tbody = table.querySelector('tbody');
    
    // Clear existing content
    thead.innerHTML = '';
    tbody.innerHTML = '';
    
    // Create header
    const headerRow = document.createElement('tr');
    columns.slice(0, 10).forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        headerRow.appendChild(th);
    });
    if (columns.length > 10) {
        const th = document.createElement('th');
        th.textContent = '...';
        headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    
    // Create rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        columns.slice(0, 10).forEach(col => {
            const td = document.createElement('td');
            const value = row[col];
            td.textContent = typeof value === 'number' ? value.toFixed(4) : value;
            tr.appendChild(td);
        });
        if (columns.length > 10) {
            const td = document.createElement('td');
            td.textContent = '...';
            tr.appendChild(td);
        }
        tbody.appendChild(tr);
    });
}

// Feature extraction
function proceedToFeatures() {
    // Switch to features tab
    const featuresTab = new bootstrap.Tab(document.getElementById('features-tab'));
    featuresTab.show();
}

function extractFeatures() {
    if (!sessionId) {
        showAlert('No data loaded. Please upload data first.', 'warning');
        return;
    }
    
    const requestData = {
        session_id: sessionId,
        smoothing: document.getElementById('smoothingCheck').checked,
        window_length: parseInt(document.getElementById('windowLength').value),
        polyorder: parseInt(document.getElementById('polyOrder').value)
    };
    
    showLoading('Extracting features...');
    
    fetch('/api/extract_features', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.error) {
            showAlert(data.error, 'danger');
            return;
        }
        
        // Display results
        document.getElementById('featureResults').style.display = 'block';
        document.getElementById('nExtractedFeatures').textContent = data.n_features;
        document.getElementById('nExtractedSamples').textContent = data.n_samples;
        
        // Show educational note if provided
        if (data.educational_note) {
            const noteDiv = document.createElement('div');
            noteDiv.className = 'alert alert-info mt-2';
            noteDiv.innerHTML = `<i class="fas fa-info-circle"></i> <strong>For Chemists:</strong> ${data.educational_note}`;
            document.getElementById('featureResults').insertBefore(noteDiv, document.getElementById('featureResults').firstChild);
        }
        
        // Display feature importance plot with tooltips
        if (data.importance_plot) {
            Plotly.newPlot('featureImportancePlot', data.importance_plot.data, data.importance_plot.layout);
            
            // Add feature importance table with tooltips
            if (data.top_features) {
                displayFeatureImportanceWithTooltips(data.top_features);
            }
        }
        
        // Update workflow
        updateWorkflowStep(2, 'completed');
        updateWorkflowStep(3, 'active');
        
        // Enable training tab
        document.getElementById('training-tab').disabled = false;
        
        showAlert('Features extracted successfully!', 'success');
    })
    .catch(error => {
        hideLoading();
        showAlert('Error extracting features: ' + error, 'danger');
    });
}

// Model training
function proceedToTraining() {
    const trainingTab = new bootstrap.Tab(document.getElementById('training-tab'));
    trainingTab.show();
}

function trainModels() {
    if (!sessionId) {
        showAlert('No data loaded. Please upload data first.', 'warning');
        return;
    }
    
    // Get selected models
    const selectedModels = [];
    document.querySelectorAll('.model-check:checked').forEach(checkbox => {
        selectedModels.push(checkbox.value);
    });
    
    if (selectedModels.length === 0) {
        showAlert('Please select at least one model.', 'warning');
        return;
    }
    
    const requestData = {
        session_id: sessionId,
        models: selectedModels,
        optimize_hyperparameters: document.getElementById('optimizeCheck').checked,
        cv_strategy: 'loo'  // Use LOO for small datasets
    };
    
    // Show progress
    document.getElementById('trainingProgress').style.display = 'block';
    document.getElementById('trainingResults').style.display = 'none';
    
    fetch('/api/train', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('trainingProgress').style.display = 'none';
        
        if (data.error) {
            showAlert(data.error, 'danger');
            return;
        }
        
        // Display results
        document.getElementById('trainingResults').style.display = 'block';
        displayTrainingResults(data.results);
        
        // Add educational interpretations
        if (data.best_model && data.model_interpretation) {
            const interpretDiv = document.createElement('div');
            interpretDiv.className = 'alert alert-success mt-3';
            interpretDiv.innerHTML = `
                <h6><i class="fas fa-trophy"></i> Best Model: ${data.best_model.replace(/_/g, ' ').toUpperCase()}</h6>
                <p class="mb-1">${data.model_interpretation}</p>
                <p class="mb-0"><strong>Performance:</strong> ${data.r2_interpretation}</p>
            `;
            document.querySelector('#trainingResults').insertBefore(interpretDiv, document.querySelector('#resultsTable').parentElement);
        }
        
        if (data.educational_note) {
            const noteDiv = document.createElement('div');
            noteDiv.className = 'alert alert-info mt-2';
            noteDiv.innerHTML = `<i class="fas fa-graduation-cap"></i> <strong>ML Note:</strong> ${data.educational_note}`;
            document.querySelector('#trainingResults').appendChild(noteDiv);
        }
        
        // Display plots
        if (data.comparison_plot) {
            Plotly.newPlot('comparisonPlot', data.comparison_plot.data, data.comparison_plot.layout);
        }
        
        if (data.prediction_plot) {
            Plotly.newPlot('predictionPlot', data.prediction_plot.data, data.prediction_plot.layout);
        }
        
        // Store trained models
        trainedModels = Object.keys(data.results);
        
        // Update model selector for prediction
        updateModelSelector(trainedModels);
        
        // Update workflow
        updateWorkflowStep(3, 'completed');
        updateWorkflowStep(4, 'active');
        
        // Enable prediction tab
        document.getElementById('prediction-tab').disabled = false;
        
        showAlert('Models trained successfully!', 'success');
    })
    .catch(error => {
        document.getElementById('trainingProgress').style.display = 'none';
        showAlert('Error training models: ' + error, 'danger');
    });
}

function displayTrainingResults(results) {
    const tbody = document.querySelector('#resultsTable tbody');
    tbody.innerHTML = '';
    
    for (const [modelName, result] of Object.entries(results)) {
        if (result.metrics) {
            const row = tbody.insertRow();
            row.insertCell(0).textContent = modelName;
            row.insertCell(1).textContent = result.metrics?.r2?.toFixed(4) || 'N/A';
            row.insertCell(2).textContent = result.metrics?.rmse?.toFixed(4) || 'N/A';
            row.insertCell(3).textContent = result.metrics?.mae?.toFixed(4) || 'N/A';
        }
    }
}

// Prediction
function proceedToPrediction() {
    const predictionTab = new bootstrap.Tab(document.getElementById('prediction-tab'));
    predictionTab.show();
}

function updateModelSelector(models) {
    const select = document.getElementById('modelSelect');
    select.innerHTML = '';
    
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model.replace('_', ' ').charAt(0).toUpperCase() + model.slice(1);
        select.appendChild(option);
    });
}

let predictionFile = null;

function handlePredictionFileSelect(e) {
    predictionFile = e.target.files[0];
}

function makePredictions() {
    if (!sessionId) {
        showAlert('No trained models available.', 'warning');
        return;
    }
    
    if (!predictionFile) {
        showAlert('Please select a file for prediction.', 'warning');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', predictionFile);
    formData.append('session_id', sessionId);
    formData.append('model_name', document.getElementById('modelSelect').value);
    
    showLoading('Making predictions...');
    
    fetch('/api/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.error) {
            showAlert(data.error, 'danger');
            return;
        }
        
        // Display results
        document.getElementById('predictionResults').style.display = 'block';
        
        if (data.plot) {
            Plotly.newPlot('predictionResultsPlot', data.plot.data, data.plot.layout);
        }
        
        // Store predictions for download
        window.currentPredictions = data.predictions;
        
        showAlert('Predictions completed successfully!', 'success');
    })
    .catch(error => {
        hideLoading();
        showAlert('Error making predictions: ' + error, 'danger');
    });
}

// Export functions
function exportResults() {
    if (!sessionId) return;
    
    window.location.href = `/api/export/${sessionId}`;
}

function downloadPredictions() {
    if (!window.currentPredictions) return;
    
    // Create CSV content
    let csv = 'Sample,Predicted Concentration,Lower CI,Upper CI\n';
    window.currentPredictions.forEach(pred => {
        csv += `${pred.sample},${pred.prediction},${pred.confidence_lower},${pred.confidence_upper}\n`;
    });
    
    // Download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'predictions.csv';
    a.click();
}

// Feature importance display with tooltips
function displayFeatureImportanceWithTooltips(topFeatures) {
    // Create or get container for feature importance details
    let container = document.getElementById('featureImportanceDetails');
    if (!container) {
        container = document.createElement('div');
        container.id = 'featureImportanceDetails';
        container.className = 'mt-3';
        const plotDiv = document.getElementById('featureImportancePlot');
        plotDiv.parentNode.insertBefore(container, plotDiv.nextSibling);
    }
    
    // Feature explanations mapping
    const featureExplanations = {
        'peak_height': 'Maximum current in the voltammogram - directly related to concentration',
        'peak_area': 'Total charge transferred - reliable for quantification',
        'peak_potential': 'Voltage at maximum current - characteristic of the analyte',
        'peak_width': 'Width at half maximum - indicates reaction reversibility',
        'mean_current': 'Average current across all voltages',
        'std_current': 'Current variability - indicates signal quality',
        'max_current': 'Absolute maximum current value',
        'min_current': 'Absolute minimum current value',
        'skewness': 'Asymmetry of current distribution',
        'kurtosis': 'Peakedness of current distribution',
        'slope_ascending': 'Rate of current increase',
        'slope_descending': 'Rate of current decrease',
        'baseline_current': 'Background current level',
        'signal_to_noise': 'Ratio of signal to background noise',
        'peak_symmetry': 'Symmetry of the main peak',
        'n_peaks': 'Number of detected peaks'
    };
    
    let html = `
        <div class="card">
            <div class="card-header bg-info text-white">
                <h6 class="mb-0">Top Contributing Features
                    <span class="electroml-tooltip">
                        <span class="tooltip-icon">?</span>
                        <span class="tooltiptext">
                            <div class="tooltip-header">Feature Importance</div>
                            <div>These features have the strongest influence on concentration prediction.</div>
                            <div class="tooltip-emphasis">Focus on optimizing these for better sensor performance!</div>
                        </span>
                    </span>
                </h6>
            </div>
            <div class="card-body">
                <table class="table table-sm">
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Importance</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
    `;
    
    topFeatures.forEach((feature, index) => {
        const importance = (feature.importance * 100).toFixed(1);
        const explanation = featureExplanations[feature.feature] || 'Extracted voltammetric feature';
        const barWidth = feature.importance * 100;
        
        html += `
            <tr>
                <td>
                    <strong>${feature.feature}</strong>
                    ${index < 3 ? '<span class="badge bg-success ms-2">Key</span>' : ''}
                </td>
                <td>
                    <div class="progress" style="height: 20px;">
                        <div class="progress-bar bg-gradient" role="progressbar" 
                             style="width: ${barWidth}%; background: linear-gradient(90deg, #667eea, #764ba2);">
                            ${importance}%
                        </div>
                    </div>
                </td>
                <td class="small text-muted">${explanation}</td>
            </tr>
        `;
    });
    
    html += `
                    </tbody>
                </table>
                <div class="alert alert-info mt-3">
                    <i class="fas fa-lightbulb"></i> 
                    <strong>Tip:</strong> Features with >10% importance are critical for accurate predictions. 
                    Consider these when optimizing your electrochemical sensor design.
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
    
    // Re-initialize tooltips for dynamically added content
    if (window.ElectroMLTooltips) {
        window.ElectroMLTooltips.initialize();
    }
}

// Helper functions
function updateWorkflowStep(step, status) {
    const stepElement = document.getElementById(`step-${step}`);
    if (stepElement) {
        stepElement.className = `step ${status}`;
    }
}

function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3`;
    alertDiv.style.zIndex = 9999;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.body.appendChild(alertDiv);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

function showLoading(message) {
    const loadingDiv = document.createElement('div');
    loadingDiv.id = 'loadingOverlay';
    loadingDiv.className = 'position-fixed top-0 start-0 w-100 h-100 d-flex justify-content-center align-items-center';
    loadingDiv.style.background = 'rgba(0,0,0,0.5)';
    loadingDiv.style.zIndex = 9998;
    loadingDiv.innerHTML = `
        <div class="bg-white p-4 rounded">
            <div class="spinner-border text-primary me-2" role="status"></div>
            <span>${message}</span>
        </div>
    `;
    document.body.appendChild(loadingDiv);
}

function hideLoading() {
    const loadingDiv = document.getElementById('loadingOverlay');
    if (loadingDiv) {
        loadingDiv.remove();
    }
}
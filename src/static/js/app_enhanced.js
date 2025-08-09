// ElectroML Enhanced Application JavaScript

let sessionId = null;
let currentData = null;
let trainedModels = [];
let customHyperparams = {};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    initializeTooltips();
    
    // Set initial CV strategy state (LOO is default, so hide params)
    handleCVStrategyChange();
});

function initializeEventListeners() {
    // File input
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    // Drag and drop
    const dropZone = document.getElementById('dropZone');
    if (dropZone) {
        dropZone.addEventListener('dragover', handleDragOver);
        dropZone.addEventListener('dragleave', handleDragLeave);
        dropZone.addEventListener('drop', handleDrop);
    }
    
    // Prediction mode radio buttons
    document.querySelectorAll('input[name="predMode"]').forEach(radio => {
        radio.addEventListener('change', handlePredictionModeChange);
    });
    
    // CV strategy change
    const cvStrategy = document.getElementById('cvStrategy');
    if (cvStrategy) {
        cvStrategy.addEventListener('change', handleCVStrategyChange);
    }
    
    // Workflow step navigation - make completed steps clickable
    initializeWorkflowNavigation();
}

function initializeWorkflowNavigation() {
    // Make workflow steps clickable when they're completed or active
    const steps = [
        { id: 'step-1', action: () => new bootstrap.Tab(document.getElementById('upload-tab')).show() },
        { id: 'step-2', action: proceedToPreprocessing },
        { id: 'step-3', action: proceedToFeatures },
        { id: 'step-4', action: proceedToTraining },
        { id: 'step-5', action: proceedToPrediction },
        { id: 'step-6', action: () => new bootstrap.Tab(document.getElementById('visualization-tab')).show() }
    ];
    
    steps.forEach(step => {
        const element = document.getElementById(step.id);
        if (element) {
            element.addEventListener('click', function() {
                if (this.classList.contains('completed') || this.classList.contains('active')) {
                    step.action();
                }
            });
        }
    });
}

function initializeTooltips() {
    // Initialize Bootstrap tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
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
    const formData = new FormData();
    formData.append('file', file);
    
    showLoading('Uploading and processing file...');
    
    fetch('/api/data/upload', {
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
        
        sessionId = data.session_id;
        currentData = data;
        
        // Update UI
        document.getElementById('fileName').textContent = data.filename;
        document.getElementById('nSamples').textContent = data.shape[0];
        document.getElementById('nFeatures').textContent = data.shape[1];
        document.getElementById('concRange').textContent = 
            `${data.concentration_range.min.toFixed(2)} - ${data.concentration_range.max.toFixed(2)} μM`;
        
        document.getElementById('dataPreview').style.display = 'block';
        
        // Update session info
        document.getElementById('session-id').textContent = sessionId.substring(0, 8);
        document.getElementById('data-status').textContent = 'Loaded';
        
        // Enable next tab
        document.getElementById('preprocessing-tab').disabled = false;
        
        updateWorkflowStep(1, 'completed');
        showAlert('File uploaded successfully!', 'success');
    })
    .catch(error => {
        hideLoading();
        showAlert('Error uploading file: ' + error, 'danger');
    });
}

// Navigation functions
function proceedToPreprocessing() {
    const preprocessingTab = new bootstrap.Tab(document.getElementById('preprocessing-tab'));
    preprocessingTab.show();
    updateWorkflowStep(2, 'active');
}

function proceedToFeatures() {
    const featuresTab = new bootstrap.Tab(document.getElementById('features-tab'));
    featuresTab.show();
    updateWorkflowStep(3, 'active');
}

function proceedToTraining() {
    const trainingTab = new bootstrap.Tab(document.getElementById('training-tab'));
    trainingTab.show();
    updateWorkflowStep(4, 'active');
}

function proceedToPrediction() {
    console.log('Navigating to prediction tab...');
    const predictionTab = document.getElementById('prediction-tab');
    if (predictionTab) {
        // First ensure the tab is enabled
        predictionTab.disabled = false;
        predictionTab.classList.remove('disabled');
        
        // Then show it
        const bsTab = new bootstrap.Tab(predictionTab);
        bsTab.show();
        updateWorkflowStep(5, 'active');
        
        // Update model selector if we have trained models
        if (trainedModels && trainedModels.length > 0) {
            updateModelSelector(trainedModels);
        }
    } else {
        console.error('Prediction tab not found');
    }
}

function applyPreprocessing() {
    if (!sessionId) {
        showAlert('No data loaded', 'warning');
        return;
    }
    
    const config = {
        session_id: sessionId,
        remove_outliers: document.getElementById('removeOutliers').checked,
        outlier_method: document.getElementById('outlierMethod').value,
        zscore_threshold: parseFloat(document.getElementById('outlierThreshold').value) || 3,
        baseline_correction: document.getElementById('baselineCorrection').checked,
        baseline_method: document.getElementById('baselineMethod').value,
        poly_order: parseInt(document.getElementById('polyOrder').value) || 3,
        apply_filter: document.getElementById('applyFilter').checked,
        filter_type: document.getElementById('filterType').value,
        window_length: parseInt(document.getElementById('filterWindow').value) || 11,
        normalize: document.getElementById('normalize').checked,
        normalization_method: document.getElementById('normMethod').value,
        compute_derivative: document.getElementById('computeDerivative').checked
    };
    
    showLoading('Applying preprocessing...');
    
    fetch('/api/preprocessing/process', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(config)
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.error) {
            showAlert(data.error, 'danger');
            return;
        }
        
        // Show results
        document.getElementById('preprocessingResults').style.display = 'block';
        document.getElementById('preprocessingSteps').innerHTML = 
            data.steps_applied.map(step => `<li>${step}</li>`).join('');
        
        // Enable features tab
        document.getElementById('features-tab').disabled = false;
        
        updateWorkflowStep(2, 'completed');
        updateWorkflowStep(3, 'active');
        showAlert('Preprocessing completed! You can now proceed to feature extraction.', 'success');
        
        // Add educational notes if available
        if (data.educational_notes && data.educational_notes.length > 0) {
            const notesHtml = data.educational_notes.map(note => 
                `<div class="alert alert-info mt-2"><i class="fas fa-info-circle"></i> ${note}</div>`
            ).join('');
            document.getElementById('preprocessingResults').innerHTML += notesHtml;
        }
    })
    .catch(error => {
        hideLoading();
        showAlert('Preprocessing error: ' + error, 'danger');
    });
}

// Training functions
function trainModels() {
    if (!sessionId) {
        showAlert('No data loaded', 'warning');
        return;
    }
    
    // First extract features if not done
    extractFeatures(() => {
        // Then train models
        const selectedModels = [];
        document.querySelectorAll('.model-check:checked').forEach(checkbox => {
            selectedModels.push(checkbox.value);
        });
        
        if (selectedModels.length === 0) {
            showAlert('Please select at least one model', 'warning');
            return;
        }
        
        const config = {
            session_id: sessionId,
            models: selectedModels,
            cv_strategy: document.getElementById('cvStrategy').value,
            n_splits: parseInt(document.getElementById('nSplits').value) || 5,
            test_size: parseFloat(document.getElementById('testSize').value) || 0.2,
            shuffle: document.getElementById('shuffleData').checked,
            random_state: parseInt(document.getElementById('randomSeed').value) || 42,
            hyperparameter_tuning: document.getElementById('hyperparamTuning').checked,
            search_type: document.getElementById('searchType').value,
            custom_params: customHyperparams
        };
        
        document.getElementById('trainingProgress').style.display = 'block';
        document.getElementById('trainingResults').style.display = 'none';
        
        fetch('/api/training/train', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(config)
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
            
            // Display predicted vs actual visualizations if available
            if (data.best_model && data.results[data.best_model]) {
                const bestModelResult = data.results[data.best_model];
                if (bestModelResult.actual_values && bestModelResult.predicted_values) {
                    displayPredictedVsActual(bestModelResult.actual_values, bestModelResult.predicted_values);
                }
            }
            
            // Update model selector
            trainedModels = Object.keys(data.results).filter(k => !data.results[k].error);
            updateModelSelector(trainedModels);
            
            // Update session info
            document.getElementById('models-count').textContent = trainedModels.length;
            
            // Enable prediction and visualization tabs
            document.getElementById('prediction-tab').disabled = false;
            document.getElementById('visualization-tab').disabled = false;
            
            // Update workflow steps
            updateWorkflowStep(4, 'completed');
            updateWorkflowStep(5, 'active');  // Make prediction step active
            updateWorkflowStep(6, 'active');  // Also activate visualization
            
            // Make the prediction step clickable
            const predictionStep = document.getElementById('step-5');
            if (predictionStep) {
                predictionStep.style.cursor = 'pointer';
                predictionStep.onclick = function() {
                    console.log('Prediction step clicked');
                    proceedToPrediction();
                };
                // Add visual feedback
                predictionStep.title = 'Click to go to predictions';
            }
            
            // Also update the tab directly to ensure it's enabled
            const predictionTab = document.getElementById('prediction-tab');
            if (predictionTab) {
                console.log('Enabling prediction tab after training');
                predictionTab.disabled = false;
                predictionTab.classList.remove('disabled');
                predictionTab.removeAttribute('disabled');
                
                // Also add direct click handler to the tab
                predictionTab.onclick = function(e) {
                    if (!this.disabled) {
                        console.log('Prediction tab clicked directly');
                    }
                };
            }
            
            // Enable visualization tab and step
            const vizTab = document.getElementById('visualization-tab');
            if (vizTab) {
                console.log('Enabling visualization tab after training');
                vizTab.disabled = false;
                vizTab.classList.remove('disabled');
                vizTab.removeAttribute('disabled');
                // Force enable by removing all possible disable states
                vizTab.style.pointerEvents = 'auto';
                vizTab.style.opacity = '1';
                console.log('Visualization tab state:', vizTab.disabled, vizTab.hasAttribute('disabled'));
            } else {
                console.error('Visualization tab not found!');
            }
            
            const vizStep = document.getElementById('step-6');
            if (vizStep) {
                vizStep.style.cursor = 'pointer';
                vizStep.onclick = function() {
                    const tab = new bootstrap.Tab(document.getElementById('visualization-tab'));
                    tab.show();
                };
                vizStep.title = 'Click to go to visualizations';
            }
            
            showAlert('Models trained successfully! You can now make predictions and create visualizations.', 'success');
            
            // Double-check tab enabling after a brief delay
            setTimeout(() => {
                const predTab = document.getElementById('prediction-tab');
                const vizTab = document.getElementById('visualization-tab');
                
                if (predTab && predTab.disabled) {
                    console.log('Re-enabling prediction tab');
                    predTab.disabled = false;
                    predTab.removeAttribute('disabled');
                }
                
                if (vizTab && vizTab.disabled) {
                    console.log('Re-enabling visualization tab');
                    vizTab.disabled = false;
                    vizTab.removeAttribute('disabled');
                }
            }, 100);
        })
        .catch(error => {
            document.getElementById('trainingProgress').style.display = 'none';
            showAlert('Training error: ' + error, 'danger');
        });
    });
}

function extractFeatures(callback) {
    if (!sessionId) {
        showAlert('No data loaded. Please upload data first.', 'warning');
        return;
    }
    
    const config = {
        session_id: sessionId,
        smoothing: document.getElementById('featureSmoothing')?.checked ?? true,
        window_length: parseInt(document.getElementById('smoothWindow')?.value) || 11,
        polyorder: parseInt(document.getElementById('polyOrder')?.value) || 3,
        extract_statistical: document.getElementById('extractStatistical')?.checked ?? true,
        extract_peak: document.getElementById('extractPeak')?.checked ?? true,
        extract_frequency: document.getElementById('extractFrequency')?.checked ?? false,
        extract_derivative: document.getElementById('extractDerivative')?.checked ?? false
    };
    
    showLoading('Extracting features from voltammetric data...');
    
    fetch('/api/training/extract_features', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(config)
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.error) {
            showAlert(data.error, 'danger');
            return;
        }
        
        // Display results
        const resultsDiv = document.getElementById('featureResults');
        if (resultsDiv) {
            resultsDiv.style.display = 'block';
            
            // Update stats
            if (document.getElementById('nExtractedFeatures')) {
                document.getElementById('nExtractedFeatures').textContent = data.n_features || '0';
            }
            if (document.getElementById('nExtractedSamples')) {
                document.getElementById('nExtractedSamples').textContent = data.n_samples || '0';
            }
            
            // Display feature importance plot if available
            if (data.importance_plot && window.Plotly) {
                Plotly.newPlot('featureImportancePlot', data.importance_plot.data, data.importance_plot.layout);
            }
            
            // Display top features with tooltips
            if (data.top_features && typeof displayFeatureImportanceWithTooltips === 'function') {
                displayFeatureImportanceWithTooltips(data.top_features);
            }
        }
        
        // Enable training tab
        document.getElementById('training-tab').disabled = false;
        
        updateWorkflowStep(3, 'completed');
        updateWorkflowStep(4, 'active');
        
        showAlert('Features extracted successfully! Ready for model training.', 'success');
        
        // If callback provided (from automatic flow), execute it
        if (callback) callback();
    })
    .catch(error => {
        hideLoading();
        showAlert('Feature extraction error: ' + error, 'danger');
    });
}

function displayPredictedVsActual(actual, predicted) {
    if (!actual || !predicted || actual.length === 0) return;
    
    // Create scatter plot
    const scatterTrace = {
        x: actual,
        y: predicted,
        mode: 'markers',
        type: 'scatter',
        name: 'Predictions',
        marker: {
            size: 10,
            color: 'rgb(33, 150, 243)',
            line: {
                color: 'rgb(0, 0, 0)',
                width: 0.5
            }
        },
        text: actual.map((a, i) => `Actual: ${a.toFixed(3)}<br>Predicted: ${predicted[i].toFixed(3)}`),
        hoverinfo: 'text'
    };
    
    // Add ideal line
    const min = Math.min(...actual, ...predicted);
    const max = Math.max(...actual, ...predicted);
    const idealLine = {
        x: [min, max],
        y: [min, max],
        mode: 'lines',
        type: 'scatter',
        name: 'Ideal',
        line: {
            dash: 'dash',
            color: 'red',
            width: 2
        }
    };
    
    const scatterLayout = {
        title: 'Actual vs Predicted Concentrations',
        xaxis: { title: 'Actual Concentration (μM)' },
        yaxis: { title: 'Predicted Concentration (μM)' },
        showlegend: true,
        hovermode: 'closest'
    };
    
    Plotly.newPlot('actualVsPredictedPlot', [scatterTrace, idealLine], scatterLayout);
    
    // Create residuals plot
    const residuals = actual.map((a, i) => predicted[i] - a);
    const residualsTrace = {
        x: actual,
        y: residuals,
        mode: 'markers',
        type: 'scatter',
        name: 'Residuals',
        marker: {
            size: 8,
            color: residuals.map(r => Math.abs(r)),
            colorscale: 'Viridis',
            showscale: true,
            colorbar: {
                title: 'Absolute Error'
            }
        },
        text: actual.map((a, i) => `Actual: ${a.toFixed(3)}<br>Error: ${residuals[i].toFixed(3)}`),
        hoverinfo: 'text'
    };
    
    // Add zero line
    const zeroLine = {
        x: [min, max],
        y: [0, 0],
        mode: 'lines',
        type: 'scatter',
        name: 'Zero Error',
        line: {
            dash: 'dash',
            color: 'gray',
            width: 1
        }
    };
    
    const residualsLayout = {
        title: 'Residuals Plot',
        xaxis: { title: 'Actual Concentration (μM)' },
        yaxis: { title: 'Residual (Predicted - Actual) (μM)' },
        showlegend: false,
        hovermode: 'closest'
    };
    
    Plotly.newPlot('residualsPlot', [residualsTrace, zeroLine], residualsLayout);
    
    // Populate table
    const tbody = document.querySelector('#predictedActualTable tbody');
    tbody.innerHTML = '';
    
    actual.forEach((actualVal, i) => {
        const predictedVal = predicted[i];
        const error = predictedVal - actualVal;
        const errorPercent = (Math.abs(error) / actualVal * 100);
        
        const row = tbody.insertRow();
        row.insertCell(0).textContent = i + 1;
        row.insertCell(1).textContent = actualVal.toFixed(3);
        row.insertCell(2).textContent = predictedVal.toFixed(3);
        row.insertCell(3).textContent = error.toFixed(3);
        row.insertCell(4).textContent = errorPercent.toFixed(1) + '%';
        
        // Color code based on error
        if (errorPercent < 5) {
            row.classList.add('table-success');
        } else if (errorPercent < 10) {
            row.classList.add('table-warning');
        } else {
            row.classList.add('table-danger');
        }
    });
}

function exportPredictions() {
    const table = document.getElementById('predictedActualTable');
    let csv = 'Sample,Actual (μM),Predicted (μM),Error (μM),Error (%)\n';
    
    const rows = table.querySelectorAll('tbody tr');
    rows.forEach(row => {
        const cells = row.querySelectorAll('td');
        csv += Array.from(cells).map(cell => cell.textContent).join(',') + '\n';
    });
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'predicted_vs_actual.csv';
    a.click();
}

function displayTrainingResults(results) {
    const tbody = document.querySelector('#resultsTable tbody');
    tbody.innerHTML = '';
    
    for (const [modelName, result] of Object.entries(results)) {
        if (!result.error) {
            const row = tbody.insertRow();
            row.insertCell(0).textContent = modelName;
            row.insertCell(1).textContent = result.metrics?.r2?.toFixed(4) || 'N/A';
            row.insertCell(2).textContent = result.metrics?.rmse?.toFixed(4) || 'N/A';
            row.insertCell(3).textContent = result.metrics?.mae?.toFixed(4) || 'N/A';
            row.insertCell(4).textContent = result.cv_score?.toFixed(4) || 'N/A';
            row.insertCell(5).textContent = 'N/A'; // Training time placeholder
        }
    }
}

// Prediction functions
function handlePredictionModeChange(e) {
    document.getElementById('fileUploadSection').style.display = 
        document.getElementById('modeFile').checked ? 'block' : 'none';
    document.getElementById('manualEntrySection').style.display = 
        document.getElementById('modeManual').checked ? 'block' : 'none';
}

function makePredictions() {
    if (!sessionId || trainedModels.length === 0) {
        showAlert('No trained models available', 'warning');
        return;
    }
    
    let mode = 'file';
    if (document.getElementById('modeManual').checked) mode = 'manual';
    if (document.getElementById('modeValidation').checked) mode = 'validation';
    
    const formData = new FormData();
    formData.append('session_id', sessionId);
    formData.append('model_name', document.getElementById('modelSelect').value);
    formData.append('mode', mode);
    
    if (mode === 'file') {
        const fileInput = document.getElementById('predFileInput');
        if (!fileInput.files[0]) {
            showAlert('Please select a file', 'warning');
            return;
        }
        formData.append('file', fileInput.files[0]);
    }
    
    showLoading('Making predictions...');
    
    fetch('/api/prediction/predict', {
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
        
        document.getElementById('predictionResults').style.display = 'block';
        document.getElementById('predictionStats').innerHTML = `
            <p>Predictions: ${data.n_samples} samples</p>
            ${data.metrics ? `
                <p>R²: ${data.metrics.r2.toFixed(4)}</p>
                <p>RMSE: ${data.metrics.rmse.toFixed(4)}</p>
            ` : ''}
        `;
        
        updateWorkflowStep(5, 'completed');
        showAlert('Predictions completed!', 'success');
    })
    .catch(error => {
        hideLoading();
        showAlert('Prediction error: ' + error, 'danger');
    });
}

// Visualization functions
function createPlot() {
    if (!sessionId) {
        showAlert('No data loaded', 'warning');
        return;
    }
    
    const plotType = document.getElementById('plotType').value;
    
    const config = {
        title: document.getElementById('plotTitle').value || `${plotType.charAt(0).toUpperCase() + plotType.slice(1)} Plot`,
        xlabel: document.getElementById('xLabel').value || 'Voltage (V)',
        ylabel: document.getElementById('yLabel').value || 'Current (μA)',
        color_scheme: document.getElementById('colorScheme').value,
        font_family: document.getElementById('fontFamily').value,
        font_size: parseInt(document.getElementById('fontSize').value),
        figure_size: [
            parseInt(document.getElementById('figWidth').value),
            parseInt(document.getElementById('figHeight').value)
        ],
        dpi: parseInt(document.getElementById('dpi').value),
        show_grid: document.getElementById('showGrid').checked,
        show_legend: document.getElementById('showLegend').checked,
        show_trendline: document.getElementById('showTrendline')?.checked || false
    };
    
    showLoading('Creating plot...');
    
    // Prepare the request based on plot type
    let endpoint = '/api/visualization/create_plot';
    let requestData = {
        session_id: sessionId,
        config: config
    };
    
    // Handle different plot types
    if (plotType === 'model_comparison') {
        endpoint = '/api/visualization/model_comparison_plot';
    } else if (plotType === 'feature_importance') {
        endpoint = '/api/visualization/feature_importance_plot';
        requestData.n_features = 20;
    } else if (plotType === 'voltammogram') {
        // For voltammogram, we need to get the actual data from session
        requestData.plot_type = plotType;
        requestData.plot_data = 'from_session'; // Tell backend to use session data
    } else {
        requestData.plot_type = plotType;
        requestData.plot_data = {}; // Backend will handle based on type
    }
    
    fetch(endpoint, {
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
        
        if (data.plot) {
            // Clear previous plot
            const container = document.getElementById('plotContainer');
            container.innerHTML = '';
            
            // Create new plot
            Plotly.newPlot('plotContainer', data.plot.data || data.plot, 
                          data.plot.layout || {}, 
                          {responsive: true});
            
            updateWorkflowStep(6, 'completed');
            showAlert('Plot created successfully!', 'success');
        }
    })
    .catch(error => {
        hideLoading();
        showAlert('Plot creation error: ' + error, 'danger');
    });
}

function exportPlot(format) {
    if (!sessionId) {
        showAlert('No plot to export', 'warning');
        return;
    }
    
    // For now, just download the plot container as image
    if (format === 'png') {
        Plotly.downloadImage('plotContainer', {
            format: 'png',
            width: 1200,
            height: 800,
            filename: 'electroml_plot'
        });
    } else {
        showAlert(`Export to ${format} requires additional setup`, 'info');
    }
}

// Helper functions
function updateWorkflowStep(step, status) {
    const stepElement = document.getElementById(`step-${step}`);
    if (stepElement) {
        stepElement.className = `step ${status}`;
    }
}

// Manual function to enable prediction tab (for testing/debugging)
function enablePredictionTab() {
    console.log('Manually enabling prediction tab...');
    
    // Enable the tab button
    const predictionTab = document.getElementById('prediction-tab');
    if (predictionTab) {
        predictionTab.disabled = false;
        predictionTab.classList.remove('disabled');
        predictionTab.removeAttribute('disabled');
        console.log('Prediction tab enabled');
    }
    
    // Make the workflow step active
    updateWorkflowStep(5, 'active');
    
    // Make the step clickable
    const predictionStep = document.getElementById('step-5');
    if (predictionStep) {
        predictionStep.style.cursor = 'pointer';
        predictionStep.onclick = function() {
            proceedToPrediction();
        };
    }
    
    // If you have dummy models, add them
    if (trainedModels.length === 0) {
        trainedModels = ['linear_regression', 'random_forest']; // Dummy models for testing
        updateModelSelector(trainedModels);
    }
    
    showAlert('Prediction tab manually enabled!', 'info');
}

// Manual function to enable visualization tab (for testing/debugging)
function enableVisualizationTab() {
    console.log('Manually enabling visualization tab...');
    
    // Enable the tab button
    const vizTab = document.getElementById('visualization-tab');
    if (vizTab) {
        vizTab.disabled = false;
        vizTab.classList.remove('disabled');
        vizTab.removeAttribute('disabled');
        vizTab.style.pointerEvents = 'auto';
        vizTab.style.opacity = '1';
        vizTab.style.cursor = 'pointer';
        
        // Add click handler directly
        vizTab.addEventListener('click', function() {
            console.log('Visualization tab clicked');
        });
        
        console.log('Visualization tab enabled:', !vizTab.disabled);
    }
    
    // Make the workflow step active
    updateWorkflowStep(6, 'active');
    
    // Make the step clickable
    const vizStep = document.getElementById('step-6');
    if (vizStep) {
        vizStep.style.cursor = 'pointer';
        vizStep.onclick = function() {
            const tab = new bootstrap.Tab(document.getElementById('visualization-tab'));
            tab.show();
        };
    }
    
    showAlert('Visualization tab manually enabled!', 'info');
}

// Enable all tabs for testing
function enableAllTabs() {
    enablePredictionTab();
    enableVisualizationTab();
    
    // Also enable other tabs
    ['preprocessing-tab', 'features-tab', 'training-tab'].forEach(tabId => {
        const tab = document.getElementById(tabId);
        if (tab) {
            tab.disabled = false;
            tab.classList.remove('disabled');
        }
    });
    
    showAlert('All tabs enabled for testing!', 'success');
}

// Debug function to check tab states
function debugTabs() {
    const tabs = ['upload-tab', 'preprocessing-tab', 'features-tab', 'training-tab', 'prediction-tab', 'visualization-tab'];
    console.log('=== Tab States ===');
    tabs.forEach(tabId => {
        const tab = document.getElementById(tabId);
        if (tab) {
            console.log(`${tabId}:`, {
                disabled: tab.disabled,
                hasDisabledAttr: tab.hasAttribute('disabled'),
                classList: tab.classList.toString(),
                style: {
                    pointerEvents: tab.style.pointerEvents,
                    opacity: tab.style.opacity
                }
            });
        } else {
            console.log(`${tabId}: NOT FOUND`);
        }
    });
    
    console.log('=== Workflow Steps ===');
    for (let i = 1; i <= 6; i++) {
        const step = document.getElementById(`step-${i}`);
        if (step) {
            console.log(`step-${i}:`, step.className);
        }
    }
    
    console.log('=== Session Info ===');
    console.log('Session ID:', sessionId);
    console.log('Current Data:', currentData);
    console.log('Trained Models:', trainedModels);
}

// Test visualization with dummy data
function testVisualization() {
    if (!sessionId) {
        showAlert('Please upload data first', 'warning');
        return;
    }
    
    console.log('Testing visualization with session:', sessionId);
    
    // Try model comparison plot first (if models exist)
    if (trainedModels.length > 0) {
        fetch('/api/visualization/model_comparison_plot', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({session_id: sessionId})
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Model comparison error:', data.error);
                showAlert('Model comparison plot failed: ' + data.error, 'warning');
            } else if (data.plot) {
                Plotly.newPlot('plotContainer', data.plot.data || data.plot, 
                              data.plot.layout || {}, {responsive: true});
                showAlert('Test plot created!', 'success');
            }
        });
    } else {
        showAlert('No trained models. Train models first for visualization.', 'info');
    }
}

function updateModelSelector(models) {
    const select = document.getElementById('modelSelect');
    if (!select) return;
    
    select.innerHTML = '';
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model.replace('_', ' ').charAt(0).toUpperCase() + model.slice(1);
        select.appendChild(option);
    });
}

function handleCVStrategyChange() {
    const strategy = document.getElementById('cvStrategy').value;
    const testSizeInput = document.getElementById('testSize')?.parentElement;
    const nSplitsInput = document.getElementById('nSplits')?.parentElement;
    
    if (strategy === 'train_test_split') {
        if (testSizeInput) testSizeInput.style.display = 'block';
        if (nSplitsInput) nSplitsInput.style.display = 'none';
    } else if (strategy === 'loo') {
        if (testSizeInput) testSizeInput.style.display = 'none';
        if (nSplitsInput) nSplitsInput.style.display = 'none';
    } else {
        if (testSizeInput) testSizeInput.style.display = 'none';
        if (nSplitsInput) nSplitsInput.style.display = 'block';
    }
}

function showHyperparamConfig() {
    // Show modal for hyperparameter configuration
    const modal = new bootstrap.Modal(document.getElementById('hyperparamModal'));
    modal.show();
}

function saveHyperparams() {
    // Save hyperparameter configuration
    const modal = bootstrap.Modal.getInstance(document.getElementById('hyperparamModal'));
    modal.hide();
    showAlert('Hyperparameters saved', 'success');
}

function showHelp() {
    const modal = new bootstrap.Modal(document.getElementById('helpModal'));
    modal.show();
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
    
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

function showLoading(message) {
    const loadingDiv = document.createElement('div');
    loadingDiv.id = 'loadingOverlay';
    loadingDiv.className = 'loading-overlay';
    loadingDiv.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-light loading-spinner" role="status"></div>
            <div class="text-white mt-3">${message}</div>
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

function downloadPredictions() {
    if (!sessionId) return;
    window.location.href = `/api/prediction/download_predictions/${sessionId}`;
}

function detectPeaks() {
    if (!sessionId) {
        showAlert('No data loaded', 'warning');
        return;
    }
    
    fetch('/api/preprocessing/peak_detection', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({session_id: sessionId})
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showAlert(data.error, 'danger');
            return;
        }
        showAlert(`Detected ${data.total_peaks} peaks across all samples`, 'success');
    })
    .catch(error => {
        showAlert('Peak detection error: ' + error, 'danger');
    });
}

function selectVoltageRange() {
    // This would open a modal for voltage range selection
    showAlert('Voltage range selection dialog would appear here', 'info');
}
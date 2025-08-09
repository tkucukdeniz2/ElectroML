/**
 * ElectroML Tooltip System
 * Educational tooltips for chemists learning machine learning
 */

// Tooltip content definitions
const tooltips = {
    // Feature Extraction Tooltips
    features: {
        peakHeight: {
            title: "Peak Height",
            content: "Maximum current value in the voltammogram. In electrochemistry, this represents the peak oxidation or reduction current, directly proportional to analyte concentration according to the Randles-Sevcik equation.",
            formula: "ip = 0.4463 × n × F × A × C × √(n × F × v × D / R × T)",
            example: "Higher peaks typically indicate higher analyte concentration"
        },
        
        peakPotential: {
            title: "Peak Potential (Ep)",
            content: "Voltage at which the maximum current occurs. This is characteristic of the specific electrochemical reaction and can shift with pH, scan rate, or surface modifications.",
            example: "Dopamine typically shows oxidation peak at ~0.2V vs Ag/AgCl"
        },
        
        peakWidth: {
            title: "Peak Width at Half Maximum (FWHM)",
            content: "Width of the peak at 50% of maximum height. Indicates reaction reversibility - narrower peaks suggest more reversible electron transfer.",
            formula: "For reversible reaction: FWHM ≈ 90.6/n mV at 25°C",
            example: "Irreversible reactions show broader peaks"
        },
        
        peakArea: {
            title: "Peak Area (Charge)",
            content: "Integrated area under the peak represents total charge transferred. More reliable than peak height for quantification in some systems.",
            formula: "Q = ∫ i dt = n × F × N",
            example: "Used for surface-confined species quantification"
        },
        
        baseline: {
            title: "Baseline Current",
            content: "Background current from capacitive charging and non-faradaic processes. Must be subtracted for accurate peak measurements.",
            example: "Higher ionic strength increases capacitive current"
        },
        
        symmetry: {
            title: "Peak Symmetry",
            content: "Ratio of peak widths on either side. Asymmetric peaks indicate complex kinetics, adsorption, or multiple electron transfers.",
            example: "Gaussian peaks indicate simple diffusion-controlled process"
        },
        
        roughness: {
            title: "Signal Roughness",
            content: "Measure of noise in the voltammogram. High roughness may indicate electrode fouling, low S/N ratio, or unstable conditions.",
            example: "Clean electrodes show smoother signals"
        },
        
        smoothing: {
            title: "Savitzky-Golay Smoothing",
            content: "Digital filter that preserves peak shape while reducing noise. Uses polynomial fitting in a moving window.",
            example: "Window=11, Order=3 is typical for voltammetry"
        }
    },
    
    // Machine Learning Model Tooltips
    models: {
        linearRegression: {
            title: "Linear Regression",
            content: "Simplest ML model that fits a straight line through data. Assumes linear relationship between features and concentration.",
            formula: "y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ",
            pros: "Fast, interpretable, no hyperparameters",
            cons: "Cannot capture non-linear relationships",
            when: "Use as baseline or when relationship is known to be linear"
        },
        
        ridge: {
            title: "Ridge Regression (L2 Regularization)",
            content: "Linear regression with penalty on large coefficients. Prevents overfitting by shrinking parameters.",
            formula: "Loss = MSE + α × Σ(βᵢ²)",
            pros: "Handles multicollinearity, reduces overfitting",
            cons: "Still assumes linearity",
            when: "Use when features are correlated or dataset is small"
        },
        
        randomForest: {
            title: "Random Forest",
            content: "Ensemble of decision trees that vote on the prediction. Each tree sees random subset of data and features.",
            pros: "Handles non-linearity, robust to outliers, feature importance",
            cons: "Can overfit with small datasets, not interpretable",
            params: "n_estimators=100 (number of trees), max_depth=10",
            when: "Use for complex non-linear relationships"
        },
        
        svm: {
            title: "Support Vector Machine (SVM)",
            content: "Finds optimal hyperplane to separate data. Uses kernel trick to handle non-linear relationships.",
            formula: "K(x,y) = exp(-γ||x-y||²) for RBF kernel",
            pros: "Effective in high dimensions, memory efficient",
            cons: "Sensitive to scaling, slow for large datasets",
            when: "Use for high-dimensional data with clear margin"
        },
        
        gradientBoosting: {
            title: "Gradient Boosting",
            content: "Builds trees sequentially, each correcting errors of previous ones. Powerful but prone to overfitting.",
            pros: "High accuracy, handles non-linearity",
            cons: "Slow training, many hyperparameters, overfitting risk",
            params: "learning_rate=0.1, n_estimators=100, max_depth=3",
            when: "Use when accuracy is critical and dataset is large"
        },
        
        neuralNetwork: {
            title: "Artificial Neural Network",
            content: "Mimics brain structure with interconnected neurons. Can learn any function with enough neurons.",
            formula: "y = f(W₂ × f(W₁ × x + b₁) + b₂)",
            pros: "Universal function approximator",
            cons: "Requires lots of data, black box, many hyperparameters",
            when: "Use for very complex patterns with large datasets"
        }
    },
    
    // Metrics Tooltips
    metrics: {
        r2: {
            title: "R² Score (Coefficient of Determination)",
            content: "Proportion of variance in concentration explained by the model. Ranges from -∞ to 1, where 1 is perfect prediction.",
            formula: "R² = 1 - (SS_res / SS_tot)",
            interpretation: "R² > 0.9: Excellent, 0.7-0.9: Good, 0.5-0.7: Moderate, < 0.5: Poor",
            example: "R²=0.95 means model explains 95% of concentration variance"
        },
        
        rmse: {
            title: "Root Mean Square Error",
            content: "Average prediction error in concentration units (μM). Lower is better. Sensitive to outliers.",
            formula: "RMSE = √(Σ(y_true - y_pred)² / n)",
            interpretation: "Compare to concentration range for context",
            example: "RMSE=0.5 μM for 0-10 μM range is 5% error"
        },
        
        mae: {
            title: "Mean Absolute Error",
            content: "Average absolute prediction error in μM. More robust to outliers than RMSE.",
            formula: "MAE = Σ|y_true - y_pred| / n",
            interpretation: "Easier to interpret than RMSE",
            example: "MAE=0.3 μM means average error of 0.3 μM"
        },
        
        crossValidation: {
            title: "Leave-One-Out Cross-Validation (LOO-CV)",
            content: "Trains model n times, each time leaving out one sample for testing. Maximizes training data for small datasets.",
            pros: "Uses all data, unbiased estimate",
            cons: "Computationally expensive for large datasets",
            when: "Standard for small electrochemical datasets (n < 100)"
        }
    },
    
    // Process Tooltips
    processes: {
        featureImportance: {
            title: "Feature Importance Analysis",
            content: "Identifies which extracted features contribute most to prediction. Uses Random Forest's mean decrease in impurity.",
            interpretation: "Higher importance = stronger influence on concentration prediction",
            use: "Focus on top features for sensor optimization"
        },
        
        normalization: {
            title: "Data Normalization",
            content: "Scales features to [0,1] range. Critical for SVM and neural networks. Prevents features with large values from dominating.",
            formula: "x_norm = (x - x_min) / (x_max - x_min)",
            warning: "Save scaler for consistent prediction scaling!"
        },
        
        hyperparameters: {
            title: "Hyperparameter Optimization",
            content: "Automatically finds best model settings using Bayesian optimization. Balances exploration vs exploitation.",
            process: "Tests different parameter combinations, evaluates with cross-validation",
            time: "Can take 5-30 minutes depending on dataset size"
        },
        
        overfitting: {
            title: "Overfitting Prevention",
            content: "Model memorizes training data instead of learning patterns. Shows high training accuracy but poor test performance.",
            signs: "Large gap between training and validation scores",
            prevention: "Regularization, cross-validation, more data, simpler models"
        }
    }
};

/**
 * Create tooltip HTML element
 * @param {string} category - Tooltip category (features, models, metrics, processes)
 * @param {string} key - Specific tooltip key
 * @param {string} customClass - Additional CSS class
 * @returns {string} HTML string for tooltip
 */
function createTooltip(category, key, customClass = '') {
    const tooltip = tooltips[category]?.[key];
    if (!tooltip) return '';
    
    let contentHtml = `<div class="tooltip-header">${tooltip.title}</div>`;
    contentHtml += `<div>${tooltip.content}</div>`;
    
    if (tooltip.formula) {
        contentHtml += `<div class="tooltip-formula">${tooltip.formula}</div>`;
    }
    
    if (tooltip.pros || tooltip.cons) {
        contentHtml += '<ul class="tooltip-list">';
        if (tooltip.pros) contentHtml += `<li><span class="tooltip-emphasis">Pros:</span> ${tooltip.pros}</li>`;
        if (tooltip.cons) contentHtml += `<li><span class="tooltip-emphasis">Cons:</span> ${tooltip.cons}</li>`;
        contentHtml += '</ul>';
    }
    
    if (tooltip.interpretation) {
        contentHtml += `<div class="mt-2"><span class="tooltip-emphasis">Interpretation:</span> ${tooltip.interpretation}</div>`;
    }
    
    if (tooltip.example) {
        contentHtml += `<div class="tooltip-example">Example: ${tooltip.example}</div>`;
    }
    
    if (tooltip.when) {
        contentHtml += `<div class="mt-2"><span class="tooltip-emphasis">When to use:</span> ${tooltip.when}</div>`;
    }
    
    return `
        <span class="electroml-tooltip ${customClass}">
            <span class="tooltip-icon">?</span>
            <span class="tooltiptext ${customClass}-tooltip">${contentHtml}</span>
        </span>
    `;
}

/**
 * Initialize all tooltips on page
 */
function initializeTooltips() {
    // Add tooltips to feature extraction section
    const featureElements = document.querySelectorAll('[data-tooltip-feature]');
    featureElements.forEach(el => {
        const feature = el.getAttribute('data-tooltip-feature');
        el.innerHTML += createTooltip('features', feature, 'feature');
    });
    
    // Add tooltips to model selection
    const modelElements = document.querySelectorAll('[data-tooltip-model]');
    modelElements.forEach(el => {
        const model = el.getAttribute('data-tooltip-model');
        el.innerHTML += createTooltip('models', model, 'model');
    });
    
    // Add tooltips to metrics
    const metricElements = document.querySelectorAll('[data-tooltip-metric]');
    metricElements.forEach(el => {
        const metric = el.getAttribute('data-tooltip-metric');
        el.innerHTML += createTooltip('metrics', metric, 'metric');
    });
    
    // Add tooltips to processes
    const processElements = document.querySelectorAll('[data-tooltip-process]');
    processElements.forEach(el => {
        const process = el.getAttribute('data-tooltip-process');
        el.innerHTML += createTooltip('processes', process);
    });
}

/**
 * Add inline tooltip to any text
 * @param {string} text - Text to add tooltip to
 * @param {string} category - Tooltip category
 * @param {string} key - Tooltip key
 * @returns {string} Text with tooltip HTML
 */
function addInlineTooltip(text, category, key) {
    return text + createTooltip(category, key);
}

// Initialize tooltips when DOM is ready
document.addEventListener('DOMContentLoaded', initializeTooltips);

// Export for use in other scripts
window.ElectroMLTooltips = {
    create: createTooltip,
    addInline: addInlineTooltip,
    initialize: initializeTooltips
};
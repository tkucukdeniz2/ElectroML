"""
Publication-ready visualization API with comprehensive customization.
"""

import json
import numpy as np
import pandas as pd
from flask import request, jsonify, send_file
import plotly.graph_objs as go
import plotly.utils
import plotly.io as pio
# Optional matplotlib imports for advanced plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
from io import BytesIO
import base64
import logging

from api import visualization_bp
from utils.session_manager import session_manager
from utils.json_helpers import clean_nan_inf

logger = logging.getLogger(__name__)


class PublicationPlotter:
    """Create publication-ready plots with extensive customization."""
    
    COLORBLIND_PALETTES = {
        'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        'colorblind': ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#ECE133',
                       '#56B4E9', '#F0E442', '#D55E00', '#009E73', '#000000'],
        'grayscale': ['#000000', '#333333', '#666666', '#999999', '#CCCCCC']
    }
    
    @staticmethod
    def create_customizable_plot(plot_type, data, config):
        """
        Create a highly customizable plot.
        
        Args:
            plot_type: Type of plot ('voltammogram', 'scatter', 'bar', etc.)
            data: Data for plotting
            config: Plot configuration dictionary
        """
        # Get customization parameters
        title = config.get('title', '')
        xlabel = config.get('xlabel', '')
        ylabel = config.get('ylabel', '')
        font_size = config.get('font_size', 12)
        font_family = config.get('font_family', 'Arial')
        color_scheme = config.get('color_scheme', 'default')
        line_width = config.get('line_width', 2)
        marker_size = config.get('marker_size', 8)
        show_grid = config.get('show_grid', True)
        legend_position = config.get('legend_position', 'top right')
        figure_size = config.get('figure_size', [10, 6])
        dpi = config.get('dpi', 300)
        
        colors = PublicationPlotter.COLORBLIND_PALETTES.get(color_scheme, 
                                                           PublicationPlotter.COLORBLIND_PALETTES['default'])
        
        if plot_type == 'voltammogram':
            return PublicationPlotter._create_voltammogram(data, config, colors)
        elif plot_type == 'scatter':
            return PublicationPlotter._create_scatter(data, config, colors)
        elif plot_type == 'bar':
            return PublicationPlotter._create_bar(data, config, colors)
        elif plot_type == 'heatmap':
            return PublicationPlotter._create_heatmap(data, config)
        elif plot_type == 'multi_panel':
            return PublicationPlotter._create_multi_panel(data, config, colors)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
    
    @staticmethod
    def _create_voltammogram(data, config, colors):
        """Create publication-ready voltammogram."""
        traces = []
        
        for i, (voltage, current, label) in enumerate(data):
            trace = go.Scatter(
                x=voltage,
                y=current,
                mode='lines',
                name=label,
                line=dict(
                    color=colors[i % len(colors)],
                    width=config.get('line_width', 2),
                    dash=config.get('line_styles', [None])[i % len(config.get('line_styles', [None]))]
                )
            )
            traces.append(trace)
        
        layout = PublicationPlotter._get_layout(config)
        fig = go.Figure(data=traces, layout=layout)
        
        return fig
    
    @staticmethod
    def _create_scatter(data, config, colors):
        """Create publication-ready scatter plot."""
        x_data = data.get('x', [])
        y_data = data.get('y', [])
        labels = data.get('labels', [])
        
        trace = go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            text=labels,
            marker=dict(
                size=config.get('marker_size', 8),
                color=colors[0],
                symbol=config.get('marker_symbol', 'circle'),
                line=dict(width=1, color='black')
            )
        )
        
        # Add trendline if requested
        if config.get('show_trendline', False):
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            trendline = go.Scatter(
                x=x_data,
                y=p(x_data),
                mode='lines',
                name='Trendline',
                line=dict(dash='dash', color='red')
            )
            traces = [trace, trendline]
        else:
            traces = [trace]
        
        layout = PublicationPlotter._get_layout(config)
        fig = go.Figure(data=traces, layout=layout)
        
        return fig
    
    @staticmethod
    def _create_bar(data, config, colors):
        """Create publication-ready bar plot."""
        categories = data.get('categories', [])
        values = data.get('values', [])
        errors = data.get('errors', None)
        
        trace = go.Bar(
            x=categories,
            y=values,
            error_y=dict(
                type='data',
                array=errors,
                visible=True
            ) if errors else None,
            marker=dict(
                color=colors[:len(categories)],
                line=dict(color='black', width=1)
            )
        )
        
        layout = PublicationPlotter._get_layout(config)
        fig = go.Figure(data=[trace], layout=layout)
        
        return fig
    
    @staticmethod
    def _create_heatmap(data, config):
        """Create publication-ready heatmap."""
        # Ensure we have data
        z_data = data.get('z', [])
        if not z_data or (isinstance(z_data, list) and len(z_data) == 0):
            # Create dummy data if no data provided
            z_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            data['x'] = ['A', 'B', 'C']
            data['y'] = ['X', 'Y', 'Z']
        
        trace = go.Heatmap(
            z=z_data,
            x=data.get('x', None),
            y=data.get('y', None),
            colorscale=config.get('colorscale', 'Viridis'),
            showscale=True,
            colorbar=dict(
                title=dict(
                    text=config.get('colorbar_title', 'Value'),
                    side='right'
                ),
                tickmode='linear',
                tick0=0,
                dtick=config.get('colorbar_dtick', 1)
            ),
            hovertemplate='%{y}<br>%{x}<br>Value: %{z:.3f}<extra></extra>'
        )
        
        layout = PublicationPlotter._get_layout(config)
        # Adjust layout for heatmap
        layout.update(dict(
            xaxis=dict(
                title=dict(
                    text=config.get('xlabel', 'X Axis'),
                    font=dict(size=config.get('font_size', 12))
                ),
                tickangle=-45 if len(data.get('x', [])) > 10 else 0
            ),
            yaxis=dict(
                title=dict(
                    text=config.get('ylabel', 'Y Axis'),
                    font=dict(size=config.get('font_size', 12))
                )
            )
        ))
        
        fig = go.Figure(data=[trace], layout=layout)
        
        return fig
    
    @staticmethod
    def _create_multi_panel(data, config, colors):
        """Create multi-panel figure for publication."""
        from plotly.subplots import make_subplots
        
        n_panels = len(data)
        rows = config.get('rows', 1)
        cols = config.get('cols', n_panels)
        
        subplot_titles = config.get('subplot_titles', [f'Panel {i+1}' for i in range(n_panels)])
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=config.get('horizontal_spacing', 0.1),
            vertical_spacing=config.get('vertical_spacing', 0.15)
        )
        
        for i, panel_data in enumerate(data):
            row = i // cols + 1
            col = i % cols + 1
            
            trace = go.Scatter(
                x=panel_data['x'],
                y=panel_data['y'],
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=config.get('line_width', 2)),
                showlegend=config.get('show_legend', True),
                name=panel_data.get('name', f'Series {i+1}')
            )
            
            fig.add_trace(trace, row=row, col=col)
        
        fig.update_layout(PublicationPlotter._get_layout(config))
        
        return fig
    
    @staticmethod
    def _get_layout(config):
        """Get standardized layout configuration."""
        return go.Layout(
            title=dict(
                text=config.get('title', ''),
                font=dict(size=config.get('title_size', 16))
            ),
            xaxis=dict(
                title=dict(
                    text=config.get('xlabel', ''),
                    font=dict(size=config.get('font_size', 12))
                ),
                tickfont=dict(size=config.get('tick_size', 10)),
                showgrid=config.get('show_grid', True),
                gridcolor='lightgray',
                zeroline=config.get('show_zeroline', True)
            ),
            yaxis=dict(
                title=dict(
                    text=config.get('ylabel', ''),
                    font=dict(size=config.get('font_size', 12))
                ),
                tickfont=dict(size=config.get('tick_size', 10)),
                showgrid=config.get('show_grid', True),
                gridcolor='lightgray',
                zeroline=config.get('show_zeroline', True)
            ),
            font=dict(
                family=config.get('font_family', 'Arial'),
                size=config.get('font_size', 12)
            ),
            showlegend=config.get('show_legend', True),
            legend=dict(
                x=config.get('legend_x', 0.02),
                y=config.get('legend_y', 0.98),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            ),
            width=config.get('figure_size', [10, 6])[0] * 100,
            height=config.get('figure_size', [10, 6])[1] * 100,
            margin=dict(l=80, r=50, t=80, b=80),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )


@visualization_bp.route('/create_plot', methods=['POST'])
def create_plot():
    """Create a customizable publication-ready plot."""
    try:
        data = request.json
        session_id = data.get('session_id')
        plot_type = data.get('plot_type', 'voltammogram')
        plot_data = data.get('plot_data')
        config = data.get('config', {})
        
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Invalid session'}), 400
        
        # If plot_data is 'from_session', get data from session
        if plot_data == 'from_session' or not plot_data:
            if plot_type == 'voltammogram':
                # Get preprocessed or original data
                df = session.get('preprocessed_data', session.get('data'))
                if df is not None and len(df) > 0:
                    # Create voltammogram data from the dataframe
                    # Assuming first column is concentration, rest are measurements
                    voltages = []
                    for col in df.columns[1:]:
                        try:
                            voltages.append(float(col))
                        except:
                            # Handle string format like 'V_neg1_000'
                            if isinstance(col, str) and col.startswith('V_'):
                                voltage_str = col[2:].replace('neg', '-').replace('_', '.')
                                try:
                                    voltages.append(float(voltage_str))
                                except:
                                    pass
                    
                    # If we couldn't parse voltages, create default range
                    if not voltages:
                        n_points = len(df.columns) - 1
                        voltages = np.linspace(-1, 1, n_points).tolist()
                    
                    # Prepare data for voltammogram (show first few samples)
                    plot_data = []
                    n_samples_to_show = min(5, len(df))
                    for i in range(n_samples_to_show):
                        current = df.iloc[i, 1:].values.tolist()
                        conc = df.iloc[i, 0]
                        plot_data.append((voltages, current, f'Sample {i+1} ({conc:.2f} μM)'))
                else:
                    return jsonify({'error': 'No data available for plotting'}), 400
            elif plot_type == 'scatter':
                # Create scatter plot data from features and predictions
                if 'features' in session and 'models' in session:
                    features_df = session['features']
                    # Use first two features for scatter plot
                    if len(features_df.columns) >= 3:
                        x_col = features_df.columns[1]  # Skip concentration
                        y_col = features_df.columns[2]
                        plot_data = {
                            'x': features_df[x_col].tolist(),
                            'y': features_df[y_col].tolist(),
                            'labels': [f'Sample {i+1}' for i in range(len(features_df))]
                        }
                        config['xlabel'] = config.get('xlabel', x_col)
                        config['ylabel'] = config.get('ylabel', y_col)
                    else:
                        return jsonify({'error': 'Not enough features for scatter plot'}), 400
                else:
                    return jsonify({'error': 'Features not available'}), 400
            elif plot_type == 'bar':
                # Create bar chart from model metrics
                if 'models' in session:
                    models = session['models']
                    model_names = []
                    r2_scores = []
                    for name, model_data in models.items():
                        if 'metrics' in model_data:
                            model_names.append(name)
                            r2_scores.append(model_data['metrics'].get('r2', 0))
                    
                    plot_data = {
                        'categories': model_names,
                        'values': r2_scores
                    }
                    config['ylabel'] = config.get('ylabel', 'R² Score')
                else:
                    return jsonify({'error': 'No models available'}), 400
            elif plot_type == 'heatmap':
                # Create heatmap from data or correlation matrix
                if 'features' in session:
                    features_df = session['features']
                    # Create correlation matrix
                    correlation_matrix = features_df.corr()
                    
                    plot_data = {
                        'z': correlation_matrix.values.tolist(),
                        'x': correlation_matrix.columns.tolist(),
                        'y': correlation_matrix.columns.tolist()
                    }
                    config['colorbar_title'] = 'Correlation'
                    config['title'] = config.get('title', 'Feature Correlation Heatmap')
                elif 'data' in session:
                    # Use raw data for heatmap
                    df = session.get('preprocessed_data', session.get('data'))
                    if df is not None and len(df) > 0:
                        # Create a heatmap of sensor responses
                        # Transpose so samples are rows, voltages are columns
                        data_matrix = df.iloc[:, 1:].values  # Skip concentration column
                        
                        # Limit to first 20 samples for clarity
                        n_samples = min(20, len(data_matrix))
                        data_matrix = data_matrix[:n_samples, :]
                        
                        # Create voltage labels
                        voltages = []
                        for col in df.columns[1:]:
                            try:
                                voltages.append(f'{float(col):.2f}V')
                            except:
                                if isinstance(col, str) and col.startswith('V_'):
                                    voltage_str = col[2:].replace('neg', '-').replace('_', '.')
                                    try:
                                        voltages.append(f'{float(voltage_str):.2f}V')
                                    except:
                                        voltages.append(col)
                                else:
                                    voltages.append(str(col))
                        
                        plot_data = {
                            'z': data_matrix.tolist(),
                            'x': voltages[::10] if len(voltages) > 50 else voltages,  # Reduce x-labels if too many
                            'y': [f'Sample {i+1}' for i in range(n_samples)]
                        }
                        config['colorbar_title'] = 'Current (μA)'
                        config['title'] = config.get('title', 'Sensor Response Heatmap')
                        config['xlabel'] = config.get('xlabel', 'Voltage')
                        config['ylabel'] = config.get('ylabel', 'Sample')
                    else:
                        return jsonify({'error': 'No data available for heatmap'}), 400
                else:
                    return jsonify({'error': 'No data available for heatmap'}), 400
            else:
                # For other plot types, create dummy data
                plot_data = {}
        
        plotter = PublicationPlotter()
        fig = plotter.create_customizable_plot(plot_type, plot_data, config)
        
        # Convert to JSON for frontend
        plot_json = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
        
        response = {
            'session_id': session_id,
            'plot': plot_json,
            'plot_type': plot_type
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Plot creation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@visualization_bp.route('/export_plot', methods=['POST'])
def export_plot():
    """Export plot in various formats."""
    try:
        data = request.json
        session_id = data.get('session_id')
        plot_data = data.get('plot_data')
        config = data.get('config', {})
        export_format = data.get('format', 'png')  # png, svg, pdf, html
        
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Invalid session'}), 400
        
        plotter = PublicationPlotter()
        fig = plotter.create_customizable_plot(
            plot_data.get('type', 'voltammogram'),
            plot_data.get('data'),
            config
        )
        
        if export_format == 'png':
            try:
                img_bytes = pio.to_image(fig, format='png', width=config.get('width', 1200), 
                                        height=config.get('height', 800), scale=config.get('dpi', 300)/72)
                
                return send_file(
                    BytesIO(img_bytes),
                    mimetype='image/png',
                    as_attachment=True,
                    download_name=f'plot_{session_id[:8]}.png'
                )
            except Exception as e:
                logger.warning(f"PNG export failed (install kaleido for PNG/PDF export): {e}")
                # Fallback to HTML
                export_format = 'html'
            
        elif export_format == 'svg':
            try:
                svg_str = pio.to_image(fig, format='svg')
                
                return send_file(
                    BytesIO(svg_str),
                    mimetype='image/svg+xml',
                    as_attachment=True,
                    download_name=f'plot_{session_id[:8]}.svg'
                )
            except Exception as e:
                logger.warning(f"SVG export failed: {e}")
                export_format = 'html'
            
        elif export_format == 'pdf':
            try:
                pdf_bytes = pio.to_image(fig, format='pdf', width=config.get('width', 1200), 
                                        height=config.get('height', 800))
                
                return send_file(
                    BytesIO(pdf_bytes),
                    mimetype='application/pdf',
                    as_attachment=True,
                    download_name=f'plot_{session_id[:8]}.pdf'
                )
            except Exception as e:
                logger.warning(f"PDF export failed (install kaleido): {e}")
                export_format = 'html'
            
        elif export_format == 'html':
            html_str = pio.to_html(fig, include_plotlyjs='cdn')
            
            return send_file(
                BytesIO(html_str.encode()),
                mimetype='text/html',
                as_attachment=True,
                download_name=f'plot_{session_id[:8]}.html'
            )
        
        else:
            return jsonify({'error': f'Unsupported format: {export_format}'}), 400
            
    except Exception as e:
        logger.error(f"Plot export error: {e}")
        return jsonify({'error': str(e)}), 500


@visualization_bp.route('/model_comparison_plot', methods=['POST'])
def create_model_comparison_plot():
    """Create comprehensive model comparison visualizations."""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        session = session_manager.get_session(session_id)
        if not session or 'models' not in session:
            return jsonify({'error': 'No models available'}), 400
        
        models = session['models']
        
        # Prepare comparison data
        model_names = []
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        
        for name, model_data in models.items():
            if 'metrics' in model_data:
                model_names.append(name)
                r2_scores.append(model_data['metrics'].get('r2', 0))
                rmse_scores.append(model_data['metrics'].get('rmse', 0))
                mae_scores.append(model_data['metrics'].get('mae', 0))
        
        # Create subplots
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('R² Score', 'RMSE', 'MAE', 'Prediction vs Actual'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # R² Score
        fig.add_trace(
            go.Bar(x=model_names, y=r2_scores, name='R²', marker_color='blue'),
            row=1, col=1
        )
        
        # RMSE
        fig.add_trace(
            go.Bar(x=model_names, y=rmse_scores, name='RMSE', marker_color='red'),
            row=1, col=2
        )
        
        # MAE
        fig.add_trace(
            go.Bar(x=model_names, y=mae_scores, name='MAE', marker_color='green'),
            row=2, col=1
        )
        
        # Best model predictions vs actual
        if model_names:
            best_model = model_names[np.argmax(r2_scores)]
            if 'predictions' in models[best_model]:
                predictions = models[best_model]['predictions']
                # Assuming we have actual values stored
                actual = session.get('features', pd.DataFrame()).get('Concentration', [])
                
                if len(actual) > 0:
                    fig.add_trace(
                        go.Scatter(x=actual, y=predictions, mode='markers', 
                                 name=f'{best_model} predictions'),
                        row=2, col=2
                    )
                    
                    # Add perfect prediction line
                    min_val = min(min(actual), min(predictions))
                    max_val = max(max(actual), max(predictions))
                    fig.add_trace(
                        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                 mode='lines', line=dict(dash='dash'),
                                 name='Perfect prediction'),
                        row=2, col=2
                    )
        
        fig.update_layout(
            title='Model Performance Comparison',
            showlegend=False,
            height=800,
            font=dict(size=12)
        )
        
        plot_json = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
        
        return jsonify({'plot': plot_json}), 200
        
    except Exception as e:
        logger.error(f"Model comparison plot error: {e}")
        return jsonify({'error': str(e)}), 500


@visualization_bp.route('/feature_importance_plot', methods=['POST'])
def create_feature_importance_plot():
    """Create feature importance visualization."""
    try:
        data = request.json
        session_id = data.get('session_id')
        n_features = data.get('n_features', 20)
        
        session = session_manager.get_session(session_id)
        if not session or 'features' not in session:
            return jsonify({'error': 'Features not available'}), 400
        
        features_df = session['features']
        X = features_df.drop(columns=['Concentration'])
        y = features_df['Concentration']
        
        # Calculate feature importance
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=True).tail(n_features)
        
        fig = go.Figure(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker=dict(color='steelblue')
        ))
        
        fig.update_layout(
            title=f'Top {n_features} Most Important Features',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=max(400, n_features * 20),
            margin=dict(l=150)
        )
        
        plot_json = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
        
        return jsonify({'plot': plot_json}), 200
        
    except Exception as e:
        logger.error(f"Feature importance plot error: {e}")
        return jsonify({'error': str(e)}), 500
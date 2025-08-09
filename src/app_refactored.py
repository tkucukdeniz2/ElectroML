#!/usr/bin/env python3
"""
ElectroML - Refactored modular Flask application
Main application entry point with blueprint registration
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template
from flask_cors import CORS
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(config_name='development'):
    """
    Application factory pattern for creating Flask app.
    
    Args:
        config_name: Configuration environment name
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'electroml-secret-key-change-in-production')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
    app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    register_blueprints(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Main route
    @app.route('/')
    def index():
        """Render main application page."""
        return render_template('index_enhanced.html')
    
    logger.info(f"ElectroML application created with config: {config_name}")
    
    return app


def register_blueprints(app):
    """Register all application blueprints."""
    from api import (
        data_bp, preprocessing_bp, training_bp, 
        prediction_bp, visualization_bp
    )
    
    app.register_blueprint(data_bp)
    app.register_blueprint(preprocessing_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(visualization_bp)
    
    logger.info("All blueprints registered successfully")


def register_error_handlers(app):
    """Register error handlers for the application."""
    
    @app.errorhandler(404)
    def not_found_error(error):
        return {'error': 'Resource not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return {'error': 'Internal server error'}, 500
    
    @app.errorhandler(413)
    def too_large_error(error):
        return {'error': 'File too large. Maximum size is 16MB'}, 413


# Create application instance
app = create_app()


if __name__ == '__main__':
    # Run in development mode
    app.run(host='127.0.0.1', port=5005, debug=True)
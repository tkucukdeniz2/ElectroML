"""
API modules for ElectroML Flask application.
"""

from flask import Blueprint

# Create blueprints for different API sections
data_bp = Blueprint('data', __name__, url_prefix='/api/data')
preprocessing_bp = Blueprint('preprocessing', __name__, url_prefix='/api/preprocessing')
training_bp = Blueprint('training', __name__, url_prefix='/api/training')
prediction_bp = Blueprint('prediction', __name__, url_prefix='/api/prediction')
visualization_bp = Blueprint('visualization', __name__, url_prefix='/api/visualization')

# Import routes to register them with blueprints
from . import data_handler
from . import preprocessing
from . import training
from . import prediction
from . import visualization
from config import Config
from flask import Flask
from app.routes.main_routes import main_bp as main
from app.routes.api_routes import api_bp

def create_app():
    """
    Initializes and configures the Flask application.

    :return: The configured Flask application instance.
    """
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize the loading status
    app.config['loading_dataframe_status'] = {"complete": False}

    # Register blueprints
    app.register_blueprint(main)
    app.register_blueprint(api_bp)

    return app

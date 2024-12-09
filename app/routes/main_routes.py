from flask import Blueprint, render_template, current_app

main_bp = Blueprint('main', __name__)

@main_bp.route('/', methods=['GET'])
def index():

    # Render the form template without prediction logic
    return render_template('index.html')
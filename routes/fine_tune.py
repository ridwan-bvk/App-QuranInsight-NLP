from flask import Blueprint, render_template

fine_tune_bp = Blueprint('fine_tune', __name__, url_prefix='/fine_tune')

@fine_tune_bp.route('/')
def fine_tune_home():
    return render_template('placeholder.html', title='fintune')

from flask import Blueprint, render_template

transformers_bp = Blueprint('transformers', __name__, url_prefix='/transformers')

@transformers_bp.route('/')
def transformers_home():
    return render_template('placeholder.html', title='Transformer Models')

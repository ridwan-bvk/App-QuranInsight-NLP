from flask import Blueprint, render_template

ollama_bp = Blueprint('ollama', __name__, url_prefix='/ollama')

@ollama_bp.route('/')
def ollama_home():
    return render_template('placeholder.html', title=' ollama')

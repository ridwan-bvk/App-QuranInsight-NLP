from flask import Blueprint, render_template

rag_bp = Blueprint('rag', __name__, url_prefix='/rag')

@rag_bp.route('/')
def rag_home():
    return render_template('placeholder.html', title=' rag')

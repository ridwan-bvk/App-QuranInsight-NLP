from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "rahasia"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['dataset']
    df = pd.read_csv(file)
    session['data'] = df.to_dict(orient='records')  # Simpan dataset ke session
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'data' not in session:
        return redirect(url_for('index'))
    return render_template('dashboard.html')

@app.route('/vector', methods=['GET', 'POST'])
def vector():
    from utils.vector_model import build_model, get_similar_words
    if 'data' not in session:
        return redirect(url_for('index'))
    
    dataset = [row['text'] for row in session['data']]
    model = build_model(dataset)
    
    result = []
    if request.method == 'POST':
        word = request.form['word']
        result = get_similar_words(model, word)
    return render_template('vector.html', result=result)

@app.route('/seq2seq')
def seq2seq():
    return render_template('placeholder.html', title='Sequence to Sequence')

@app.route('/transformers')
def transformers():
    return render_template('placeholder.html', title='Transformers')

@app.route('/fine_tune')
def fine_tune():
    return render_template('placeholder.html', title='Fine Tuning LLM')

@app.route('/rag')
def rag():
    return render_template('placeholder.html', title='RAG (Retrieval Augmented Generation)')

@app.route('/ollama')
def ollama():
    return render_template('placeholder.html', title='OLLAMA API QA')



if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    app.run(debug=False)

# TODO: Tambahkan route seq2seq, rag, etc.

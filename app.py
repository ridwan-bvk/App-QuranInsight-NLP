from flask import Flask, render_template, request, redirect, flash, url_for, send_file, session
from werkzeug.utils import secure_filename
import csv
import os
from routes.vector import vector_bp
from routes.seq2seq import seq2seq_bp
from routes.transformers import transformers_bp
from routes.fine_tune import fine_tune_bp
from routes.rag import rag_bp
from routes.ollama import ollama_bp

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key'

@app.route('/upload', methods=['POST'])
def upload():
    if 'dataset' not in request.files:
        flash('⚠️ Tidak ada file yang dipilih', 'danger')
        return redirect(url_for('index'))

    file = request.files['dataset']
    if file.filename == '':
        flash('⚠️ Nama file kosong', 'danger')
        return redirect(url_for('index'))

    if not file.filename.lower().endswith('.csv'):
        flash('❌ File harus berformat CSV', 'danger')
        return render_template('dashboard.html', show_template=True)

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Simpan file dulu
        file.save(filepath)
        
        # Baca dengan mode binary dan decode manual
        with open(filepath, 'rb') as f:
            content = f.read().decode('utf-8-sig')  # Handle BOM
            
        # Bersihkan karakter khusus
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Parse CSV dengan quote non-standar
        reader = csv.DictReader(content.splitlines(), quotechar='"', escapechar='\\')
        
        # Validasi
        required_columns = {'Surat', 'Ayat', 'Teks'}
        if not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"Kolom wajib tidak ditemukan: {required_columns}")
        
        rows = list(reader)
        for i, row in enumerate(rows[:5]):  # Cek 5 baris pertama
            if not all(row.get(col, '').strip() for col in required_columns):
                raise ValueError(f"Data kosong pada baris {i+2}")
        
        # Jika berhasil
        session['sample_data'] = rows[:3]
        session['uploaded_file'] = filename
        flash('✅ File CSV berhasil diproses!, silahkan pilih model yang akan diproses', 'success')
        return redirect(url_for('index'))

    except Exception as e:
        print("Error detail:", str(e))
        flash(f'❌ Kesalahan format CSV: {str(e)}. Gunakan format contoh', 'danger')
        return render_template('dashboard.html', show_template=True)
@app.route('/download-template')
def download_template():
    return send_file('static/template.csv', as_attachment=True)

# Register Blueprints
app.register_blueprint(vector_bp, url_prefix='/vector') 
app.register_blueprint(seq2seq_bp)
app.register_blueprint(transformers_bp)
app.register_blueprint(fine_tune_bp)
app.register_blueprint(rag_bp)
app.register_blueprint(ollama_bp)

@app.route('/')
def index():
    if 'uploaded_file' in session:
        flash('Data CSV masih tersedia untuk analisis', 'info')
    return render_template('dashboard.html')

@app.route('/clear_session')
def clear_session():
    # Hapus session hanya jika diminta
    session.pop('uploaded_file', None)
    session.pop('sample_data', None)
    flash('Data CSV telah direset', 'warning')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
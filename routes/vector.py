from flask import Blueprint, render_template, session, current_app
from gensim.models import Word2Vec
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Penting untuk prevent GUI error
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
from sklearn.decomposition import PCA
from utils.vector_model import preprocess_text

vector_bp = Blueprint('vector', __name__)

@vector_bp.route('/analysis')
def vector_analysis():
    try:
        # 1. Get uploaded file
        uploaded_file = session.get('uploaded_file')
        if not uploaded_file:
            return redirect(url_for('index'))
        
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], uploaded_file)
        
        # 2. Load and preprocess data
        df = pd.read_csv(filepath)
        print("Data loaded:", df.head())  # Debug
        
        # Pastikan kolom Teks ada
        if 'Teks' not in df.columns:
            raise ValueError("Kolom 'Teks' tidak ditemukan")
            
        sentences = df['Teks'].apply(preprocess_text).tolist()
        print("Contoh kalimat terproses:", sentences[:2])  # Debug
        
        # 3. Train model
        model = Word2Vec(
            sentences=sentences,
            vector_size=100,
            window=5,
            min_count=1,  # Ubah ke 1 untuk dataset kecil
            workers=4
        )
        
        # 4. Prepare visualization
        words = ['allah', 'maha', 'pengasih', 'penyayang', 'jalan', 'nikmat']
        valid_words = [w for w in words if w in model.wv]
        
        if not valid_words:
            return render_template('vector.html', 
                                error="Kata kunci tidak ditemukan dalam vocabulary")
        
        # 5. Create plot
        plt.switch_backend('Agg')  # Pastikan menggunakan backend non-GUI
        word_vectors = np.array([model.wv[word] for word in valid_words])
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(word_vectors)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
        for i, word in enumerate(valid_words):
            plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]), 
                        xytext=(5, 2), textcoords='offset points')
        
        # Convert plot to image
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()
        
        # 6. Get similar words
        similar_words = {}
        for word in valid_words[:3]:
            try:
                similar_words[word] = model.wv.most_similar(word, topn=3)
            except KeyError:
                continue
        
        return render_template('vector.html',
                            plot_url=plot_url,
                            similar_words=similar_words,
                            sample_data=df.head(3).to_dict('records'))
        
    except Exception as e:
        print("Error in vector analysis:", str(e))
        return render_template('vector.html', 
                            error=f"Gagal memproses: {str(e)}")
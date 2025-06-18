# utils/vector_model.py
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

def build_model(texts):
    """
    Membangun model Word2Vec dari list teks ayat Al-Qur’an.
    :param texts: List[str] - kumpulan terjemahan ayat
    :return: Word2Vec model
    """
    corpus = [simple_preprocess(text) for text in texts if isinstance(text, str)]
    model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=2)
    return model

def get_similar_words(model, word, topn=10):
    """
    Mencari kata-kata mirip berdasarkan word vector.
    :param model: Word2Vec model
    :param word: Kata untuk dicari kemiripannya
    :param topn: Jumlah hasil teratas
    :return: List of (word, similarity)
    """
    try:
        return model.wv.most_similar(word, topn=topn)
    except KeyError:
        return [("Kata tidak ditemukan dalam model.", 0.0)]

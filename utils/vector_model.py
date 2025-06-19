import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

def preprocess_text(text):
    # Handle missing values
    if not isinstance(text, str):
        return []
    
    # Clean and tokenize
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    
    # Custom stopwords for Quran translation
    stopwords = {
        'yang', 'lagi', 'pada', 'bagi', 'dengan', 'ini', 'itu', 
        'dan', 'di', 'ke', 'kami', 'engkau', 'telah', 'bukan'
    }
    
    return [word for word in tokens if word not in stopwords and len(word) > 2]
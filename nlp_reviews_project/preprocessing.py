"""preprocessing.py
Text cleaning and spaCy pipeline helpers.
"""
import re
import spacy
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load spaCy model lazily
_nlp = None

def get_nlp(model='en_core_web_sm'):
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(model, disable=['ner'])
    return _nlp


def clean_text(text: str) -> str:
    """Basic text cleaning: remove URLs, normalize whitespace, strip emojis, lower-case."""
    if text is None:
        return ''
    s = str(text)
    # Remove URLs
    s = re.sub(r'http\S+|www\.\S+', ' ', s)
    # Remove emojis and non-printable
    s = s.encode('ascii', 'ignore').decode('ascii')
    # Remove extra spaces and newlines
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def tokenize_lemmatize(text: str, remove_stopwords=True) -> List[str]:
    nlp = get_nlp()
    doc = nlp(text)
    tokens = []
    stops = set(stopwords.words('english'))
    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        lemma = tok.lemma_.lower().strip()
        if remove_stopwords and lemma in stops:
            continue
        if lemma == '':
            continue
        tokens.append(lemma)
    return tokens


def pos_tags(text: str) -> List[Tuple[str, str]]:
    nlp = get_nlp()
    doc = nlp(text)
    return [(tok.text, tok.pos_) for tok in doc]

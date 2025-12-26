"""preprocessing.py
Text cleaning and NLTK pipeline helpers.
"""
import re
from typing import List, Tuple
import nltk
import ssl
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def download_nltk_resources():
    """Download required NLTK resources if not already present."""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    resources = {
        "tokenizers/punkt": "punkt",
        "corpora/wordnet": "wordnet",
        "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
        "corpora/stopwords": "stopwords"
    }
    for resource_path, resource_id in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_id)

download_nltk_resources()

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


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
    return s.lower()


def get_wordnet_pos(treebank_tag):
    """Convert treebank POS tags to WordNet POS tags."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun


def tokenize_lemmatize(text: str, remove_stopwords=True) -> List[str]:
    """Tokenize and lemmatize text using NLTK."""
    tokens = word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    lemmas = []
    for i in range(len(tokens)):
        # Basic filter for punctuation and short tokens
        if not tokens[i].isalpha() or len(tokens[i]) < 2:
            continue
        # Stopword filter
        if remove_stopwords and tokens[i].lower() in stop_words:
            continue
        # Lemmatize with POS context
        lemma = lemmatizer.lemmatize(tokens[i], pos=get_wordnet_pos(pos[i][1]))
        lemmas.append(lemma.lower())
    return lemmas


def pos_tags(text: str) -> List[Tuple[str, str]]:
    """Get POS tags for tokens in text."""
    tokens = word_tokenize(text)
    return nltk.pos_tag(tokens)

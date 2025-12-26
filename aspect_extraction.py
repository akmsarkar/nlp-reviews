"""aspect_extraction.py
Simple aspect extraction using noun-phrases and n-grams.
"""
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nltk
import ssl
from nltk.tokenize import word_tokenize
from nltk.chunk import RegexpParser

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
        "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger"
    }
    for resource_path, resource_id in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_id)

download_nltk_resources()

# Define a simple grammar for noun phrases
NP_GRAMMAR = r"""
    NP: {<DT|JJ|NN.*>+}   # Chunk sequences of DT, JJ, NN
"""
chunker = RegexpParser(NP_GRAMMAR)


def top_noun_phrases(series, top_n=30):
    phrases = []
    for doc in series.fillna(''):
        tokens = word_tokenize(doc)
        pos_tags = nltk.pos_tag(tokens)
        tree = chunker.parse(pos_tags)
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            phrase = " ".join(word for word, tag in subtree.leaves())
            if len(phrase.split()) <= 4:
                phrases.append(phrase.lower().strip())
    c = pd.Series(phrases).value_counts().head(top_n)
    return c


def top_ngrams(series, ngram_range=(1,2), top_n=30):
    cv = CountVectorizer(ngram_range=ngram_range, stop_words='english', min_df=2)
    X = cv.fit_transform(series.fillna(''))
    s = X.sum(axis=0)
    freqs = [(word, s[0, idx]) for word, idx in cv.vocabulary_.items()]
    freqs = sorted(freqs, key=lambda x: -x[1])
    return freqs[:top_n]


if __name__ == '__main__':
    # Quick CLI demo
    import argparse
    from data_loader import load_and_clean
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()
    df = load_and_clean(args.input)
    print('Top noun phrases:')
    print(top_noun_phrases(df['review']).head(20))

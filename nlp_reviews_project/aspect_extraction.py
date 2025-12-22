"""aspect_extraction.py
Simple aspect extraction using noun-phrases and n-grams.
"""
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from preprocessing import get_nlp, clean_text


def top_noun_phrases(series, top_n=30):
    nlp = get_nlp()
    phrases = []
    for doc in series.fillna(''):
        sp = nlp(doc)
        phrases.extend([chunk.text.lower().strip() for chunk in sp.noun_chunks if len(chunk.text.split())<=4])
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

"""features.py
Compute linguistic features for each review using NLTK/VADER.
"""
import pandas as pd
import numpy as np
from collections import Counter
from preprocessing import tokenize_lemmatize, clean_text, pos_tags
import textstat
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pathlib import Path

# Define the base directory of the project
BASE_DIR = Path(__file__).resolve().parent

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()


def lexical_diversity(tokens):
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def repetition_ratio(tokens):
    if not tokens:
        return 0.0
    return 1.0 - (len(set(tokens)) / len(tokens))


def pos_ratios(text):
    tags = pos_tags(text)
    counts = Counter([tag for _, tag in tags])
    total = sum(counts.values()) or 1
    return {
        'adj_ratio': counts.get('JJ', 0) / total,
        'noun_ratio': counts.get('NN', 0) / total,
        'verb_ratio': counts.get('VB', 0) / total,
    }


def superlative_count(text):
    tags = pos_tags(text)
    return sum(1 for _, tag in tags if tag in ('JJS', 'RBS'))


def exclamation_count(text):
    return str(text).count('!')


def sentiment_scores(text):
    return sia.polarity_scores(str(text))


def apply_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Basic counts
    df['char_count'] = df['review'].astype(str).str.len()
    df['word_count'] = df['review'].astype(str).str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Cleaned tokens
    df['clean_review'] = df['review'].astype(str).apply(clean_text)
    df['tokens'] = df['clean_review'].apply(tokenize_lemmatize)

    df['lexical_diversity'] = df['tokens'].apply(lexical_diversity)
    df['repetition_ratio'] = df['tokens'].apply(repetition_ratio)

    # POS ratios and counts
    pos = df['clean_review'].apply(pos_ratios).apply(pd.Series)
    df = pd.concat([df, pos], axis=1)

    df['superlative_count'] = df['clean_review'].apply(superlative_count)
    df['exclamation_count'] = df['review'].apply(exclamation_count)

    # Readability
    df['flesch_reading_ease'] = df['clean_review'].apply(lambda x: textstat.flesch_reading_ease(x) if x and len(x.split())>3 else np.nan)

    # Sentiment
    senti = df['clean_review'].apply(sentiment_scores)
    senti_df = pd.json_normalize(senti)
    df = pd.concat([df, senti_df], axis=1)

    return df


if __name__ == '__main__':
    import argparse
    from data_loader import load_and_clean
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default=BASE_DIR / 'data' / 'feature_reviews.csv')
    args = parser.parse_args()

    df = load_and_clean(args.input)
    feat = apply_features(df)
    feat.to_csv(args.output, index=False)
    print('Saved features to', args.output)

"""bertopic_aspects.py
Fit BERTopic to the reviews and extract aspect topics.
Outputs:
 - {output_prefix}_topic_info.csv  : topic_id, name, n_docs, top_words, avg_compound_sentiment
 - {output_prefix}_assignments.csv: review-level topic assignments
 - {output_prefix}_model.pkl      : saved BERTopic model (optional)

Usage:
 python bertopic_aspects.py --input nlp_reviews_project/data/feature_reviews.csv --output-prefix nlp_reviews_project/data/bertopic_output
"""
import argparse
import os
import pandas as pd
from bertopic import BERTopic
from umap import UMAP
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
from tqdm import tqdm


def load_docs(df, text_col='clean_review'):
    if text_col in df.columns:
        docs = df[text_col].fillna('').astype(str).tolist()
    elif 'review' in df.columns:
        docs = df['review'].fillna('').astype(str).tolist()
    else:
        raise ValueError('No suitable text column found')
    return docs


def run_bertopic(input_csv, output_prefix, embedding_model='all-MiniLM-L6-v2', top_n_words=10, n_neighbors=10, n_components=5, min_topic_size=10):
    df = pd.read_csv(input_csv, parse_dates=['time'], low_memory=False)
    docs = load_docs(df)

    # Compute embeddings
    embedder = SentenceTransformer(embedding_model)
    embeddings = embedder.encode(docs, show_progress_bar=True)

    # Fit BERTopic with faster UMAP settings and a minimum topic size
    umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine', random_state=42)
    topic_model = BERTopic(umap_model=umap_model, min_topic_size=min_topic_size, verbose=True)
    topics, probs = topic_model.fit_transform(docs, embeddings)

    # Attach topics to dataframe
    df['topic'] = topics
    # Robustly handle different shapes/types returned in `probs`
    topic_probs = []
    if probs is None:
        topic_probs = [np.nan] * len(docs)
    elif isinstance(probs, np.ndarray):
        if probs.ndim == 2:
            for row in probs:
                try:
                    topic_probs.append(float(row.max()))
                except Exception:
                    topic_probs.append(np.nan)
        else:
            # 1D array
            topic_probs = [float(x) if not np.isnan(x) else np.nan for x in probs]
    elif isinstance(probs, list):
        for p in probs:
            try:
                if hasattr(p, '__len__'):
                    topic_probs.append(float(np.max(p)))
                else:
                    topic_probs.append(float(p))
            except Exception:
                topic_probs.append(np.nan)
    else:
        topic_probs = [np.nan] * len(docs)

    # Ensure length matches
    if len(topic_probs) != len(docs):
        topic_probs = topic_probs[:len(docs)] + [np.nan] * max(0, len(docs) - len(topic_probs))

    df['topic_prob'] = topic_probs

    # Topic info
    topics_info = []
    topic_freq = topic_model.get_topic_info()
    for t in topic_freq['Topic'].unique():
        if t == -1:
            # outliers
            continue
        words = topic_model.get_topic(t)
        top_words = ", ".join([w for w, _ in words[:top_n_words]])
        # documents in topic
        mask = df['topic'] == t
        n_docs = int(mask.sum())
        # average compound sentiment if available
        avg_comp = df.loc[mask, 'compound'].dropna()
        avg_comp = float(avg_comp.mean()) if len(avg_comp)>0 else np.nan
        topics_info.append({'topic': int(t), 'n_docs': n_docs, 'top_words': top_words, 'avg_compound': avg_comp})

    topics_df = pd.DataFrame(topics_info).sort_values('n_docs', ascending=False).reset_index(drop=True)

    # Outputs
    out_dir = os.path.dirname(output_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    topics_df.to_csv(f"{output_prefix}_topic_info.csv", index=False)
    df.to_csv(f"{output_prefix}_assignments.csv", index=False)

    # Save model
    with open(f"{output_prefix}_model.pkl", 'wb') as f:
        pickle.dump(topic_model, f)

    print('Saved topic info to', f"{output_prefix}_topic_info.csv")
    print('Saved assignments to', f"{output_prefix}_assignments.csv")
    print('Saved model to', f"{output_prefix}_model.pkl")

    return topic_model, topics_df, df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Feature CSV (with clean_review and compound sentiment)')
    parser.add_argument('--output-prefix', default='nlp_reviews_project/data/bertopic_output', help='Prefix for output files')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2')
    args = parser.parse_args()

    run_bertopic(args.input, args.output_prefix, embedding_model=args.embedding_model)

"""Run BERTopic on a sample subset for quick results."""
import pandas as pd
from bertopic_aspects import run_bertopic

INPUT = 'nlp_reviews_project/data/feature_reviews.csv'
OUT_PREFIX = 'nlp_reviews_project/data/bertopic_output_sample'

if __name__ == '__main__':
    df = pd.read_csv(INPUT, parse_dates=['time'], low_memory=False)
    sample = df.sample(n=min(1000, len(df)), random_state=42)
    tmp = 'nlp_reviews_project/data/_sample_reviews.csv'
    sample.to_csv(tmp, index=False)
    run_bertopic(tmp, OUT_PREFIX)

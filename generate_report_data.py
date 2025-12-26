"""generate_report_data.py
Helper script to generate data for the final report.
"""
import pandas as pd
from nlp_reviews_project.aspect_extraction import top_noun_phrases
from pathlib import Path

# Define the base directory of the project
BASE_DIR = Path(__file__).resolve().parent

def generate_aspects():
    """Load feature reviews and print top noun phrases for 1-star and 5-star reviews."""
    feature_file = BASE_DIR / 'data' / 'feature_reviews.csv'
    if not feature_file.exists():
        print(f"Error: {feature_file} not found. Please run the feature generation pipeline first.")
        return

    df = pd.read_csv(feature_file)

    df_1_star = df[df['rating'] == 1]
    df_5_star = df[df['rating'] == 5]

    print("--- Top Noun Phrases in 1-Star Reviews (Complaints) ---")
    top_1_star = top_noun_phrases(df_1_star['clean_review'], top_n=20)
    print(top_1_star)
    print("\n" + "="*50 + "\n")

    print("--- Top Noun Phrases in 5-Star Reviews (Praise) ---")
    top_5_star = top_noun_phrases(df_5_star['clean_review'], top_n=20)
    print(top_5_star)

if __name__ == '__main__':
    generate_aspects()


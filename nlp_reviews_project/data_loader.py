"""data_loader.py
Utilities to load and perform initial cleaning on the restaurant reviews CSV.
"""
import argparse
import os
import pandas as pd
from datetime import datetime
import re


def parse_metadata(meta):
    """Extract review_count and follower_count from the Metadata field."""
    if pd.isna(meta):
        return pd.NA, pd.NA
    # Examples: "1 Review , 2 Followers" or "3 Reviews , 190 Followers" or "1 Review" or "5 Reviews"
    try:
        # find numbers
        nums = re.findall(r"(\d+)", str(meta))
        if len(nums) == 0:
            return pd.NA, pd.NA
        if len(nums) == 1:
            return int(nums[0]), pd.NA
        return int(nums[0]), int(nums[1])
    except Exception:
        return pd.NA, pd.NA


def parse_time(ts):
    """Parse the Time column into pandas datetime. Handles multiple common formats."""
    if pd.isna(ts):
        return pd.NaT
    for fmt in ("%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M", "%m/%d/%Y"):
        try:
            return datetime.strptime(str(ts).strip(), fmt)
        except Exception:
            continue
    # Fallback to pandas
    try:
        return pd.to_datetime(ts, errors='coerce')
    except Exception:
        return pd.NaT


def load_and_clean(path):
    """Load CSV and perform initial cleaning and column normalization."""
    df = pd.read_csv(path, encoding='utf-8', engine='python')
    # Normalize columns
    df.columns = [c.strip() for c in df.columns]

    # Some files may have trailing unnamed cols; keep relevant ones
    wanted = ['Restaurant', 'Reviewer', 'Review', 'Rating', 'Metadata', 'Time']
    available = [c for c in wanted if c in df.columns]
    df = df[available].copy()

    # Drop exact-duplicate rows
    df = df.drop_duplicates()

    # Rename to standard names
    rename_map = {
        'Restaurant': 'restaurant',
        'Reviewer': 'reviewer',
        'Review': 'review',
        'Rating': 'rating',
        'Metadata': 'metadata',
        'Time': 'time'
    }
    df = df.rename(columns=rename_map)

    # Parse metadata
    df[['review_count', 'follower_count']] = df['metadata'].apply(lambda x: pd.Series(parse_metadata(x)))

    # Parse time
    df['time'] = df['time'].apply(parse_time)

    # Ensure rating numeric
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Basic review text cleanup: strip whitespace
    df['review'] = df['review'].astype(str).str.strip()

    return df


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to raw CSV file')
    parser.add_argument('--output', default='nlp_reviews_project/data/cleaned_reviews.csv', help='Path to save cleaned CSV')
    parsed = parser.parse_args(args=args)

    df = load_and_clean(parsed.input)

    outdir = os.path.dirname(parsed.output)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    df.to_csv(parsed.output, index=False)
    print(f"Saved cleaned data to {parsed.output}, {len(df)} rows")


if __name__ == '__main__':
    main()

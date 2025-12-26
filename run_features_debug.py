"""Run features with verbose debug output to capture progress and errors."""
import traceback
from pathlib import Path

# Define the base directory of the project, which is the 'nlp_reviews_project' folder
BASE_DIR = Path(__file__).resolve().parent

try:
    print('Importing modules...')
    from data_loader import load_and_clean
    from features import apply_features

    print('Loading CSV...')
    input_csv_path = BASE_DIR / 'Restaurant reviews.csv'
    df = load_and_clean(input_csv_path)
    print('Rows loaded:', len(df))

    print('Applying features (this may take a while)...')
    feat = apply_features(df)
    print('Feature dataframe shape:', feat.shape)

    output_csv_path = BASE_DIR / 'data' / 'feature_reviews.csv'
    # Create data directory if it doesn't exist
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    feat.to_csv(output_csv_path, index=False)
    print('Saved features to', output_csv_path)

except Exception as e:
    print('Error during feature run:')
    traceback.print_exc()

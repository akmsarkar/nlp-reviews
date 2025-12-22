"""Run features with verbose debug output to capture progress and errors."""
import traceback

try:
    print('Importing modules...')
    from data_loader import load_and_clean
    from features import apply_features
    print('Loading CSV...')
    df = load_and_clean('Restaurant reviews.csv')
    print('Rows loaded:', len(df))
    print('Applying features (this may take a while)...')
    feat = apply_features(df)
    print('Feature dataframe shape:', feat.shape)
    out = 'nlp_reviews_project/data/feature_reviews.csv'
    feat.to_csv(out, index=False)
    print('Saved features to', out)
except Exception as e:
    print('Error during feature run:')
    traceback.print_exc()

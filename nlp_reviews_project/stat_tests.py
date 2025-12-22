"""stat_tests.py
Compute summary statistics and statistical tests comparing 1-star vs 5-star reviews.
Outputs CSV: nlp_reviews_project/data/stat_tests_results.csv
"""
import pandas as pd
import numpy as np
from scipy import stats

INPUT = 'nlp_reviews_project/data/feature_reviews.csv'
OUTPUT = 'nlp_reviews_project/data/stat_tests_results.csv'

metrics = [
    'word_count','char_count','lexical_diversity','repetition_ratio',
    'adj_ratio','noun_ratio','verb_ratio','superlative_count','exclamation_count',
    'flesch_reading_ease','compound'
]


def cohen_d(a, b):
    a = np.array(a)
    b = np.array(b)
    na = len(a)
    nb = len(b)
    sa = a.std(ddof=1)
    sb = b.std(ddof=1)
    # pooled sd
    pooled = np.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2))
    if pooled == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled


def run():
    df = pd.read_csv(INPUT, parse_dates=['time'], low_memory=False)
    g1 = df[df['rating'] == 1]
    g5 = df[df['rating'] == 5]

    rows = []
    for m in metrics:
        a = g1[m].dropna()
        b = g5[m].dropna()
        # summary stats
        ra = {'group':'1-star','metric':m,'n':len(a),'mean':a.mean(),'std':a.std()}
        rb = {'group':'5-star','metric':m,'n':len(b),'mean':b.mean(),'std':b.std()}
        # t-test (Welch)
        try:
            tstat, p_t = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
        except Exception:
            tstat, p_t = np.nan, np.nan
        # Mann-Whitney U
        try:
            ustat, p_u = stats.mannwhitneyu(a, b, alternative='two-sided')
        except Exception:
            ustat, p_u = np.nan, np.nan
        # Effect size
        d = cohen_d(a, b)
        rows.append({
            'metric': m,
            'n_1': len(a), 'mean_1': a.mean(), 'std_1': a.std(),
            'n_5': len(b), 'mean_5': b.mean(), 'std_5': b.std(),
            't_stat': tstat, 'p_ttest': p_t,
            'u_stat': ustat, 'p_mwu': p_u,
            'cohen_d': d
        })

    res = pd.DataFrame(rows)
    res.to_csv(OUTPUT, index=False)
    print('Saved test results to', OUTPUT)
    print(res)


if __name__ == '__main__':
    run()

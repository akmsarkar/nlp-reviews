"""Streamlit EDA page
Loads feature and stat-test CSVs and provides interactive violin/box plots
comparing 1-star vs 5-star reviews with p-values and Cohen's d annotations.
Run with: `streamlit run streamlit_eda.py` from project root.
"""
from typing import Optional
import streamlit as st
import pandas as pd
import plotly.express as px
from scipy import stats
import numpy as np

st.set_page_config(layout='wide', page_title='EDA — 1-star vs 5-star')


@st.cache_data
def load_data(features_path: str, stats_path: str):
    df = pd.read_csv(features_path, parse_dates=['time'], low_memory=False)
    stt = pd.read_csv(stats_path)
    return df, stt


FEATURE_CSV = 'nlp_reviews_project/data/feature_reviews.csv'
STATS_CSV = 'nlp_reviews_project/data/stat_tests_results.csv'


def main():
    st.title('EDA: 1-star vs 5-star — Statistical Comparison')

    st.markdown('This page visualizes distributions for selected linguistic metrics and annotates plots with p-values and Cohen\'s d from precomputed tests.')

    df, stt = load_data(FEATURE_CSV, STATS_CSV)

    # Restaurant selector (per-restaurant toggle)
    restaurants = ['All'] + sorted(df['restaurant'].dropna().unique().tolist())
    sel_rest = st.sidebar.selectbox('Restaurant', restaurants)

    # Filter to ratings 1 and 5 and optionally by restaurant
    df_sub = df[df['rating'].isin([1, 5])].copy()
    if sel_rest != 'All':
        df_sub = df_sub[df_sub['restaurant'] == sel_rest]
        st.sidebar.write(f'Showing: {sel_rest} — {len(df_sub)} reviews')
        if len(df_sub) < 5:
            st.sidebar.warning('Selected restaurant has few reviews; statistics may be unstable.')

    df_sub['rating'] = df_sub['rating'].astype(int).astype(str)

    st.sidebar.header('Plot options')
    view_mode = st.sidebar.radio('View mode', ['Single restaurant', 'Multi-restaurant comparison'], index=0)
    # metrics selection
    metrics = st.sidebar.multiselect('Select metrics to plot', options=stt['metric'].tolist(), default=['word_count','lexical_diversity','compound'])
    show_table = st.sidebar.checkbox('Show stat-test table', value=True)

    # Multi-restaurant chooser (limit choices for performance)
    restaurants_all = sorted(df['restaurant'].dropna().unique().tolist())
    top_rest = df['restaurant'].value_counts().head(12).index.tolist()
    if view_mode == 'Multi-restaurant comparison':
        sel_multi = st.sidebar.multiselect('Select restaurants (max 6)', options=top_rest, default=top_rest[:4])
        if len(sel_multi) > 6:
            st.sidebar.warning('Showing up to 6 restaurants; only first 6 will be used for plotting.')

    if show_table:
        st.subheader('Statistical test results (1-star vs 5-star)')
        st.dataframe(stt.style.format({'p_ttest':'{:.2e}','cohen_d':'{:.3f}'}))

    for m in metrics:
        row = stt[stt['metric'] == m]
        p_val = None
        d = None
        if not row.empty:
            p_val = float(row.iloc[0]['p_ttest'])
            d = float(row.iloc[0]['cohen_d'])

        title = f"{m}"
        if p_val is not None and d is not None:
            title = f"{m} — t p={p_val:.2e} | cohen_d={d:.3f}"

        # Multi-restaurant comparison: facet by restaurant
        if view_mode == 'Multi-restaurant comparison':
            sel = sel_multi if 'sel_multi' in locals() else top_rest[:4]
            sel = sel[:6]
            if not sel:
                st.warning('No restaurants selected for multi-restaurant comparison.')
                continue
            df_facet = df[df['restaurant'].isin(sel) & df['rating'].isin([1,5])].copy()
            if df_facet.empty:
                st.warning('No data for selected restaurants/ratings.')
                continue
            # compute per-restaurant p-values and cohen's d and create labels
            def cohen_d(a, b):
                a = np.array(a)
                b = np.array(b)
                na = len(a)
                nb = len(b)
                sa = a.std(ddof=1) if na > 1 else 0.0
                sb = b.std(ddof=1) if nb > 1 else 0.0
                pooled = np.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2)) if (na + nb - 2) > 0 else 0.0
                if pooled == 0:
                    return np.nan
                return (a.mean() - b.mean()) / pooled

            labels = {}
            for r in sel:
                sub = df_facet[df_facet['restaurant'] == r]
                a = sub[sub['rating'] == 1][m].dropna()
                b = sub[sub['rating'] == 5][m].dropna()
                pval = np.nan
                dval = np.nan
                try:
                    if len(a) > 0 and len(b) > 0:
                        tstat, pval = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
                        dval = cohen_d(a, b)
                except Exception:
                    pval = np.nan; dval = np.nan
                labels[r] = f"{r}\n p={pval:.2e} d={np.nan if np.isnan(dval) else f'{dval:.2f}'}"

            df_facet['restaurant_label'] = df_facet['restaurant'].map(labels)
            hover_cols = [c for c in ['reviewer','word_count','char_count','compound'] if c in df_facet.columns]
            fig = px.violin(df_facet, x='rating', y=m, color='rating', facet_col='restaurant_label', box=True, points='all', title=title, hover_data=hover_cols)
            fig.update_traces(meanline_visible=True)
            fig.update_layout(legend_title_text='rating')
            st.plotly_chart(fig, use_container_width=True)
        else:
            hover_cols = [c for c in ['reviewer','word_count','char_count','compound'] if c in df_sub.columns]
            fig = px.violin(df_sub, x='rating', y=m, color='rating', box=True, points='all', title=title, hover_data=hover_cols)
            fig.update_traces(meanline_visible=True)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')
    st.subheader('Quick interpretation')
    st.markdown('- Large and significant differences in `compound` sentiment indicate very different valence between 1-star and 5-star reviews.')
    st.markdown('- POS-ratio differences (adjectives, verbs) point to stylistic differences; check effect sizes for magnitude.')
    st.markdown('- Use the sidebar to explore other metrics or show/hide the stat-test table.')


if __name__ == '__main__':
    main()

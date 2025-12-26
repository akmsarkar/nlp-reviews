"""streamlit_app.py
Lightweight Streamlit dashboard to compare 1-star vs 5-star reviews and explore pros/cons.
Run with: streamlit run streamlit_app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Define the base directory of the project
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(layout='wide', page_title='Restaurant Reviews NLP')

@st.cache_data
def load_data(path):
    return pd.read_csv(path, parse_dates=['time'], low_memory=False)

st.title('Restaurant Reviews — Linguistic Analysis')

# Integrate pages: Overview + EDA
page = st.sidebar.selectbox('Page', ['Overview', 'EDA — Statistical comparison'])

if page.startswith('EDA'):
    try:
        import streamlit_eda as eda
        eda.main()
    except Exception as e:
        st.error(f'Unable to load EDA page: {e}')
else:
    # Overview page
    data_path = st.text_input('Path to feature CSV', value=BASE_DIR / 'data' / 'feature_reviews.csv')
    if data_path:
        try:
            df = load_data(data_path)
        except Exception as e:
            st.error(f'Error loading data: {e}')
            st.stop()

        restaurants = ['All'] + sorted(df['restaurant'].dropna().unique().tolist())
        sel_rest = st.selectbox('Restaurant', restaurants)
        if sel_rest != 'All':
            df = df[df['restaurant'] == sel_rest]

        st.markdown('## Rating distribution')
        fig = px.histogram(df, x='rating', nbins=5)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('## 1-star vs 5-star comparison')
        col1, col2 = st.columns(2)
        with col1:
            s1 = df[df['rating'] == 1]
            st.metric('1-star count', len(s1))
            if not s1.empty:
                fig1 = px.box(s1, y='word_count', points='all', title='Word count (1-star)')
                st.plotly_chart(fig1, use_container_width=True)
        with col2:
            s5 = df[df['rating'] == 5]
            st.metric('5-star count', len(s5))
            if not s5.empty:
                fig5 = px.box(s5, y='word_count', points='all', title='Word count (5-star)')
                st.plotly_chart(fig5, use_container_width=True)

        st.markdown('## Top noun phrases (pros/cons)')
        if st.button('Show top noun phrases'):
            from aspect_extraction import top_noun_phrases
            top_all = top_noun_phrases(df['clean_review'], top_n=30)
            st.write(top_all.head(30))

        st.markdown('## Time series — average rating')
        if 'time' in df.columns:
            ts = df.set_index('time').resample('M').rating.mean().reset_index()
            figts = px.line(ts, x='time', y='rating', title='Monthly average rating')
            st.plotly_chart(figts, use_container_width=True)

        st.markdown('## Inspect raw examples')
        idx = st.number_input('Review index (0-based)', min_value=0, max_value=max(0, len(df) - 1), value=0)
        st.write(df.iloc[int(idx)][['restaurant', 'reviewer', 'rating', 'review']])

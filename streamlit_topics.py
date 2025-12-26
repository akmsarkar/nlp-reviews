"""Streamlit page for BERTopic pros/cons exploration.
Loads BERTopic outputs (topic info + assignments) and lets the user inspect topics,
see top words, average sentiment, and example positive/negative reviews per topic.
"""
import streamlit as st
import pandas as pd
import plotly.express as px

TOPIC_INFO = 'nlp_reviews_project/data/bertopic_output_sample_topic_info.csv'
ASSIGNMENTS = 'nlp_reviews_project/data/bertopic_output_sample_assignments.csv'


@st.cache_data
def load_topics(topic_info_path: str, assignments_path: str):
    ti = pd.read_csv(topic_info_path)
    assign = pd.read_csv(assignments_path, parse_dates=['time'], low_memory=False)
    return ti, assign


def main():
    st.title('Topics — Pros & Cons Explorer')
    try:
        ti, assign = load_topics(TOPIC_INFO, ASSIGNMENTS)
    except Exception as e:
        st.error('Topic outputs not found. Run BERTopic script first: ' + str(e))
        return

    st.sidebar.header('Topic options')
    restaurants = ['All'] + sorted(assign['restaurant'].dropna().unique().tolist())
    sel_rest = st.sidebar.selectbox('Restaurant', restaurants)

    topics = ti.sort_values('n_docs', ascending=False)
    topic_choices = topics['topic'].astype(str).tolist()
    sel_topic = st.sidebar.selectbox('Topic (by id)', ['All'] + topic_choices)

    df = assign.copy()
    if sel_rest != 'All':
        df = df[df['restaurant'] == sel_rest]

    st.subheader('Topic summary')
    # Convert numeric columns to strings for display to avoid Arrow conversion issues
    topics_display = topics.copy()
    topics_display['topic'] = topics_display['topic'].astype(str)
    topics_display['n_docs'] = topics_display['n_docs'].astype(str)
    topics_display['avg_compound'] = topics_display['avg_compound'].astype(str)
    st.dataframe(topics_display)

    # If specific topic selected, show details
    if sel_topic != 'All':
        tid = int(sel_topic)
        info = topics[topics['topic'] == tid]
        if info.empty:
            st.warning('Topic not found in topic info.')
        else:
            info = info.iloc[0]
            st.markdown(f"**Topic {tid} — top words:** {info['top_words']}")
            st.markdown(f"**Documents in topic:** {int(info['n_docs'])}")
            st.markdown(f"**Average sentiment (compound):** {float(info['avg_compound']):.3f}")

            sub = df[df['topic'] == tid].copy()
            if sub.empty:
                st.info('No reviews for this topic with current filters.')
                return

            st.markdown('### Top positive examples (Pros)')
            pos = sub.sort_values('compound', ascending=False).head(10)
            if not pos.empty:
                st.dataframe(pos[['restaurant','reviewer','rating','compound','review']].head(10))
            else:
                st.write('No positive examples')

            st.markdown('### Top negative examples (Cons)')
            neg = sub.sort_values('compound', ascending=True).head(10)
            if not neg.empty:
                st.dataframe(neg[['restaurant','reviewer','rating','compound','review']].head(10))
            else:
                st.write('No negative examples')

            # Topic frequency by rating
            freq = sub['rating'].value_counts().sort_index()
            fig = px.bar(x=freq.index.astype(str), y=freq.values, labels={'x':'rating','y':'count'}, title='Topic frequency by rating')
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown('Select a topic to view pros/cons and examples. You can filter by restaurant on the left.')


if __name__ == '__main__':
    main()

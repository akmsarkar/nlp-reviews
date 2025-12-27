# NLP Restaurant Reviews Analysis

This project analyzes a dataset of restaurant reviews to understand the linguistic differences between 1-star and 5-star ratings and extracts key aspects driving customer satisfaction or dissatisfaction.

## Project Overview

The goal is to answer:
1.  **Linguistic Differences**: How do 1-star and 5-star reviews differ in length, emotion, and word usage?
2.  **Key Aspects**: What specific features (Food, Service, Ambience) are mentioned most in positive vs. negative reviews?

## Installation & Setup

1.  **Install Dependencies**
    Ensure you have Python installed. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download NLTK Data**
    This project requires specific NLTK datasets. If you encounter a `LookupError` for `punkt_tab`, run the following command:
    ```bash
    python -m nltk.downloader punkt punkt_tab stopwords averaged_perceptron_tagger wordnet
    ```

## Running the Dashboard

To launch the interactive Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

## Key Insights

Based on the analysis of ~1,000 reviews:

*   **Linguistic Patterns**:
    *   **1-Star Reviews**: Significantly longer and more narrative-driven (more verbs). They tend to tell a story of what went wrong.
    *   **5-Star Reviews**: Shorter, concise, and packed with adjectives (e.g., "delicious", "great").
*   **Core Themes**:
    *   **Service**: The biggest differentiator. Rude/slow service drives 1-star reviews; friendly/prompt service drives 5-star reviews.
    *   **Food**: Always critical, but often secondary to service in negative reviews.
    *   **Ambience**: Frequently mentioned in positive experiences.

## Limitations

*   **Model Complexity**: Due to hardware resource constraints, heavy transformer-based models (like BERTopic) were excluded.
*   **Methodology**: We utilized dictionary-based approaches (VADER) and standard NLP techniques (N-grams, POS tagging) instead of deep learning models to ensure the application remains lightweight and responsive.
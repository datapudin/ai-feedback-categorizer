import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline
import time
import numpy as np
import io
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.set_page_config(page_title="AI Feedback Categorizer", layout="wide")
st.title("🧠 AI-Generated User Feedback Categorizer")

@st.cache_resource(show_spinner=False)
def load_models():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=-1)
    return classifier, sentiment_analyzer

classifier, sentiment_analyzer = load_models()

@st.cache_data(show_spinner=False)
def process_feedback(df, classifier, sentiment_analyzer):
    categories = ["Bug", "Feature Request", "Complaint", "Compliment", "General Feedback"]
    labels, sentiments, urgencies, summaries = [], [], [], []

    for feedback in df['feedback']:
        try:
            result = classifier(feedback, candidate_labels=categories)
            top_labels = [lbl for lbl, score in zip(result['labels'], result['scores']) if score > 0.3]
            labels.append(top_labels if top_labels else ['General Feedback'])

            sent_result = sentiment_analyzer(feedback)[0]
            sentiments.append(sent_result['label'])

            urgency = 'High' if ('bug' in [l.lower() for l in top_labels] and sent_result['label'] == 'NEGATIVE') else 'Low'
            urgencies.append(urgency)

            # Summarization skipped for performance, can be added if needed
            summaries.append(feedback[:120] + '...')

        except Exception as e:
            labels.append(['Uncategorized'])
            sentiments.append('Neutral')
            urgencies.append('Low')
            summaries.append(feedback[:120] + '...')

    df['category'] = labels
    df['sentiment'] = sentiments
    df['urgency'] = urgencies
    df['summary'] = summaries

    return df

def display_dashboard(df):
    st.write("### 📊 Feedback Analysis Dashboard")
    col1, col2 = st.columns([2, 1])

    # Category distribution
    all_categories = [cat for sublist in df['category'] for cat in sublist]
    cat_df = pd.DataFrame({'category': all_categories})
    fig, ax1 = plt.subplots()
    sns.countplot(y='category', data=cat_df, ax=ax1, order=cat_df['category'].value_counts().index, palette="viridis")
    ax1.set_title("Category Distribution")
    col1.pyplot(fig)

    # Sentiment distribution
    fig, ax2 = plt.subplots()
    sns.countplot(x='sentiment', data=df, palette="Set2", ax=ax2)
    ax2.set_title("Sentiment Overview")
    col2.pyplot(fig)

    # Urgency overview
    fig, ax3 = plt.subplots()
    sns.countplot(x='urgency', data=df, palette="Reds", ax=ax3)
    ax3.set_title("Urgency Breakdown")
    st.pyplot(fig)

    # WordCloud of keywords
    words = " ".join(df['feedback']).lower()
    words = " ".join([word for word in words.split() if word not in ENGLISH_STOP_WORDS])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
    st.image(wordcloud.to_array(), caption='Most Common Keywords')

    # Summary Stats
    st.markdown("### 📌 Business Insights")
    st.markdown(f"- Total Feedbacks: **{len(df)}**")
    st.markdown(f"- High Urgency Issues: **{sum(df['urgency'] == 'High')}**")
    st.markdown(f"- Positive Feedbacks: **{sum(df['sentiment'] == 'POSITIVE')}**")
    st.markdown(f"- Complaints Identified: **{sum(df['category'].apply(lambda x: 'Complaint' in x))}**")

    # Download report
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button("Download Categorized Feedback", csv_buffer.getvalue(), file_name="categorized_feedback.csv", mime="text/csv")

uploaded_file = st.file_uploader("Upload a CSV file with a 'feedback' column", type="csv")
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    if 'feedback' not in df_raw.columns:
        st.error("CSV must contain a column named 'feedback'.")
    else:
        with st.spinner("Processing feedback... this may take a while on CPU..."):
            df_processed = process_feedback(df_raw.copy(), classifier, sentiment_analyzer)
        display_dashboard(df_processed)
else:
    st.info("📥 Please upload a CSV file to begin analysis.")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
from io import StringIO

# Load pipelines
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentiment_analyzer = pipeline("sentiment-analysis")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Categories for classification
CATEGORIES = ["Bug", "Feature Request", "Complaint", "Compliment", "General"]

@st.cache_data(show_spinner=False)
def process_feedback(df):
    def classify(feedback):
        class_result = classifier(feedback, CATEGORIES, multi_label=True)
        assigned = [label for label, score in zip(class_result['labels'], class_result['scores']) if score > 0.5]
        if not assigned:
            assigned = ["General"]
        sentiment = sentiment_analyzer(feedback)[0]["label"]
        urgency = "High" if any(x in assigned for x in ["Bug", "Complaint"]) else "Medium" if "Feature Request" in assigned else "Low"
        summary = summarizer(feedback, max_length=30, min_length=5, do_sample=False)[0]['summary_text']
        return pd.Series([assigned, sentiment, urgency, summary])

    df[['categories', 'sentiment', 'urgency', 'summary']] = df['feedback'].apply(classify)
    vectorizer = CountVectorizer(stop_words='english', max_features=10)
    keywords = vectorizer.fit_transform(df['feedback'])
    top_keywords = vectorizer.get_feature_names_out().tolist()
    return df, top_keywords

# UI
st.title("AI-Generated User Feedback Categorizer")

# File Upload
uploaded_file = st.file_uploader("Upload CSV with feedback column", type=["csv"])
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    if 'review' in df_raw.columns:
        df_raw = df_raw.rename(columns={'review': 'feedback'})
    if 'feedback' not in df_raw.columns:
        st.error("CSV must have a 'feedback' or 'review' column.")
        st.stop()
    if 'timestamp' not in df_raw.columns:
        df_raw['timestamp'] = pd.date_range(start="2025-01-01", periods=len(df_raw), freq='D')
    df, keywords = process_feedback(df_raw.copy())

    # Filters
    st.sidebar.title("Filter Feedback")
    sentiment_filter = st.sidebar.multiselect("Sentiment", options=df['sentiment'].unique(), default=df['sentiment'].unique())
    urgency_filter = st.sidebar.multiselect("Urgency", options=df['urgency'].unique(), default=df['urgency'].unique())
    search_text = st.sidebar.text_input("Search text")

    filtered = df[
        df['sentiment'].isin(sentiment_filter) &
        df['urgency'].isin(urgency_filter) &
        df['feedback'].str.contains(search_text, case=False, na=False)
    ]

    st.subheader("Filtered Feedback")
    st.dataframe(filtered)

    # Alerts for urgent feedback
    critical = filtered[(filtered['sentiment'] == "NEGATIVE") & (filtered['urgency'] == "High")]
    if not critical.empty:
        st.warning(f"🚨 {len(critical)} urgent negative feedback(s) found!")

    # Distributions
    st.subheader("Category Distribution")
    all_cats = [c for sub in filtered['categories'] for c in sub]
    cat_df = pd.DataFrame({'category': all_cats})
    st.bar_chart(cat_df['category'].value_counts())

    st.subheader("Sentiment Distribution")
    st.bar_chart(filtered['sentiment'].value_counts())

    st.subheader("Urgency Distribution")
    st.bar_chart(filtered['urgency'].value_counts())

    # Trend over time
    st.subheader("Sentiment Over Time")
    filtered['date'] = pd.to_datetime(filtered['timestamp']).dt.date
    time_data = filtered.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
    st.line_chart(time_data)

    # Keywords
    st.subheader("Top Keywords")
    st.write(", ".join(keywords))

    # Feedback summary section
    st.subheader("Summarized Feedback")
    for _, row in filtered.iterrows():
        st.markdown(f"**Feedback:** {row['feedback']}")
        st.markdown(f"- **Summary:** {row['summary']}")
        st.markdown(f"- **Categories:** {row['categories']}")
        st.markdown(f"- **Sentiment:** {row['sentiment']}, **Urgency:** {row['urgency']}")
        st.markdown("---")

    # Export filtered
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered CSV", csv, "filtered_feedback.csv", "text/csv")

else:
    st.info("Please upload a CSV file to begin analysis.")
